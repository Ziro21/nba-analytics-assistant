"""Adversarial robustness + cross-feature + fuzz (T2).

The core guarantee: the assistant FAILS CLOSED on any input — it always returns a structured
``AssistantResult`` with a valid status and a non-empty message, never raises, never hangs, never
fails open. These tests try hard to break that, then lock it in.
"""

from __future__ import annotations

import io
import json
import random
import time

import pytest

from src.assistant import answer_query
from src.assistant_runtime import build_default_runtime
from src.assistant_types import ASSISTANT_STATUSES
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view
from src.llm_query_parser import LLMQueryParser
from src.rule_parser import parse_rule_query
from src.tool_registry import DEFAULT_REGISTRY
from src.validation_context import build_validation_context


@pytest.fixture(scope="module")
def clean():
    return build_clean_view(load_raw_dataset())


@pytest.fixture(scope="module")
def context(clean):
    return build_validation_context(clean, registry=DEFAULT_REGISTRY)


@pytest.fixture(scope="module")
def runtime():
    return build_default_runtime()


def _is_valid_result(result) -> bool:
    return (result.status in ASSISTANT_STATUSES
            and isinstance(result.message, str) and bool(result.message))


ADVERSARIAL = [
    "", "   \t\n  ", "compare", "vs", "and and and and", "compare vs and head to head",
    "Warriors Lakers Celtics Heat Bucks compare vs and against",
    "last 999999999999 games warriors record", "top 0 teams", "top -5 teams",
    "top 99999 scoring teams", "warriors record last 0 games",
    "warriors record " * 500, "Compare " + "Warriors and " * 200 + "Celtics",
    "🏀🔥 warriors record 🏀", "Wàrríörs récörd", "ВАРРИОРС рекорд",
    "warriors\x00record", "warriors​record", "'; DROP TABLE teams; --",
    "{{7*7}} warriors {%raw%}", "%s%s%n warriors record", "../../../../etc/passwd warriors",
    "<script>alert(1)</script> celtics record", "WARRIORS!!!???... RECORD,,,",
    "compare LA and LA", "compare warriors and warriors", "team_average_points",
    "{'tool': 'team_record'}", "n" * 50000,
]


@pytest.mark.parametrize("query", ADVERSARIAL)
def test_adversarial_query_fails_closed(runtime, query) -> None:
    result = runtime.answer(query)
    assert _is_valid_result(result)


@pytest.mark.parametrize("bad", [None, 123, ["warriors", "record"], {"q": "x"}, b"bytes"])
def test_non_string_input_fails_closed(runtime, bad) -> None:
    # the assistant boundary must turn a non-string into a safe error, not a crash.
    result = runtime.answer(bad)
    assert _is_valid_result(result) and result.status == "error"


def test_parser_has_no_catastrophic_backtracking() -> None:
    # pathological long inputs must parse quickly (no ReDoS / quadratic blow-up). Generous bound.
    for query in ("last 5 games " * 8000, "Warriors " * 14000,
                  "compare vs record last 5 points allowed efficiency " * 5000):
        start = time.perf_counter()
        parse_rule_query(query)
        assert time.perf_counter() - start < 5.0


def test_fuzz_never_raises_and_returns_valid_result(runtime) -> None:
    random.seed(20260627)  # seeded for reproducibility
    vocab = ["compare", "vs", "and", "with", "record", "average", "points", "allowed",
             "efficiency", "head to head", "top", "scoring", "teams", "last", "games", "home",
             "away", "profile", "Warriors", "Celtics", "Lakers", "LA", "GSW", "Heat", "Knicks",
             "better", "bet", "5", "10", "0", "-3", "999999", "?", "!", "🏀", "ñ", "—", "\t", "the"]
    for _ in range(400):
        query = " ".join(random.choice(vocab) for _ in range(random.randint(0, 12)))
        result = runtime.answer(query)
        assert _is_valid_result(result), query


# --- cross-feature: the LLM seam reaches the same answer as the rule path -----

_TOOL_MATRIX = {
    "team_average_points": ("How many points do the Warriors average over the last 5 games?",
                            {"team": "Warriors", "window": 5}),
    "average_points_allowed": ("How many points do the Lakers allow at home?",
                               {"team": "Lakers", "location": "home"}),
    "team_record": ("What is the Boston Celtics away record?", {"team": "Boston", "location": "away"}),
    "top_scoring_teams": ("Top 7 scoring teams", {"n": 7}),
    "head_to_head": ("Celtics vs Heat head to head", {"team_a": "Celtics", "team_b": "Heat"}),
    "team_efficiency_summary": ("Warriors efficiency last 10 games", {"team": "Warriors", "window": 10}),
    "team_advanced_profile": ("How are the Warriors performing away over the last 5 games?",
                              {"team": "Warriors", "location": "away", "window": 5}),
    "compare_team_profiles": ("Compare Lakers and Knicks at home over the last 10 games",
                              {"team_a": "Lakers", "team_b": "Knicks", "location": "home", "window": 10}),
}


@pytest.mark.parametrize("tool,query_and_args", list(_TOOL_MATRIX.items()))
def test_llm_seam_matches_rule_path_per_tool(clean, context, tool, query_and_args) -> None:
    query, intent_args = query_and_args
    rule = answer_query(query, clean_df=clean, validation_context=context, registry=DEFAULT_REGISTRY)
    assert rule.status == "answer" and rule.tool_name == tool
    json.dumps(rule.to_dict())  # JSON always round-trips
    provider = (lambda payload: (lambda _p: json.dumps(payload)))(
        {"tool": tool, "arguments": intent_args})
    via_llm = answer_query(query, clean_df=clean, validation_context=context,
                           registry=DEFAULT_REGISTRY, parser=LLMQueryParser(provider))
    assert via_llm.status == "answer" and via_llm.tool_name == tool
    assert via_llm.data == rule.data  # identical computed result regardless of parser path


def test_every_result_renders_without_crashing(runtime) -> None:
    pytest.importorskip("rich")
    from rich.console import Console

    from src.rich_renderer import render_result
    queries = [q for q, _ in _TOOL_MATRIX.values()] + [
        "Compare LA and Celtics.", "Should I bet on Warriors?", "asdf qwer zxcv",
        "Compare GSW and Warriors.", "", "🏀", "n" * 1000,
    ]
    for query in queries:
        result = runtime.answer(query)
        render_result(result, console=Console(file=io.StringIO(), width=100))  # must not raise
