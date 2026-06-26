"""Phase 8F: final Phase 8 acceptance gate.

Cross-cutting proof that the deterministic parser stack (8A–8E) is correct, deterministic, safe,
and correctly scoped. This file does NOT re-duplicate per-module tests for volume; it asserts the
whole-of-Phase-8 invariants in one place: the catalogue is the single source of truth, raw teams
survive until validation, vague time never becomes all-games, the validator stays the safety
boundary (invalid -> no execution), and the production parser stack imports nothing forbidden.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.intent_types import AMBIGUOUS_TEAM, SAME_TEAM_HEAD_TO_HEAD, UNKNOWN_TEAM
from src.intent_validator import validate_intent
from src.rule_parser import parse_rule_query
from src.rule_query_catalogue import (
    ALL_QUERY_EXAMPLES,
    SUPPORTED_QUERY_EXAMPLES,
    UNSUPPORTED_QUERY_EXAMPLES,
)
from src.tool_registry import DEFAULT_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- A. supported catalogue is the single source of truth -------------------

def test_all_supported_examples_parse_to_exact_intent() -> None:
    for ex in SUPPORTED_QUERY_EXAMPLES:
        res = parse_rule_query(ex.query)
        assert res.status == "parsed", (ex.query, res.to_dict())
        intent = res.parsed_intent
        assert intent.tool_name == ex.expected_tool
        assert dict(intent.arguments) == dict(ex.expected_arguments)
        assert intent.parser_mode == "rule"
        assert intent.confidence is None
        assert intent.raw_query == ex.query
        assert res.errors == ()
        json.dumps(res.to_dict())


# --- B. no unsupported example yields a parsed intent -----------------------

def test_no_unsupported_example_produces_parsed_intent() -> None:
    for ex in UNSUPPORTED_QUERY_EXAMPLES:
        res = parse_rule_query(ex.query)
        assert res.status != "parsed"
        assert res.parsed_intent is None
        assert res.errors
        if "compare" in ex.query.lower():
            assert res.status == "ambiguous"  # generic compare is never head_to_head


# --- C. determinism ---------------------------------------------------------

@pytest.mark.parametrize("query", [
    "How many points do the Warriors average over the last 5 games?",
    "How many points do GSW allow over the last 5 games?",
    "Celtics vs Heat head to head",
    "Top 5 scoring teams",
    "How many points do LA average?",
    "How many points do Celics average?",
    "Warriors average points recently",
    "Who is better?",
])
def test_determinism(query) -> None:
    first = parse_rule_query(query).to_dict()
    for _ in range(3):
        assert parse_rule_query(query).to_dict() == first


# --- D. raw teams preserved until validation --------------------------------

def test_raw_team_preserved_until_validation() -> None:
    assert dict(parse_rule_query("How many points do LA average?").parsed_intent.arguments) == {"team": "LA"}
    assert dict(parse_rule_query("How many points do Celics average?").parsed_intent.arguments) == {"team": "Celics"}
    assert dict(parse_rule_query("How many points do GSW average?").parsed_intent.arguments) == {"team": "GSW"}
    same = parse_rule_query("Celtics vs Celtics head to head")
    assert same.status == "parsed"  # parser does not reject same-team h2h
    assert dict(same.parsed_intent.arguments) == {"team_a": "Celtics", "team_b": "Celtics"}


# --- E. vague time never becomes all-games ----------------------------------

@pytest.mark.parametrize("query", [
    "Warriors average points recently", "Warriors last few games", "Show me recent Warriors games",
    "Warriors recent form", "Warriors average points of late",
])
def test_vague_time_never_all_games(query) -> None:
    res = parse_rule_query(query)
    assert res.status != "parsed"                      # never a valid all-games request
    assert res.parsed_intent is None
    if res.status == "incomplete":                     # when a metric routed, fail as incomplete
        assert "unsupported_time_expression" in [e.code for e in res.errors]


def test_compare_never_routes_to_h2h() -> None:
    for query in ("Compare Lakers and Celtics", "Compare LA teams", "Compare Celtics and Heat"):
        res = parse_rule_query(query)
        assert res.status in {"ambiguous", "no_parse"}
        assert res.parsed_intent is None


# --- F. parser -> validator boundary (validator protects) -------------------

@pytest.fixture(scope="module")
def env():
    from src.data_loader import load_raw_dataset
    from src.data_model import build_clean_view, validate_clean_view
    from src.data_validation import validate_dataset
    from src.validation_context import build_validation_context

    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return build_validation_context(clean, registry=DEFAULT_REGISTRY), clean


def _validate(query, context):
    res = parse_rule_query(query)
    assert res.status == "parsed"
    return validate_intent(res.parsed_intent, context=context)


def test_valid_outputs_validate_and_execute(env) -> None:
    context, clean = env
    vr = _validate("How many points do the Warriors average over the last 5 games?", context)
    assert vr.is_valid
    exec_result = DEFAULT_REGISTRY.execute(
        vr.validated_intent.tool_name, dict(vr.validated_intent.arguments), clean_df=clean)
    assert exec_result["status"] in {"ok", "no_data"}


@pytest.mark.parametrize("query,code", [
    ("How many points do LA average?", AMBIGUOUS_TEAM),
    ("How many points do Celics average?", UNKNOWN_TEAM),
    ("Celtics vs Celtics head to head", SAME_TEAM_HEAD_TO_HEAD),
])
def test_invalid_outputs_fail_validation(query, code, env) -> None:
    context, _ = env
    vr = _validate(query, context)
    assert not vr.is_valid
    assert code in [e.code for e in vr.errors]


def test_invalid_validation_blocks_execution(env, monkeypatch) -> None:
    context, clean = env
    calls = []
    monkeypatch.setattr(DEFAULT_REGISTRY, "execute", lambda *a, **k: calls.append(1))
    vr = _validate("How many points do LA average?", context)  # ambiguous -> invalid
    if vr.is_valid:  # guard: only execute valid intents
        DEFAULT_REGISTRY.execute(vr.validated_intent.tool_name,
                                 dict(vr.validated_intent.arguments), clean_df=clean)
    assert calls == []  # never executed an invalid intent


# --- G. consolidated production scope guard ---------------------------------

def test_production_parser_stack_imports_nothing_forbidden() -> None:
    code = (
        "import sys;"
        "import src.rule_parser, src.rule_slot_extractor, src.rule_intent_router,"
        " src.rule_query_normalisation, src.team_surface_catalogue, src.rule_query_catalogue,"
        " src.rule_parser_types;"
        "forbidden = ['pandas', 'src.data_loader', 'src.data_model', 'src.data_validation',"
        " 'src.tool_registry', 'src.tools', 'src.validation_context', 'src.intent_validator',"
        " 'src.team_resolution', 'src.llm_query_parser', 'src.response_formatter', 'src.assistant'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_no_orchestration_llm_or_web_modules_exist() -> None:
    for module in (
        "src.rule_parser_validation_integration", "src.parse_validate_execute",
        "src.llm_query_parser",
        "src.api", "src.web", "src.database", "src.rag", "src.agent", "src.server",
    ):
        assert importlib.util.find_spec(module) is None, f"{module} must not exist yet"


def test_all_query_examples_partition() -> None:
    assert ALL_QUERY_EXAMPLES == SUPPORTED_QUERY_EXAMPLES + UNSUPPORTED_QUERY_EXAMPLES
    queries = [ex.query for ex in ALL_QUERY_EXAMPLES]
    assert len(queries) == len(set(queries))  # no duplicate queries across the catalogue
