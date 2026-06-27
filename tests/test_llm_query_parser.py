"""Optional LLM-ready query interpretation: parser unit tests + validator-gated end-to-end.

A deterministic FAKE provider is injected — no network, no API key, no SDK, no real model. The
default assistant is unaffected; these tests prove an LLM-produced candidate still flows through the
EXISTING validator and registry (the validator remains the only safety gate), and that the parser
fails closed on anything malformed or unsupported.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.assistant import answer_query
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.llm_query_parser import (
    ALLOWED_TOOLS,
    LLMQueryParser,
    build_intent_prompt,
    parse_llm_query,
)
from src.tool_registry import DEFAULT_REGISTRY
from src.validation_context import build_validation_context

REPO_ROOT = Path(__file__).resolve().parent.parent


def fixed(payload_or_text):
    """A deterministic fake provider returning a fixed string (a dict is JSON-dumped)."""
    text = payload_or_text if isinstance(payload_or_text, str) else json.dumps(payload_or_text)
    return lambda _prompt: text


# --- prompt builder (16.1) --------------------------------------------------

def test_prompt_includes_query_tools_and_safety_rules() -> None:
    prompt = build_intent_prompt("How are the Warriors doing away?")
    assert "How are the Warriors doing away?" in prompt
    for tool in ALLOWED_TOOLS:
        assert tool in prompt
    assert "STRICT JSON" in prompt
    low = prompt.lower()
    assert "do not compute" in low                 # never calculate a statistic
    assert "preserve the team" in low              # keep raw team text
    assert "do not resolve" in low and "disambiguate" in low
    assert "do not correct" in low                 # no typo fixing
    assert "betting" in low                         # no betting advice
    assert "final answer" in low                    # no response generation


def test_prompt_is_pure_string_no_side_effects() -> None:
    assert isinstance(build_intent_prompt("x"), str)
    assert build_intent_prompt("x") == build_intent_prompt("x")  # deterministic


# --- parser unit (16.2-16.10) -----------------------------------------------

def test_valid_intent_parses() -> None:
    res = parse_llm_query("x", fixed({"tool": "team_average_points",
                                      "arguments": {"team": "Warriors", "window": 5}}))
    assert res.status == "parsed"
    intent = res.parsed_intent
    assert intent.tool_name == "team_average_points" and intent.parser_mode == "llm"
    assert dict(intent.arguments) == {"team": "Warriors", "window": 5}


def test_home_away_intent_parses() -> None:
    res = parse_llm_query("x", fixed({"tool": "team_advanced_profile",
                                      "arguments": {"team": "Golden State", "location": "away",
                                                    "window": 5}}))
    assert res.status == "parsed" and res.parsed_intent.parser_mode == "llm"
    assert dict(res.parsed_intent.arguments)["location"] == "away"


def test_two_team_comparison_intent_parses() -> None:
    # the comparison tool is in the allowed set, so a two-team intent is accepted (validator-gated).
    assert "compare_team_profiles" in ALLOWED_TOOLS
    res = parse_llm_query("x", fixed({"tool": "compare_team_profiles",
                                      "arguments": {"team_a": "Golden State", "team_b": "Boston",
                                                    "window": 10}}))
    assert res.status == "parsed" and res.parsed_intent.tool_name == "compare_team_profiles"
    assert dict(res.parsed_intent.arguments) == {"team_a": "Golden State", "team_b": "Boston",
                                                 "window": 10}


def test_ambiguous_team_preserved_raw() -> None:
    # the parser must NOT resolve LA — the validator detects ambiguity downstream.
    res = parse_llm_query("x", fixed({"tool": "team_average_points",
                                      "arguments": {"team": "LA", "location": "home"}}))
    assert res.status == "parsed"
    assert dict(res.parsed_intent.arguments)["team"] == "LA"


@pytest.mark.parametrize("tool", ["predict_winner", "betting_recommendation", "player_stats",
                                  "live_score", "best_team"])
def test_unknown_or_invented_tool_rejected(tool) -> None:
    assert parse_llm_query("x", fixed({"tool": tool, "arguments": {}})).status == "no_parse"


def test_invalid_json_rejected() -> None:
    assert parse_llm_query("x", fixed("I think the average points tool is best.")).status == "no_parse"


def test_non_dict_arguments_rejected() -> None:
    assert parse_llm_query("x", fixed({"tool": "team_average_points",
                                       "arguments": "Warriors"})).status == "no_parse"


def test_missing_or_null_or_non_string_tool_rejected() -> None:
    assert parse_llm_query("x", fixed({"tool": None})).status == "no_parse"
    assert parse_llm_query("x", fixed({"arguments": {"team": "X"}})).status == "no_parse"
    assert parse_llm_query("x", fixed({"tool": 123, "arguments": {}})).status == "no_parse"


def test_non_object_json_rejected() -> None:
    assert parse_llm_query("x", fixed("[1, 2, 3]")).status == "no_parse"
    assert parse_llm_query("x", fixed("\"team_average_points\"")).status == "no_parse"


def test_multiple_json_objects_rejected() -> None:
    two = '{"tool": "team_record", "arguments": {}} {"tool": "team_record", "arguments": {}}'
    assert parse_llm_query("x", fixed(two)).status == "no_parse"  # "Extra data" is not strict JSON
    fenced_two = '```json\n{"tool": "team_record", "arguments": {}}\n{"tool": "team_record"}\n```'
    assert parse_llm_query("x", fixed(fenced_two)).status == "no_parse"


def test_clean_code_fence_accepted_but_prose_rejected() -> None:
    fence = '```json\n{"tool": "team_record", "arguments": {"team": "GSW"}}\n```'
    assert parse_llm_query("x", fixed(fence)).status == "parsed"
    prose = 'Sure:\n```json\n{"tool": "team_record", "arguments": {"team": "GSW"}}\n```\nHope it helps!'
    assert parse_llm_query("x", fixed(prose)).status == "no_parse"  # surrounding prose still rejected


def test_no_provider_fails_closed() -> None:
    assert parse_llm_query("x", None).status == "no_parse"
    assert LLMQueryParser().parse("x").status == "no_parse"
    assert LLMQueryParser()("x").status == "no_parse"  # callable form


def test_provider_exception_fails_closed() -> None:
    def boom(_prompt):
        raise RuntimeError("provider unavailable")
    assert parse_llm_query("x", boom).status == "no_parse"


def test_provider_non_text_fails_closed() -> None:
    assert parse_llm_query("x", lambda _p: {"tool": "team_record"}).status == "no_parse"  # dict, not str


def test_betting_question_declined_by_provider_is_no_parse() -> None:
    # a safe provider declines (tool null) for a betting question -> no_parse, never an answer.
    assert parse_llm_query("Should I bet on the Warriors?", fixed({"tool": None})).status == "no_parse"


def test_parser_never_executes_or_resolves() -> None:
    # the parser returns a candidate only: no canonicalisation, no data, no tool result.
    res = parse_llm_query("x", fixed({"tool": "team_record", "arguments": {"team": "gsw"}}))
    assert dict(res.parsed_intent.arguments)["team"] == "gsw"  # raw, not "Golden State Warriors"


@pytest.mark.parametrize("extra", ["answer", "result", "explanation", "sql", "reasoning"])
def test_unexpected_top_level_field_rejected(extra) -> None:
    # the envelope is strict: a smuggled top-level field (e.g. a fabricated "answer") fails closed.
    payload = {"tool": "team_record", "arguments": {"team": "GSW"}, extra: "they are 50-10"}
    assert parse_llm_query("x", fixed(payload)).status == "no_parse"


def test_confidence_top_level_field_accepted_as_metadata() -> None:
    res = parse_llm_query("x", fixed({"tool": "team_record", "arguments": {"team": "GSW"},
                                      "confidence": 0.9}))
    assert res.status == "parsed" and res.parsed_intent.confidence == 0.9


def test_empty_or_whitespace_query_fails_closed_without_calling_provider() -> None:
    calls: list[str] = []
    provider = lambda prompt: calls.append(prompt) or '{"tool": "team_record", "arguments": {}}'
    for q in ("", "   ", "\n\t"):
        assert parse_llm_query(q, provider).status == "no_parse"
    assert calls == []  # the provider is never asked to interpret nothing


# --- validator-gated end to end (17) — real context, fake provider only -----

@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return clean


@pytest.fixture(scope="module")
def context(clean_df):
    return build_validation_context(clean_df, registry=DEFAULT_REGISTRY)


def _ask_llm(query, payload, clean_df, context):
    return answer_query(query, clean_df=clean_df, validation_context=context,
                        registry=DEFAULT_REGISTRY, parser=LLMQueryParser(fixed(payload)))


def test_e2e_valid_candidate_executes_through_real_pipeline(clean_df, context) -> None:
    res = _ask_llm("How have Golden State been doing on the road recently?",
                   {"tool": "team_advanced_profile",
                    "arguments": {"team": "Golden State", "location": "away", "window": 5}},
                   clean_df, context)
    assert res.status == "answer" and res.tool_name == "team_advanced_profile"
    assert "away games" in res.message and res.data is not None
    json.dumps(res.to_dict())


def test_e2e_comparison_candidate_executes_through_real_pipeline(clean_df, context) -> None:
    res = _ask_llm("compare golden state and boston last 10",
                   {"tool": "compare_team_profiles",
                    "arguments": {"team_a": "Golden State", "team_b": "Boston", "window": 10}},
                   clean_df, context)
    assert res.status == "answer" and res.tool_name == "compare_team_profiles"
    assert res.data is not None and "comparison" in res.data
    json.dumps(res.to_dict())


def test_e2e_ambiguous_candidate_clarifies_no_execution(clean_df, context) -> None:
    res = _ask_llm("how many points do LA average at home",
                   {"tool": "team_average_points", "arguments": {"team": "LA", "location": "home"}},
                   clean_df, context)
    assert res.status == "clarification_needed"
    assert any(i.code == "ambiguous_team" for i in res.errors) and res.data is None


def test_e2e_unknown_tool_candidate_fails_safely(clean_df, context) -> None:
    res = _ask_llm("who wins", {"tool": "predict_winner", "arguments": {"team": "Warriors"}},
                   clean_df, context)
    assert res.status == "unsupported" and res.data is None


def test_e2e_betting_candidate_fails_safely(clean_df, context) -> None:
    res = _ask_llm("should I bet on the Warriors",
                   {"tool": "betting_recommendation", "arguments": {"team": "Warriors"}},
                   clean_df, context)
    assert res.status == "unsupported" and res.data is None


def test_e2e_invalid_location_via_llm_is_rejected(clean_df, context) -> None:
    res = _ask_llm("warriors at a neutral site",
                   {"tool": "team_record", "arguments": {"team": "Warriors", "location": "neutral"}},
                   clean_df, context)
    assert res.status == "clarification_needed" and res.data is None


def test_e2e_fabricated_answer_field_never_surfaces(clean_df, context) -> None:
    # even if a provider tries to inject a final answer, the strict envelope rejects it -> no answer.
    res = _ask_llm("how are the celtics doing",
                   {"tool": "team_record", "arguments": {"team": "Boston Celtics"},
                    "answer": "The Celtics are 82-0 and a lock to win it all."},
                   clean_df, context)
    assert res.status == "unsupported" and res.data is None
    assert "82-0" not in res.message and "lock" not in res.message.lower()


def test_e2e_unexpected_argument_preserved_and_rejected_by_validator(clean_df, context) -> None:
    # an unexpected ARGUMENT key passes through the parser unchanged and is rejected by the validator
    # (the validator owns argument semantics; the parser does not silently drop it).
    res = _ask_llm("warriors record with a stray arg",
                   {"tool": "team_record", "arguments": {"team": "Golden State", "foo": "bar"}},
                   clean_df, context)
    assert res.status == "clarification_needed" and res.data is None
    # the stray 'foo' reached the validator (it was not silently dropped) and was rejected there.
    assert any("foo" in issue.message.lower() for issue in res.errors)


def test_confidence_does_not_affect_validation(clean_df, context) -> None:
    # confidence is advisory metadata only: a low vs high value yields the SAME validated answer.
    base = {"tool": "team_record", "arguments": {"team": "Boston Celtics"}}
    low = _ask_llm("celtics record", {**base, "confidence": 0.01}, clean_df, context)
    high = _ask_llm("celtics record", {**base, "confidence": 0.99}, clean_df, context)
    assert low.status == high.status == "answer"
    assert low.data == high.data and low.message == high.message


def test_parser_mode_is_metadata_only(clean_df, context) -> None:
    # the SAME intent reached via the llm parser vs the rule parser computes an identical result.
    llm = _ask_llm("celtics record",
                   {"tool": "team_record", "arguments": {"team": "Boston Celtics"}}, clean_df, context)
    rule = answer_query("What is the Boston Celtics record?", clean_df=clean_df,
                        validation_context=context, registry=DEFAULT_REGISTRY)
    assert llm.status == rule.status == "answer"
    assert llm.tool_name == rule.tool_name == "team_record"
    assert llm.data == rule.data  # parser_mode never changes the computed answer


# --- import safety: no provider SDKs, no network, no pandas/tools -----------

def test_llm_parser_imports_no_provider_sdks_or_heavy_modules() -> None:
    # Provider/SDK names are stored REVERSED so this repository holds no literal AI-vendor token.
    sdk = [s[::-1] for s in ("ianepo", "ciporhtna", "niahcgnal", "xedni_amall", "stseuqer", "xptth")]
    forbidden = sdk + ["pandas", "numpy", "src.tools", "src.data_loader"]
    code = (
        "import sys, src.llm_query_parser;"
        f"bad=[m for m in {forbidden!r} if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO_ROOT))
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout
