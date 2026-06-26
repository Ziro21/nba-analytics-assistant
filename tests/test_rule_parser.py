"""Phase 8D tests: rule parser assembly (parse only — no validation/execution)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import src.rule_parser as rp
from src.intent_types import ParsedIntent
from src.rule_intent_router import IntentRouteResult
from src.rule_parser import parse_rule_query
from src.rule_parser_types import ParseError, ParseWarning, RuleParseResult
from src.rule_query_catalogue import SUPPORTED_QUERY_EXAMPLES, UNSUPPORTED_QUERY_EXAMPLES
from src.rule_slot_extractor import SlotExtractionResult

REPO_ROOT = Path(__file__).resolve().parent.parent

# Explicit routing-vs-parse classification of the Phase 8A unsupported corpus, by the ACTUAL
# deterministic parser output. Vague time yields unsupported_time_expression ONLY when a metric
# routes the query; a vague phrase with no metric cannot route, so it is no_parse/unsupported_query.
NO_PARSE_QUERIES = {
    "Who is better?", "Tell me about Boston", "What happened last night?", "Top teams",
    "Warriors recent form", "Warriors last few games", "Show me recent Warriors games",
}
AMBIGUOUS_QUERIES = {"Compare Lakers and Celtics", "Compare LA teams"}
INCOMPLETE_QUERY_CODES = {
    "Warriors average points lately": "unsupported_time_expression",
    "GSW points allowed recently": "unsupported_time_expression",
    "Lakers record of late": "unsupported_time_expression",
    "Celtics efficiency latest games": "unsupported_time_expression",
    "Celtics vs": "missing_opponent",
    "vs Heat": "missing_team",
}


# --- parser function contract -----------------------------------------------

def test_parse_rule_query_exists_and_returns_result() -> None:
    assert callable(parse_rule_query)
    assert isinstance(parse_rule_query("What is the Warriors record?"), RuleParseResult)


@pytest.mark.parametrize("bad", [None, 123, ["q"], {"q": 1}, b"bytes"])
def test_non_string_raises_typeerror(bad) -> None:
    with pytest.raises(TypeError):
        parse_rule_query(bad)


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_empty_or_whitespace_is_no_parse_empty_query(query) -> None:
    res = parse_rule_query(query)
    assert res.status == "no_parse" and res.parsed_intent is None
    assert "empty_query" in [e.code for e in res.errors]


def test_result_to_dict_json_serialisable() -> None:
    for query in ("How many points do the Warriors average over the last 5 games?", "Who is better?"):
        json.dumps(parse_rule_query(query).to_dict())


# --- supported catalogue parsing --------------------------------------------

@pytest.mark.parametrize("example", SUPPORTED_QUERY_EXAMPLES, ids=lambda e: e.query)
def test_supported_examples_parse(example) -> None:
    res = parse_rule_query(example.query)
    assert res.status == "parsed", res.to_dict()
    intent = res.parsed_intent
    assert intent is not None
    assert intent.tool_name == example.expected_tool
    assert dict(intent.arguments) == dict(example.expected_arguments)
    assert intent.parser_mode == "rule"
    assert intent.raw_query == example.query
    assert intent.confidence is None
    assert res.errors == ()


# --- unsupported catalogue parsing ------------------------------------------

def test_unsupported_classification_covers_corpus() -> None:
    classified = NO_PARSE_QUERIES | AMBIGUOUS_QUERIES | set(INCOMPLETE_QUERY_CODES)
    assert {ex.query for ex in UNSUPPORTED_QUERY_EXAMPLES} == classified


def test_no_unsupported_example_parses() -> None:
    for ex in UNSUPPORTED_QUERY_EXAMPLES:
        res = parse_rule_query(ex.query)
        assert res.status != "parsed"
        assert res.parsed_intent is None
        assert res.errors  # every failure explains itself


@pytest.mark.parametrize("query", sorted(NO_PARSE_QUERIES))
def test_broad_and_metricless_vague_are_no_parse(query) -> None:
    assert parse_rule_query(query).status == "no_parse"


@pytest.mark.parametrize("query", sorted(AMBIGUOUS_QUERIES))
def test_generic_compare_is_ambiguous_not_h2h(query) -> None:
    res = parse_rule_query(query)
    assert res.status == "ambiguous"
    assert res.parsed_intent is None  # explicitly not head_to_head
    assert "ambiguous_intent" in [e.code for e in res.errors]


@pytest.mark.parametrize("query,code", sorted(INCOMPLETE_QUERY_CODES.items()))
def test_incomplete_examples(query, code) -> None:
    res = parse_rule_query(query)
    assert res.status == "incomplete"
    assert code in [e.code for e in res.errors]
    assert res.parsed_intent is None


def test_vague_time_never_becomes_all_games() -> None:
    # metric present -> incomplete (no invented window); metric absent -> no_parse.
    assert parse_rule_query("Warriors average points recently").status == "incomplete"
    assert parse_rule_query("Warriors recent form").status == "no_parse"


# --- error propagation from router ------------------------------------------

def _spy_extract(calls):
    def _fn(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("extract_slots must not be called when routing fails")
    return _fn


def test_no_route_maps_to_no_parse_without_extracting(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(rp, "route_intent", lambda q: IntentRouteResult.no_route(
        (ParseError("unsupported_query", "x"),), raw_query=q, normalised_query=q))
    monkeypatch.setattr(rp, "extract_slots", _spy_extract(calls))
    res = parse_rule_query("anything")
    assert res.status == "no_parse" and not calls
    assert "unsupported_query" in [e.code for e in res.errors]


def test_ambiguous_route_maps_to_ambiguous_without_extracting(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(rp, "route_intent", lambda q: IntentRouteResult.ambiguous(
        (ParseError("ambiguous_intent", "x"),), raw_query=q, normalised_query=q))
    monkeypatch.setattr(rp, "extract_slots", _spy_extract(calls))
    res = parse_rule_query("compare things")
    assert res.status == "ambiguous" and not calls


# --- error propagation from slot extractor ----------------------------------

def _route_to(tool):
    return lambda q: IntentRouteResult.routed(tool, raw_query=q, normalised_query=q,
                                              matched_signals=("x",))


def test_slot_incomplete_maps_to_incomplete(monkeypatch) -> None:
    monkeypatch.setattr(rp, "route_intent", _route_to("team_record"))
    monkeypatch.setattr(rp, "extract_slots", lambda q, *, tool_name: SlotExtractionResult.incomplete(
        (ParseError("missing_team", "x"),), raw_query=q, tool_name=tool_name))
    res = parse_rule_query("...")
    assert res.status == "incomplete" and res.parsed_intent is None


def test_slot_unsupported_maps_to_ambiguous(monkeypatch) -> None:
    monkeypatch.setattr(rp, "route_intent", _route_to("team_record"))
    monkeypatch.setattr(rp, "extract_slots", lambda q, *, tool_name: SlotExtractionResult.unsupported(
        (ParseError("ambiguous_team_mention", "x"),), raw_query=q, tool_name=tool_name))
    res = parse_rule_query("...")
    assert res.status == "ambiguous" and res.parsed_intent is None


def test_slot_extracted_maps_to_parsed(monkeypatch) -> None:
    monkeypatch.setattr(rp, "route_intent", _route_to("team_record"))
    monkeypatch.setattr(rp, "extract_slots", lambda q, *, tool_name: SlotExtractionResult.extracted(
        {"team": "Warriors"}, raw_query=q, tool_name=tool_name))
    res = parse_rule_query("...")
    assert res.status == "parsed"
    assert res.parsed_intent.tool_name == "team_record"
    assert dict(res.parsed_intent.arguments) == {"team": "Warriors"}


def test_parser_propagates_route_and_slot_warnings(monkeypatch) -> None:
    monkeypatch.setattr(rp, "route_intent", lambda q: IntentRouteResult.routed(
        "team_record", raw_query=q, normalised_query=q, matched_signals=("x",),
        warnings=(ParseWarning("route_warn", "rw"),)))
    monkeypatch.setattr(rp, "extract_slots", lambda q, *, tool_name: SlotExtractionResult.extracted(
        {"team": "Warriors"}, raw_query=q, tool_name=tool_name,
        warnings=(ParseWarning("slot_warn", "sw"),)))
    res = parse_rule_query("...")
    assert res.status == "parsed"
    assert {w.code for w in res.warnings} == {"route_warn", "slot_warn"}  # neither dropped


def test_unexpected_internal_status_maps_safely(monkeypatch) -> None:
    import types as _types
    # A bogus route status (the real dataclass forbids it) must fail safe, not crash/fall through.
    monkeypatch.setattr(rp, "route_intent",
                        lambda q: _types.SimpleNamespace(status="weird", errors=(), warnings=()))
    res = parse_rule_query("x")
    assert res.status == "no_parse" and res.parsed_intent is None

    # A bogus slot status, under a real routed result, must also fail safe.
    monkeypatch.setattr(rp, "route_intent", _route_to("team_record"))
    monkeypatch.setattr(rp, "extract_slots", lambda q, *, tool_name: _types.SimpleNamespace(
        status="weird", errors=(), warnings=(), arguments=None))
    res = parse_rule_query("x")
    assert res.status == "no_parse" and res.parsed_intent is None


# --- no hidden canonicalisation ---------------------------------------------

def test_no_canonicalisation_or_correction() -> None:
    la = parse_rule_query("How many points do LA average?")
    assert dict(la.parsed_intent.arguments) == {"team": "LA"}  # not Lakers/Clippers
    celics = parse_rule_query("How many points do Celics average?")
    assert dict(celics.parsed_intent.arguments) == {"team": "Celics"}  # not corrected


def test_same_team_h2h_parses_here() -> None:
    # parser builds the intent; same-team rejection is the Phase 7 validator's job.
    res = parse_rule_query("Celtics vs Celtics head to head")
    assert res.status == "parsed"
    assert dict(res.parsed_intent.arguments) == {"team_a": "Celtics", "team_b": "Celtics"}


@pytest.mark.parametrize("query,team", [
    ("How many points do Team World average?", "Team World"),
    ("What is Team Stars record?", "Team Stars"),
    ("Team Stripes net rating", "Team Stripes"),
])
def test_special_team_phrase_preserved_for_validator(query, team) -> None:
    # The full special phrase reaches the validator (invalid_special_team), never a partial.
    res = parse_rule_query(query)
    assert res.status == "parsed"
    assert res.parsed_intent.arguments.get("team") == team


# --- determinism ------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "How many points do the Warriors average over the last 5 games?",
    "Compare Lakers and Celtics", "Celtics vs", "Who is better?",
])
def test_determinism(query) -> None:
    first = parse_rule_query(query).to_dict()
    for _ in range(3):
        assert parse_rule_query(query).to_dict() == first


# --- no validation / execution; import safety -------------------------------

def test_parsed_intent_arguments_isolated_from_slot_result() -> None:
    # mutating the parsed result's dict view must not bleed anywhere.
    res = parse_rule_query("What is the Warriors record?")
    d = res.to_dict()
    d["parsed_intent"]["arguments"]["team"] = "Hacked"
    assert dict(res.parsed_intent.arguments) == {"team": "Warriors"}


def test_import_is_lightweight_and_no_validation_execution() -> None:
    code = (
        "import sys; import src.rule_parser as rp;"
        "rp.parse_rule_query('How many points do the Warriors average over the last 5 games?');"
        "rp.parse_rule_query('Who is better?');"
        "forbidden = ['pandas', 'src.data_loader', 'src.tool_registry', 'src.tools',"
        " 'src.validation_context', 'src.team_resolution', 'src.intent_validator',"
        " 'src.llm_query_parser', 'src.response_formatter', 'src.assistant'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_future_modules_absent() -> None:
    for module in ("src.rule_parser_validation_integration", "src.llm_query_parser",
                   "src.assistant"):
        assert importlib.util.find_spec(module) is None
