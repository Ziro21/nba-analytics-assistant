"""Phase 8B tests: deterministic intent routing (no slot extraction)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.rule_intent_router import (
    ROUTABLE_TOOL_NAMES,
    ROUTE_STATUS_AMBIGUOUS,
    ROUTE_STATUS_NO_ROUTE,
    ROUTE_STATUS_ROUTED,
    IntentRouteResult,
    route_intent,
)
from src.rule_parser_types import ParseError
from src.rule_query_catalogue import SUPPORTED_QUERY_EXAMPLES, SUPPORTED_TOOL_NAMES, UNSUPPORTED_QUERY_EXAMPLES

REPO_ROOT = Path(__file__).resolve().parent.parent

SLOT_KEYS = ("team", "team_a", "team_b", "window", "n", "season_id")

FORBIDDEN_MODULES = (  # 8C/8D legitimately added the catalogue, slot extractor, and parser
    "src.rule_parser_validation_integration",
    "src.llm_query_parser",
)

# Routing-vs-parse distinction for the Phase 8A UNSUPPORTED corpus.
# Routing detects INTENT only; missing operands / vague time are caught later in 8C/8D.
NO_ROUTE_QUERIES = {
    "Who is better?",
    "Tell me about Boston",
    "What happened last night?",
    "Top teams",                    # no "scoring" -> not a ranking
    "Warriors recent form",         # no metric keyword
    "Warriors last few games",      # no metric keyword
    "Show me recent Warriors games",  # no metric keyword
    "Warriors advanced profile at home",  # location split -> unsupported
    "Lakers record away",                 # location split -> unsupported
}
AMBIGUOUS_QUERIES = {
    "Compare Lakers and Celtics",
    "Compare LA teams",
}
# Clear intent now, but fail later in 8C/8D (vague time has no window; h2h missing an operand).
ROUTES_BUT_FAILS_LATER = {
    "Warriors average points lately": "team_average_points",
    "GSW points allowed recently": "average_points_allowed",
    "Lakers record of late": "team_record",
    "Celtics efficiency latest games": "team_efficiency_summary",
    "Celtics vs": "head_to_head",
    "vs Heat": "head_to_head",
}


def _error() -> ParseError:
    return ParseError(code="unsupported_query", message="x")


# --- Route result contract --------------------------------------------------

def test_routed_result_contract() -> None:
    res = IntentRouteResult.routed("team_record", raw_query="r", normalised_query="r",
                                   matched_signals=("record",))
    assert res.status == ROUTE_STATUS_ROUTED and res.tool_name == "team_record"
    json.dumps(res.to_dict())
    with pytest.raises((TypeError, ValueError)):
        IntentRouteResult(ROUTE_STATUS_ROUTED, "not_a_tool")  # invalid tool
    with pytest.raises((TypeError, ValueError)):
        IntentRouteResult(ROUTE_STATUS_ROUTED, "team_record", (_error(),))  # routed + errors


@pytest.mark.parametrize("ctor,status", [
    (IntentRouteResult.no_route, ROUTE_STATUS_NO_ROUTE),
    (IntentRouteResult.ambiguous, ROUTE_STATUS_AMBIGUOUS),
])
def test_non_routed_result_contract(ctor, status) -> None:
    res = ctor((_error(),), raw_query="q", normalised_query="q")
    assert res.status == status and res.tool_name is None and res.errors
    json.dumps(res.to_dict())


def test_invalid_status_and_missing_pieces_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        IntentRouteResult("weird", None, (_error(),))
    with pytest.raises((TypeError, ValueError)):
        IntentRouteResult(ROUTE_STATUS_NO_ROUTE, "team_record", (_error(),))  # non-routed + tool
    with pytest.raises((TypeError, ValueError)):
        IntentRouteResult(ROUTE_STATUS_NO_ROUTE, None, ())  # non-routed + no errors


def test_routable_tool_names_match_catalogue() -> None:
    assert set(ROUTABLE_TOOL_NAMES) == set(SUPPORTED_TOOL_NAMES)
    assert len(ROUTABLE_TOOL_NAMES) == 7


def test_route_result_to_dict_mutation_does_not_affect_object() -> None:
    routed = route_intent("What is the Warriors record?")
    d = routed.to_dict()
    d["matched_signals"].append("y")
    d["status"] = "hacked"
    assert routed.matched_signals == ("record",) and routed.status == ROUTE_STATUS_ROUTED

    failed = route_intent("Who is better?")
    d2 = failed.to_dict()
    d2["errors"].append("x")
    assert len(failed.errors) == 1 and failed.status == ROUTE_STATUS_NO_ROUTE


# --- Supported catalogue routing --------------------------------------------

@pytest.mark.parametrize("example", SUPPORTED_QUERY_EXAMPLES, ids=lambda e: e.query)
def test_supported_queries_route_to_expected_tool(example) -> None:
    res = route_intent(example.query)
    assert res.status == ROUTE_STATUS_ROUTED, res.to_dict()
    assert res.tool_name == example.expected_tool
    assert res.matched_signals  # the route is explained by at least one signal


def test_supported_routing_extracts_no_slots() -> None:
    for example in SUPPORTED_QUERY_EXAMPLES:
        d = route_intent(example.query).to_dict()
        assert set(d) == {
            "status", "tool_name", "errors", "warnings",
            "raw_query", "normalised_query", "matched_signals",
        }
        for slot in SLOT_KEYS:
            assert slot not in d


# --- Unsupported catalogue routing (routing-vs-parse partition) -------------

def test_unsupported_partition_covers_whole_corpus() -> None:
    classified = NO_ROUTE_QUERIES | AMBIGUOUS_QUERIES | set(ROUTES_BUT_FAILS_LATER)
    assert {ex.query for ex in UNSUPPORTED_QUERY_EXAMPLES} == classified


@pytest.mark.parametrize("query", sorted(NO_ROUTE_QUERIES))
def test_no_route_queries(query) -> None:
    res = route_intent(query)
    assert res.status == ROUTE_STATUS_NO_ROUTE
    assert res.tool_name is None


@pytest.mark.parametrize("query", sorted(AMBIGUOUS_QUERIES))
def test_compare_queries_are_ambiguous_not_h2h(query) -> None:
    res = route_intent(query)
    assert res.status == ROUTE_STATUS_AMBIGUOUS
    assert res.tool_name is None  # explicitly NOT head_to_head


@pytest.mark.parametrize("query,tool", sorted(ROUTES_BUT_FAILS_LATER.items()))
def test_clear_intent_routes_even_if_incomplete(query, tool) -> None:
    res = route_intent(query)
    assert res.status == ROUTE_STATUS_ROUTED
    assert res.tool_name == tool


# --- Priority tests ---------------------------------------------------------

@pytest.mark.parametrize("query,tool", [
    ("Celtics record against Heat", "head_to_head"),       # against (h2h) beats record
    ("How many points do GSW allow?", "average_points_allowed"),  # allowed beats average
    ("Celtics offensive rating and defensive rating", "team_efficiency_summary"),
    ("Top 5 scoring teams", "top_scoring_teams"),
    ("How many points do Warriors average?", "team_average_points"),
    ("Warriors points against last 5 games", "average_points_allowed"),  # not hijacked to h2h
])
def test_priority_routing(query, tool) -> None:
    res = route_intent(query)
    assert res.status == ROUTE_STATUS_ROUTED
    assert res.tool_name == tool


def test_points_conceded_routes_to_average_points_allowed() -> None:
    for query in ("How many points have Lakers conceded?", "Warriors points conceded",
                  "How many points do the Lakers concede?"):
        res = route_intent(query)
        assert res.status == ROUTE_STATUS_ROUTED
        assert res.tool_name == "average_points_allowed"


def test_compare_with_metric_policy_is_explicit() -> None:
    # Locked policy: a supported metric overrides the bare-compare ambiguity at routing.
    # 8B cannot see two teams; Phase 8C must reject two teams supplied to a single-team tool.
    res = route_intent("Compare Celtics and Heat record")
    assert res.status == ROUTE_STATUS_ROUTED
    assert res.tool_name == "team_record"


# --- Ambiguity / unsupported tests ------------------------------------------

@pytest.mark.parametrize("query", ["Compare Lakers and Celtics", "Compare LA teams", "Compare Celtics and Heat"])
def test_compare_never_routes_to_h2h(query) -> None:
    res = route_intent(query)
    assert res.tool_name != "head_to_head"
    assert res.status in {ROUTE_STATUS_AMBIGUOUS, ROUTE_STATUS_NO_ROUTE}


@pytest.mark.parametrize("query", ["Who is better?", "Tell me about Boston"])
def test_unsupported_do_not_route(query) -> None:
    assert route_intent(query).status == ROUTE_STATUS_NO_ROUTE


def test_empty_query_no_route() -> None:
    for q in ("", "   ", "?!"):
        res = route_intent(q)
        assert res.status == ROUTE_STATUS_NO_ROUTE
        assert res.errors[0].code in {"empty_query", "unsupported_query"}
    assert route_intent("").errors[0].code == "empty_query"


# --- No slot extraction -----------------------------------------------------

def test_route_result_carries_no_parsed_intent_attr() -> None:
    res = route_intent("What is the Warriors record?")
    assert not hasattr(res, "parsed_intent")
    assert not hasattr(res, "arguments")


# --- Import / scope safety --------------------------------------------------

def test_forbidden_modules_absent() -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


def test_router_import_is_lightweight() -> None:
    code = (
        "import sys; import src.rule_intent_router;"
        "forbidden = ['pandas', 'src.data_loader', 'src.tool_registry', 'src.tools',"
        " 'src.validation_context', 'src.team_resolution', 'src.intent_validator',"
        " 'src.rule_slot_extractor', 'src.team_surface_catalogue', 'src.rule_parser'];"
        "assert not any(m in sys.modules for m in forbidden), [m for m in forbidden if m in sys.modules];"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
