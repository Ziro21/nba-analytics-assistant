"""Phase 8C tests: raw slot extraction (no validation, no resolution, no execution)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.rule_parser_types import ParseError
from src.rule_query_catalogue import SUPPORTED_QUERY_EXAMPLES, UNSUPPORTED_QUERY_EXAMPLES
from src.rule_slot_extractor import (
    SLOT_STATUS_EXTRACTED,
    SLOT_STATUS_INCOMPLETE,
    SLOT_STATUS_UNSUPPORTED,
    SlotExtractionResult,
    extract_slots,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_MODULES = (
    "src.rule_parser",
    "src.rule_parser_validation_integration",
    "src.llm_query_parser",
    "src.response_formatter",
    "src.assistant",
)


def _args(query, tool):
    res = extract_slots(query, tool_name=tool)
    return res.status, (dict(res.arguments) if res.arguments is not None else None), \
        [e.code for e in res.errors]


# --- slot result contract ---------------------------------------------------

def test_extracted_result_contract() -> None:
    res = SlotExtractionResult.extracted({"team": "Warriors"}, raw_query="q", tool_name="team_record")
    assert res.status == SLOT_STATUS_EXTRACTED
    json.dumps(res.to_dict())
    with pytest.raises((TypeError, ValueError)):
        SlotExtractionResult.extracted({"team": "x"}, errors=(ParseError("missing_team", "m"),))
    with pytest.raises((TypeError, ValueError)):
        SlotExtractionResult(SLOT_STATUS_EXTRACTED, None)  # extracted needs an args dict


def test_non_extracted_result_contract() -> None:
    for ctor, status in ((SlotExtractionResult.incomplete, SLOT_STATUS_INCOMPLETE),
                         (SlotExtractionResult.unsupported, SLOT_STATUS_UNSUPPORTED)):
        res = ctor((ParseError("missing_team", "m"),), raw_query="q")
        assert res.status == status and res.arguments is None and res.errors
        json.dumps(res.to_dict())
    with pytest.raises((TypeError, ValueError)):
        SlotExtractionResult.incomplete(())  # needs at least one error
    with pytest.raises((TypeError, ValueError)):
        SlotExtractionResult("weird", {"team": "x"})


def test_extracted_arguments_are_immutable_and_isolated() -> None:
    res = SlotExtractionResult.extracted({"team": "Warriors"}, tool_name="team_record")
    with pytest.raises(TypeError):
        res.arguments["team"] = "Hacked"
    d = res.to_dict()
    d["arguments"]["team"] = "Hacked"
    d["matched_surfaces"].append("x")
    assert dict(res.arguments) == {"team": "Warriors"}


# --- number extraction ------------------------------------------------------

@pytest.mark.parametrize("query,tool,expected", [
    ("Warriors last 5 games", "team_record", 5),
    ("Warriors past 10 games", "team_record", 10),
    ("Warriors previous 3 games", "team_record", 3),
    ("Celtics vs Heat last 5 meetings", "head_to_head", 5),
    ("Boston over the last 7 games", "team_average_points", 7),
])
def test_window_extraction(query, tool, expected) -> None:
    status, args, _ = _args(query, tool)
    assert args.get("window") == expected


def test_top_n_extraction() -> None:
    assert _args("Top 5 scoring teams", "top_scoring_teams")[1] == {"n": 5}
    assert _args("top 10 teams by points", "top_scoring_teams")[1] == {"n": 10}


def test_top_scoring_without_number_omits_n() -> None:
    status, args, _ = _args("Top scoring teams", "top_scoring_teams")
    assert status == SLOT_STATUS_EXTRACTED and args == {}  # no invented n


def test_season_id_extraction() -> None:
    assert _args("Top scoring teams in season 26", "top_scoring_teams")[1] == {"season_id": 26}
    assert _args("season id 26 top teams", "top_scoring_teams")[1] == {"season_id": 26}


@pytest.mark.parametrize("query", [
    "Top teams in the 2023-24 season", "Top teams this season", "Top teams last season",
])
def test_unsupported_season_labels_not_parsed(query) -> None:
    _, args, _ = _args(query, "top_scoring_teams")
    assert "season_id" not in (args or {})


# --- vague time expressions -------------------------------------------------

@pytest.mark.parametrize("query,tool", [
    ("Warriors average points recently", "team_average_points"),
    ("Warriors recent form", "team_record"),
    ("Warriors last few games", "team_record"),
    ("Lakers record of late", "team_record"),
    ("Celtics efficiency latest games", "team_efficiency_summary"),
])
def test_vague_time_is_unsupported(query, tool) -> None:
    status, args, codes = _args(query, tool)
    assert status == SLOT_STATUS_INCOMPLETE
    assert "unsupported_time_expression" in codes
    assert args is None  # no invented window=5


# --- gazetteer team extraction ----------------------------------------------

@pytest.mark.parametrize("query,tool,team", [
    ("Average points for Boston Celtics last 10 games", "team_average_points", "Boston Celtics"),
    ("What is GSW averaging over the last 5 games?", "team_average_points", "GSW"),
    ("Warriors net rating", "team_efficiency_summary", "Warriors"),
    ("How many points do LA average?", "team_average_points", "LA"),
    ("Philadelphia 76ers record", "team_record", "Philadelphia 76ers"),
])
def test_gazetteer_team_extraction(query, tool, team) -> None:
    _, args, _ = _args(query, tool)
    assert args.get("team") == team  # raw span, not canonicalised


def test_h2h_gazetteer_extraction() -> None:
    _, args, _ = _args("Celtics vs Heat head to head", "head_to_head")
    assert args == {"team_a": "Celtics", "team_b": "Heat"}


def test_ambiguous_surface_extracted_raw_not_resolved() -> None:
    _, args, _ = _args("How many points do LA average?", "team_average_points")
    assert args == {"team": "LA"}  # parser does not resolve/decide ambiguity


# --- structural fallback for typos ------------------------------------------

@pytest.mark.parametrize("query", [
    "How many points do Celics average?",
    "What is Celics averaging over the last 5 games?",
])
def test_typo_fallback_extracts_raw_candidate(query) -> None:
    status, args, _ = _args(query, "team_average_points")
    assert status == SLOT_STATUS_EXTRACTED
    assert args.get("team") == "Celics"  # extraction only, never corrected
    res = extract_slots(query, tool_name="team_average_points")
    assert all(not e.suggestions for e in res.errors)  # no suggestions generated here


# --- must-not-extract negatives ---------------------------------------------

@pytest.mark.parametrize("query,tool", [
    ("How many points do teams average?", "team_average_points"),
    ("What is the average points?", "team_average_points"),
    ("Show me defensive rating", "team_efficiency_summary"),
    ("How many points average?", "team_average_points"),
    ("What is the record?", "team_record"),
])
def test_must_not_extract_fake_team(query, tool) -> None:
    status, args, codes = _args(query, tool)
    assert status == SLOT_STATUS_INCOMPLETE
    assert "missing_team" in codes
    assert args is None


# --- special/exhibition team phrases (Codex 8C finding) ----------------------

@pytest.mark.parametrize("query,team", [
    ("How many points do Team World average?", "Team World"),
    ("What is Team Stars record?", "Team Stars"),
    ("Team Stripes efficiency", "Team Stripes"),
])
def test_special_team_phrase_extracted_whole_not_partial(query, team) -> None:
    # Policy: extract the FULL special phrase raw (so the validator can reject it with
    # invalid_special_team), never a mangled partial like "World"/"Stars"/"Stripes".
    tool = "team_record" if "record" in query else (
        "team_efficiency_summary" if "efficiency" in query else "team_average_points")
    status, args, _ = _args(query, tool)
    assert status == SLOT_STATUS_EXTRACTED
    assert args.get("team") == team
    assert args.get("team") not in {"World", "Stars", "Stripes"}


def test_bare_special_component_does_not_leak_as_team() -> None:
    # A bare remnant (no "Team" prefix, so no surface match) must NOT become a fallback team.
    for query in ("How many points does World average?", "What is Stars record?"):
        tool = "team_record" if "record" in query else "team_average_points"
        status, args, codes = _args(query, tool)
        assert status == SLOT_STATUS_INCOMPLETE and "missing_team" in codes
        assert args is None


# --- two teams for a single-team tool (8C constraint from 8B) ----------------

def test_two_teams_single_tool_is_flagged_not_truncated() -> None:
    status, args, codes = _args("Compare Celtics and Heat record", "team_record")
    assert status == SLOT_STATUS_UNSUPPORTED
    assert "ambiguous_team_mention" in codes
    assert args is None  # never silently takes the first team


# --- head-to-head extraction ------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("Celtics vs Heat head to head", {"team_a": "Celtics", "team_b": "Heat"}),
    ("Boston against Miami", {"team_a": "Boston", "team_b": "Miami"}),
    ("How have the Celtics done against the Heat?", {"team_a": "Celtics", "team_b": "Heat"}),
    ("Celtics record against Heat", {"team_a": "Celtics", "team_b": "Heat"}),
    ("Celtics vs Heat last 5 meetings",
     {"team_a": "Celtics", "team_b": "Heat", "window": 5}),
])
def test_h2h_extraction(query, expected) -> None:
    status, args, _ = _args(query, "head_to_head")
    assert status == SLOT_STATUS_EXTRACTED and args == expected


def test_h2h_missing_opponent() -> None:
    status, args, codes = _args("Celtics vs", "head_to_head")
    assert status == SLOT_STATUS_INCOMPLETE and "missing_opponent" in codes


def test_h2h_missing_team() -> None:
    status, args, codes = _args("vs Heat", "head_to_head")
    assert status == SLOT_STATUS_INCOMPLETE and "missing_team" in codes


def test_h2h_same_team_not_rejected_here() -> None:
    # identical teams are the validator's job (same_team_head_to_head), not the parser's.
    status, args, _ = _args("Celtics vs Celtics", "head_to_head")
    assert status == SLOT_STATUS_EXTRACTED and args == {"team_a": "Celtics", "team_b": "Celtics"}


# --- catalogue-driven extraction --------------------------------------------

@pytest.mark.parametrize("example", SUPPORTED_QUERY_EXAMPLES, ids=lambda e: e.query)
def test_supported_examples_extract_expected_arguments(example) -> None:
    res = extract_slots(example.query, tool_name=example.expected_tool)
    assert res.status == SLOT_STATUS_EXTRACTED, res.to_dict()
    assert dict(res.arguments) == dict(example.expected_arguments)


def test_unsupported_vague_time_and_h2h_examples_surface_expected_codes() -> None:
    # Only the slot-relevant unsupported examples are meaningful at this phase.
    routing = {  # query -> (tool to extract under, expected code)
        "Warriors average points lately": ("team_average_points", "unsupported_time_expression"),
        "GSW points allowed recently": ("average_points_allowed", "unsupported_time_expression"),
        "Lakers record of late": ("team_record", "unsupported_time_expression"),
        "Celtics efficiency latest games": ("team_efficiency_summary", "unsupported_time_expression"),
        "Celtics vs": ("head_to_head", "missing_opponent"),
        "vs Heat": ("head_to_head", "missing_team"),
    }
    present = {ex.query for ex in UNSUPPORTED_QUERY_EXAMPLES}
    for query, (tool, code) in routing.items():
        assert query in present
        _, _, codes = _args(query, tool)
        assert code in codes


# --- unknown tool -----------------------------------------------------------

def test_unknown_tool_is_unsupported() -> None:
    res = extract_slots("Warriors record", tool_name="not_a_tool")
    assert res.status == SLOT_STATUS_UNSUPPORTED


# --- no validation / no execution -------------------------------------------

def test_extractor_does_not_import_validation_or_execution_modules() -> None:
    code = (
        "import sys; import src.rule_slot_extractor; extract = src.rule_slot_extractor.extract_slots;"
        "extract('Celtics vs Heat', tool_name='head_to_head');"
        "extract('How many points do Celics average?', tool_name='team_average_points');"
        "forbidden = ['pandas', 'src.data_loader', 'src.tool_registry', 'src.tools',"
        " 'src.validation_context', 'src.intent_validator', 'src.team_resolution',"
        " 'src.rule_parser', 'src.llm_query_parser', 'src.response_formatter', 'src.assistant'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --- import / scope safety --------------------------------------------------

def test_future_modules_absent() -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None
