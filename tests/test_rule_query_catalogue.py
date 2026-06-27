"""Phase 8A tests: the executable rule-query catalogue (static data only)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import src.rule_query_catalogue as catalogue
from src.rule_parser_types import PARSE_ERROR_CODES, PARSE_STATUS_PARSED
from src.rule_query_catalogue import (
    ALL_QUERY_EXAMPLES,
    SUPPORTED_QUERY_EXAMPLES,
    SUPPORTED_TOOL_NAMES,
    UNSUPPORTED_QUERY_EXAMPLES,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

REGISTERED_TOOLS = {
    "team_average_points", "average_points_allowed", "team_record",
    "top_scoring_teams", "head_to_head", "team_efficiency_summary",
    "team_advanced_profile", "compare_team_profiles",
}
PARSER_FUNCTION_NAMES = ("parse_rule_query", "route_intent", "extract_slots", "normalise_query")


# --- Catalogue structure ----------------------------------------------------

def test_supported_tool_names() -> None:
    assert set(SUPPORTED_TOOL_NAMES) == REGISTERED_TOOLS
    assert len(SUPPORTED_TOOL_NAMES) == len(set(SUPPORTED_TOOL_NAMES)) == 8


# --- Supported examples -----------------------------------------------------

def test_supported_examples_are_parsed_and_well_formed() -> None:
    for ex in SUPPORTED_QUERY_EXAMPLES:
        assert ex.expected_status == PARSE_STATUS_PARSED
        assert ex.expected_tool in SUPPORTED_TOOL_NAMES
        assert ex.expected_arguments is not None
        assert ex.expected_error_codes == ()
        json.dumps(ex.to_dict())


def test_every_tool_has_a_supported_example() -> None:
    covered = {ex.expected_tool for ex in SUPPORTED_QUERY_EXAMPLES}
    assert covered == REGISTERED_TOOLS


def test_brief_style_example_present() -> None:
    queries = {ex.query for ex in SUPPORTED_QUERY_EXAMPLES}
    assert "What is the average points scored by the Warriors in their last 5 games?" in queries
    brief = next(
        ex for ex in SUPPORTED_QUERY_EXAMPLES
        if ex.query == "What is the average points scored by the Warriors in their last 5 games?"
    )
    assert brief.expected_tool == "team_average_points"
    assert dict(brief.expected_arguments) == {"team": "Warriors", "window": 5}


def test_top_scoring_no_number_omits_n() -> None:
    highest = next(ex for ex in SUPPORTED_QUERY_EXAMPLES if ex.query == "Highest scoring teams")
    assert dict(highest.expected_arguments) == {}  # no invented n


def test_raw_team_strings_preserved() -> None:
    gsw = next(ex for ex in SUPPORTED_QUERY_EXAMPLES if ex.query.startswith("What is GSW"))
    assert dict(gsw.expected_arguments)["team"] == "GSW"  # raw, not canonicalised


def test_rule_query_example_expected_arguments_are_immutable() -> None:
    ex = next(e for e in SUPPORTED_QUERY_EXAMPLES if e.expected_arguments)
    with pytest.raises(TypeError):
        ex.expected_arguments["team"] = "Hacked"


# --- Unsupported examples ---------------------------------------------------

def test_unsupported_examples_are_non_parsed_with_error_codes() -> None:
    for ex in UNSUPPORTED_QUERY_EXAMPLES:
        assert ex.expected_status != PARSE_STATUS_PARSED
        assert ex.expected_tool is None
        assert ex.expected_arguments is None
        assert ex.expected_error_codes
        assert all(code in PARSE_ERROR_CODES for code in ex.expected_error_codes)


def test_unsupported_compare_is_incomplete_not_parsed() -> None:
    # Explicit "compare A and B" is now a SUPPORTED comparison; the only unsupported compare
    # ("Compare LA teams") has no clear second team, so it is incomplete (never head_to_head).
    compares = [ex for ex in UNSUPPORTED_QUERY_EXAMPLES if "compare" in ex.query.lower()]
    assert compares
    for ex in compares:
        assert ex.expected_status == "incomplete"
        assert "missing_opponent" in ex.expected_error_codes


def test_vague_time_examples_use_unsupported_time_expression() -> None:
    vague = [ex for ex in UNSUPPORTED_QUERY_EXAMPLES if "vague_time" in ex.tags]
    assert vague
    for ex in vague:
        assert "unsupported_time_expression" in ex.expected_error_codes


def test_more_vague_time_phrases_are_unsupported() -> None:
    phrases = ("recently", "lately", "of late", "latest")
    for phrase in phrases:
        matches = [ex for ex in UNSUPPORTED_QUERY_EXAMPLES if phrase in ex.query.lower()]
        assert matches, f"no catalogue example for vague phrase {phrase!r}"
        for ex in matches:
            assert "unsupported_time_expression" in ex.expected_error_codes


def test_incomplete_h2h_examples_use_missing_team_or_opponent() -> None:
    h2h = [ex for ex in UNSUPPORTED_QUERY_EXAMPLES if "h2h_incomplete" in ex.tags]
    assert h2h
    for ex in h2h:
        assert {"missing_team", "missing_opponent"} & set(ex.expected_error_codes)


# --- Full catalogue ---------------------------------------------------------

def test_all_examples_well_formed() -> None:
    assert ALL_QUERY_EXAMPLES == SUPPORTED_QUERY_EXAMPLES + UNSUPPORTED_QUERY_EXAMPLES
    queries = [ex.query for ex in ALL_QUERY_EXAMPLES]
    assert all(q for q in queries)
    assert len(queries) == len(set(queries))  # no duplicates
    for ex in ALL_QUERY_EXAMPLES:
        json.dumps(ex.to_dict())
        assert all(isinstance(t, str) for t in ex.tags)
        assert ex.notes is None or isinstance(ex.notes, str)


# --- No parsing logic -------------------------------------------------------

def test_catalogue_exposes_no_parser_functions() -> None:
    for name in PARSER_FUNCTION_NAMES:
        assert not hasattr(catalogue, name), f"{name} must not live in the catalogue"


# --- Import / scope safety --------------------------------------------------

def test_catalogue_import_is_lightweight() -> None:
    code = (
        "import sys; import src.rule_query_catalogue;"
        "forbidden = ['pandas', 'src.tool_registry', 'src.tools', 'src.validation_context',"
        " 'src.team_resolution', 'src.intent_validator', 'src.rule_parser'];"
        "assert not any(m in sys.modules for m in forbidden), [m for m in forbidden if m in sys.modules];"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
