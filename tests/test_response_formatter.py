"""Phase 9B tests: deterministic response formatter only."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.assistant_types import (
    AMBIGUOUS_INTENT,
    AMBIGUOUS_TEAM,
    ASSISTANT_STATUS_ANSWER,
    ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    ASSISTANT_STATUS_ERROR,
    ASSISTANT_STATUS_UNSUPPORTED,
    EXECUTION_FAILED,
    INTERNAL_ERROR,
    INVALID_SPECIAL_TEAM,
    MISSING_INFORMATION,
    NO_DATA,
    SAME_TEAM_HEAD_TO_HEAD,
    UNKNOWN_TEAM,
    UNSUPPORTED_QUERY,
    VALIDATION_FAILED,
)
from src.intent_types import (
    AMBIGUOUS_TEAM as VALIDATION_AMBIGUOUS_TEAM,
    INVALID_N,
    INVALID_SEASON_ID,
    INVALID_SPECIAL_TEAM as VALIDATION_INVALID_SPECIAL_TEAM,
    INVALID_WINDOW,
    MISSING_REQUIRED_ARGUMENT,
    PARSER_MODE_RULE,
    SAME_TEAM_HEAD_TO_HEAD as VALIDATION_SAME_TEAM_HEAD_TO_HEAD,
    SEVERITY_WARNING,
    UNKNOWN_TEAM as VALIDATION_UNKNOWN_TEAM,
    ParsedIntent,
    ValidatedIntent,
    ValidationError,
    ValidationResult,
)
from src.response_formatter import (
    SUPPORTED_TOOL_NAMES,
    format_parse_failure,
    format_tool_result,
    format_validation_failure,
)
from src.rule_parser_types import (
    AMBIGUOUS_INTENT as PARSE_AMBIGUOUS_INTENT,
    MISSING_TEAM,
    PARSE_STATUS_PARSED,
    UNSUPPORTED_QUERY as PARSE_UNSUPPORTED_QUERY,
    UNSUPPORTED_TIME_EXPRESSION,
    ParseError,
    ParseWarning,
    RuleParseResult,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _meta(
    *,
    team: str | None = None,
    games_used: int | None = None,
    window_requested: int | None = None,
    season_id: int | None = None,
) -> dict[str, object]:
    return {
        "team": team,
        "games_used": games_used,
        "date_range": ["2024-01-01", "2024-01-10"] if games_used else None,
        "window_requested": window_requested,
        "season_id": season_id,
    }


OK_TOOL_CASES = (
    (
        {
            "status": "ok",
            "tool": "team_average_points",
            "result": {"team": "Golden State Warriors", "average_points": 114.4, "games_used": 5},
            "meta": _meta(team="Golden State Warriors", games_used=5, window_requested=5),
            "warnings": [],
        },
        "Golden State Warriors averaged 114.4 points over the last 5 games.",
    ),
    (
        {
            "status": "ok",
            "tool": "average_points_allowed",
            "result": {
                "team": "Golden State Warriors",
                "average_points_allowed": 108.2,
                "games_used": 5,
            },
            "meta": _meta(team="Golden State Warriors", games_used=5, window_requested=5),
            "warnings": [],
        },
        "Golden State Warriors allowed 108.2 points per game over the last 5 games.",
    ),
    (
        {
            "status": "ok",
            "tool": "team_record",
            "result": {
                "team": "Boston Celtics",
                "wins": 7,
                "losses": 3,
                "record": "7-3",
                "games_used": 10,
                "win_percentage": 0.7,
            },
            "meta": _meta(team="Boston Celtics", games_used=10, window_requested=10),
            "warnings": [],
        },
        "Boston Celtics are 7-3 over the last 10 games.",
    ),
    (
        {
            "status": "ok",
            "tool": "top_scoring_teams",
            "result": {
                "teams": [
                    {
                        "rank": 1,
                        "team": "Boston Celtics",
                        "average_points": 121.4,
                        "games_used": 82,
                    },
                    {
                        "rank": 2,
                        "team": "Denver Nuggets",
                        "average_points": 119.1,
                        "games_used": 82,
                    },
                ],
                "teams_returned": 2,
                "n_requested": 2,
            },
            "meta": _meta(games_used=164, season_id=26),
            "warnings": [],
        },
        (
            "Top scoring teams in season ID 26: 1. Boston Celtics - 121.4 points per game; "
            "2. Denver Nuggets - 119.1 points per game."
        ),
    ),
    (
        {
            "status": "ok",
            "tool": "head_to_head",
            "result": {
                "team_a": "Boston Celtics",
                "team_b": "Miami Heat",
                "meetings": 5,
                "team_a_wins": 3,
                "team_b_wins": 2,
                "record": "3-2",
                "average_points_for": 110.2,
                "average_points_against": 107.8,
                "average_point_differential": 2.4,
            },
            "meta": _meta(team="Boston Celtics", games_used=5, window_requested=5),
            "warnings": [],
        },
        "Boston Celtics are 3-2 against Miami Heat over the last 5 meetings.",
    ),
    (
        {
            "status": "ok",
            "tool": "team_efficiency_summary",
            "result": {
                "team": "Los Angeles Lakers",
                "average_ortg": 118.25,
                "average_drtg": 112.1,
                "average_net_rating": 6.15,
                "average_possessions": 99.5,
                "games_used": 10,
            },
            "meta": _meta(team="Los Angeles Lakers", games_used=10, window_requested=10),
            "warnings": [],
        },
        "Los Angeles Lakers over the last 10 games: ORTG 118.25, DRTG 112.1, net rating 6.15.",
    ),
)


@pytest.mark.parametrize("tool_result,expected_message", OK_TOOL_CASES)
def test_format_tool_result_ok_for_supported_tools(tool_result, expected_message) -> None:
    result = format_tool_result(tool_result, query="Example query")
    assert result.status == ASSISTANT_STATUS_ANSWER
    assert result.message == expected_message
    assert result.query == "Example query"
    assert result.tool_name == tool_result["tool"]
    assert result.errors == ()
    out = result.to_dict()
    assert out["data"] == tool_result["result"]
    assert out["meta"] == tool_result["meta"]
    json.dumps(out)


def test_supported_tool_catalogue_is_exact() -> None:
    assert SUPPORTED_TOOL_NAMES == {
        "team_average_points",
        "average_points_allowed",
        "team_record",
        "top_scoring_teams",
        "head_to_head",
        "team_efficiency_summary",
    }


def test_format_tool_result_preserves_payload_without_mutation_leaks() -> None:
    tool_result = {
        "status": "ok",
        "tool": "team_average_points",
        "result": {"team": "Golden State Warriors", "average_points": 114.4, "games_used": 5},
        "meta": _meta(team="Golden State Warriors", games_used=5, window_requested=5),
        "warnings": ["Requested last 10 games but only 5 available; using all 5."],
    }
    result = format_tool_result(tool_result, query="q")
    tool_result["result"]["team"] = "Hacked"
    tool_result["meta"]["team"] = "Hacked"
    out = result.to_dict()
    out["data"]["team"] = "Changed"
    out["warnings"].append({"code": "changed"})
    assert result.data["team"] == "Golden State Warriors"
    assert result.meta["team"] == "Golden State Warriors"
    assert result.warnings[0].code == VALIDATION_FAILED
    assert "Requested last 10 games" in result.warnings[0].message


@pytest.mark.parametrize("warnings", [
    "one warning",
    {"message": "one warning"},
    object(),
    None,
])
def test_format_tool_result_rejects_malformed_warnings_shape(warnings) -> None:
    tool_result = {
        "status": "ok",
        "tool": "team_average_points",
        "result": {"team": "Golden State Warriors", "average_points": 114.4, "games_used": 5},
        "meta": _meta(team="Golden State Warriors", games_used=5, window_requested=5),
        "warnings": warnings,
    }
    result = format_tool_result(tool_result, query="q")
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == INTERNAL_ERROR
    assert "malformed tool warnings" in result.message
    assert result.tool_name == "team_average_points"
    json.dumps(result.to_dict())


@pytest.mark.parametrize("warnings", [
    [123],
    ["valid warning", 123],
    (object(),),
])
def test_format_tool_result_rejects_non_string_warning_items(warnings) -> None:
    tool_result = {
        "status": "no_data",
        "tool": "team_average_points",
        "result": {"team": "Seattle SuperSonics", "average_points": None, "games_used": 0},
        "meta": _meta(team="Seattle SuperSonics", games_used=0, window_requested=5),
        "warnings": warnings,
    }
    result = format_tool_result(tool_result, query="q")
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == INTERNAL_ERROR
    assert "malformed tool warnings" in result.message
    assert result.tool_name == "team_average_points"


def test_format_tool_result_no_data_becomes_clarification_and_preserves_data() -> None:
    tool_result = {
        "status": "no_data",
        "tool": "team_average_points",
        "result": {"team": "Seattle SuperSonics", "average_points": None, "games_used": 0},
        "meta": _meta(team="Seattle SuperSonics", games_used=0, window_requested=5),
        "warnings": ["No games found for team 'Seattle SuperSonics'."],
    }
    result = format_tool_result(tool_result, query="How many points do Seattle average?")
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert result.tool_name == "team_average_points"
    assert result.errors[0].code == NO_DATA
    assert result.warnings[0].code == NO_DATA
    assert result.data["team"] == "Seattle SuperSonics"
    assert result.meta["window_requested"] == 5
    json.dumps(result.to_dict())


def test_format_tool_result_error_becomes_assistant_error_and_preserves_data() -> None:
    tool_result = {
        "status": "error",
        "tool": "team_record",
        "result": {"message": "window must be a positive integer, got 0"},
        "meta": _meta(team="Boston Celtics"),
        "warnings": [],
    }
    result = format_tool_result(tool_result, query="Celtics record last 0 games")
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.tool_name == "team_record"
    assert result.errors[0].code == EXECUTION_FAILED
    assert "window must be a positive integer" in result.errors[0].message
    assert "Traceback" not in result.errors[0].message
    assert result.data == {"message": "window must be a positive integer, got 0"}
    json.dumps(result.to_dict())


@pytest.mark.parametrize("bad_result", [
    None,
    {"status": "ok", "tool": "team_average_points", "result": {}, "meta": {}},
    {"status": "partial", "tool": "team_average_points", "result": {}, "meta": {}, "warnings": []},
    {"status": "ok", "tool": "unknown_tool", "result": {}, "meta": {}, "warnings": []},
    {
        "status": "ok",
        "tool": "team_average_points",
        "result": [],
        "meta": {},
        "warnings": [],
    },
    {
        "status": "ok",
        "tool": "team_average_points",
        "result": {"team": "Golden State Warriors"},
        "meta": {},
        "warnings": [],
    },
])
def test_format_tool_result_malformed_input_fails_safely(bad_result) -> None:
    result = format_tool_result(bad_result, query="q")
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == INTERNAL_ERROR
    assert result.message
    assert "Traceback" not in result.message
    json.dumps(result.to_dict())


def test_format_parse_failure_no_parse_becomes_unsupported() -> None:
    parse_result = RuleParseResult.no_parse(
        (
            ParseError(
                PARSE_UNSUPPORTED_QUERY,
                "This query is outside the supported catalogue.",
                value="Who is better?",
            ),
        ),
        raw_query="Who is better?",
    )
    result = format_parse_failure(parse_result)
    assert result.status == ASSISTANT_STATUS_UNSUPPORTED
    assert result.query == "Who is better?"
    assert result.errors[0].code == UNSUPPORTED_QUERY
    json.dumps(result.to_dict())


def test_format_parse_failure_ambiguous_becomes_clarification() -> None:
    parse_result = RuleParseResult.ambiguous(
        (
            ParseError(
                PARSE_AMBIGUOUS_INTENT,
                "Generic comparison requests are ambiguous.",
                value="Compare Lakers and Celtics",
            ),
        ),
        raw_query="Compare Lakers and Celtics",
    )
    result = format_parse_failure(parse_result)
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert result.errors[0].code == AMBIGUOUS_INTENT


@pytest.mark.parametrize("parse_code", [MISSING_TEAM, UNSUPPORTED_TIME_EXPRESSION])
def test_format_parse_failure_incomplete_becomes_clarification(parse_code) -> None:
    parse_result = RuleParseResult.incomplete(
        (ParseError(parse_code, "More information is needed.", field="team"),),
        raw_query="Warriors last few games",
        warnings=(ParseWarning("future_warning", "Non-blocking parse warning."),),
    )
    result = format_parse_failure(parse_result, query="Override query")
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert result.query == "Override query"
    assert result.errors[0].code == MISSING_INFORMATION
    assert result.warnings[0].code == "parse_failed"


def test_format_parse_failure_rejects_parsed_or_malformed_inputs_safely() -> None:
    parsed_intent = ParsedIntent(
        "team_record",
        {"team": "Golden State Warriors"},
        PARSER_MODE_RULE,
        raw_query="Warriors record",
    )
    parsed = RuleParseResult.parsed(parsed_intent, raw_query="Warriors record")
    parsed_result = format_parse_failure(parsed)
    malformed_result = format_parse_failure({"status": PARSE_STATUS_PARSED})
    assert parsed_result.status == ASSISTANT_STATUS_ERROR
    assert parsed_result.errors[0].code == INTERNAL_ERROR
    assert malformed_result.status == ASSISTANT_STATUS_ERROR
    assert malformed_result.errors[0].code == INTERNAL_ERROR


@pytest.mark.parametrize("validation_code,assistant_code", [
    (VALIDATION_AMBIGUOUS_TEAM, AMBIGUOUS_TEAM),
    (VALIDATION_UNKNOWN_TEAM, UNKNOWN_TEAM),
    (VALIDATION_INVALID_SPECIAL_TEAM, INVALID_SPECIAL_TEAM),
    (VALIDATION_SAME_TEAM_HEAD_TO_HEAD, SAME_TEAM_HEAD_TO_HEAD),
    (MISSING_REQUIRED_ARGUMENT, MISSING_INFORMATION),
    (INVALID_WINDOW, MISSING_INFORMATION),
    (INVALID_N, MISSING_INFORMATION),
    (INVALID_SEASON_ID, MISSING_INFORMATION),
    ("future_validation_code", VALIDATION_FAILED),
])
def test_format_validation_failure_maps_validation_errors(validation_code, assistant_code) -> None:
    validation_result = ValidationResult.invalid(
        (
            ValidationError(
                validation_code,
                "Validation failed.",
                field="team",
                value="LA",
                suggestions=("Los Angeles Lakers", "Los Angeles Clippers"),
            ),
        ),
        warnings=(
            ValidationError(
                INVALID_WINDOW,
                "Window is larger than available data.",
                field="window",
                value=200,
                severity=SEVERITY_WARNING,
            ),
        ),
    )
    result = format_validation_failure(
        validation_result,
        query="How many points do LA average?",
        tool_name="team_average_points",
    )
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert result.tool_name == "team_average_points"
    assert result.errors[0].code == assistant_code
    assert result.errors[0].suggestions == ("Los Angeles Lakers", "Los Angeles Clippers")
    assert result.warnings[0].code == VALIDATION_FAILED
    json.dumps(result.to_dict())


def test_format_validation_failure_rejects_valid_or_malformed_inputs_safely() -> None:
    validated = ValidatedIntent(
        "team_record",
        {"team": "Golden State Warriors"},
        PARSER_MODE_RULE,
        raw_query="Warriors record",
    )
    valid_result = format_validation_failure(
        ValidationResult.valid(validated),
        query="Warriors record",
        tool_name="team_record",
    )
    malformed_result = format_validation_failure({"is_valid": False}, query="q")
    assert valid_result.status == ASSISTANT_STATUS_ERROR
    assert valid_result.errors[0].code == INTERNAL_ERROR
    assert malformed_result.status == ASSISTANT_STATUS_ERROR
    assert malformed_result.errors[0].code == INTERNAL_ERROR


def test_response_formatter_import_is_lightweight() -> None:
    code = (
        "import sys; import src.response_formatter;"
        "forbidden = ['pandas', 'src.data_loader', 'src.data_model', 'src.data_validation',"
        " 'src.tool_registry', 'src.tools', 'src.validation_context', 'src.intent_validator',"
        " 'src.team_resolution', 'src.rule_parser', 'src.rule_slot_extractor',"
        " 'src.assistant', 'src.llm_query_parser', 'src.parse_validate_execute'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_response_formatter_does_not_contain_orchestration_calls() -> None:
    source = (REPO_ROOT / "src" / "response_formatter.py").read_text()
    for forbidden in (
        "parse_rule_query",
        "validate_intent",
        "resolve_team_name",
        "DEFAULT_REGISTRY",
        ".execute(",
        "load_raw_dataset",
        "build_clean_view",
        "import pandas",
        "src.tool_registry",
        "src.tools",
        "src.intent_validator",
        "src.team_resolution",
        "src.llm_query_parser",
    ):
        assert forbidden not in source


def test_future_orchestration_and_llm_modules_absent() -> None:
    for module in (
        "src.llm_query_parser",
        "src.parse_validate_execute",
        "src.api",
        "src.web",
        "src.database",
        "src.rag",
        "src.agent",
        "src.server",
    ):
        assert importlib.util.find_spec(module) is None
