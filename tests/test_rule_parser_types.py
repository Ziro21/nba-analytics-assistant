"""Phase 8A tests: rule-parser contract objects only."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.intent_types import ParsedIntent
from src.rule_parser_types import (
    PARSE_ERROR_CODES,
    PARSE_STATUS_AMBIGUOUS,
    PARSE_STATUS_INCOMPLETE,
    PARSE_STATUS_NO_PARSE,
    PARSE_STATUS_PARSED,
    PARSE_STATUSES,
    UNSUPPORTED_QUERY,
    ParseError,
    ParseWarning,
    RuleParseResult,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_MODULES = (  # 8B-8D and 9B modules may now exist; these still must not.
    "src.rule_parser_validation_integration",
    "src.llm_query_parser",
)

EXPECTED_STATUSES = {"parsed", "no_parse", "ambiguous", "incomplete"}
EXPECTED_ERROR_CODES = {
    "empty_query", "unsupported_query", "ambiguous_intent", "missing_team",
    "missing_opponent", "missing_number", "ambiguous_team_mention", "unsupported_time_expression",
}


def _intent() -> ParsedIntent:
    return ParsedIntent("team_average_points", {"team": "Warriors", "window": 5}, "rule")


def _error() -> ParseError:
    return ParseError(code=UNSUPPORTED_QUERY, message="unsupported")


# --- Status constants -------------------------------------------------------

def test_status_constants() -> None:
    assert set(PARSE_STATUSES) == EXPECTED_STATUSES
    assert all(isinstance(s, str) and s for s in PARSE_STATUSES)
    assert len(PARSE_STATUSES) == len(set(PARSE_STATUSES))


# --- Error-code constants ---------------------------------------------------

def test_error_code_constants() -> None:
    assert set(PARSE_ERROR_CODES) == EXPECTED_ERROR_CODES
    assert all(isinstance(c, str) and c for c in PARSE_ERROR_CODES)
    assert len(PARSE_ERROR_CODES) == len(set(PARSE_ERROR_CODES))


# --- ParseError -------------------------------------------------------------

def test_parse_error_valid_and_serialisable() -> None:
    err = ParseError(code="missing_team", message="No team found.", field="team",
                     suggestions=["Boston Celtics"])
    assert err.suggestions == ("Boston Celtics",)
    json.dumps(err.to_dict())


@pytest.mark.parametrize("code,message", [("", "m"), ("c", "")])
def test_parse_error_empty_code_or_message_rejected(code, message) -> None:
    with pytest.raises((TypeError, ValueError)):
        ParseError(code=code, message=message)


def test_parse_error_string_suggestions_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ParseError(code="c", message="m", suggestions="Boston Celtics")


def test_parse_error_non_string_suggestion_items_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ParseError(code="c", message="m", suggestions=("Boston Celtics", 123))


def test_parse_error_non_json_value_safely_represented() -> None:
    err = ParseError(code="c", message="m", value={1, 2, 3})  # a set is not JSON
    d = err.to_dict()
    json.dumps(d)
    assert isinstance(d["value"], str)


# --- ParseWarning -----------------------------------------------------------

def test_parse_warning_valid_and_serialisable() -> None:
    warn = ParseWarning(code="some_warning", message="note", suggestions=("x",))
    assert warn.suggestions == ("x",)
    json.dumps(warn.to_dict())


@pytest.mark.parametrize("code,message", [("", "m"), ("c", "")])
def test_parse_warning_empty_rejected(code, message) -> None:
    with pytest.raises((TypeError, ValueError)):
        ParseWarning(code=code, message=message)


def test_parse_warning_non_string_suggestion_items_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ParseWarning(code="c", message="m", suggestions=("Boston Celtics", 123))


# --- RuleParseResult --------------------------------------------------------

def test_parsed_result_requires_intent_and_no_errors() -> None:
    res = RuleParseResult.parsed(_intent(), raw_query="q")
    assert res.status == PARSE_STATUS_PARSED and res.parsed_intent is not None
    assert res.raw_query == "q"
    json.dumps(res.to_dict())
    with pytest.raises((TypeError, ValueError)):
        RuleParseResult(PARSE_STATUS_PARSED, None)
    with pytest.raises((TypeError, ValueError)):
        RuleParseResult(PARSE_STATUS_PARSED, _intent(), (_error(),))


@pytest.mark.parametrize("ctor", [
    RuleParseResult.no_parse, RuleParseResult.ambiguous, RuleParseResult.incomplete,
])
def test_non_parsed_results_have_errors_and_no_intent(ctor) -> None:
    res = ctor((_error(),), raw_query="q")
    assert res.status in {PARSE_STATUS_NO_PARSE, PARSE_STATUS_AMBIGUOUS, PARSE_STATUS_INCOMPLETE}
    assert res.parsed_intent is None and res.errors
    json.dumps(res.to_dict())


def test_non_parsed_result_with_intent_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        RuleParseResult(PARSE_STATUS_NO_PARSE, _intent(), (_error(),))


def test_non_parsed_result_without_errors_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        RuleParseResult.no_parse(())


def test_invalid_status_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        RuleParseResult("weird", None, (_error(),))


def test_to_dict_mutation_does_not_affect_result() -> None:
    res = RuleParseResult.no_parse((_error(),), raw_query="q")
    d = res.to_dict()
    d["errors"].append("x")
    d["status"] = "hacked"
    assert len(res.errors) == 1 and res.status == PARSE_STATUS_NO_PARSE


# --- Import / scope safety --------------------------------------------------

def test_forbidden_modules_absent() -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


def test_rule_parser_types_import_is_lightweight() -> None:
    code = (
        "import sys; import src.rule_parser_types;"
        "forbidden = ['pandas', 'src.tool_registry', 'src.tools', 'src.validation_context',"
        " 'src.team_resolution', 'src.intent_validator'];"
        "assert not any(m in sys.modules for m in forbidden), [m for m in forbidden if m in sys.modules];"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
