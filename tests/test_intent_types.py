"""Phase 7A tests: shared intent/validation contract objects only.

No validation logic, no team resolution, no registry, no pandas, no data. Pure contracts.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.intent_types import (
    ERROR_CODES,
    PARSER_MODE_LLM,
    PARSER_MODE_RULE,
    PARSER_MODES,
    ParsedIntent,
    ValidatedIntent,
    ValidationError,
    ValidationResult,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# validation_context/team_resolution (7B), intent_validator (7C), and
# response_formatter (9B) now exist; these later-layer modules still must not.
FORBIDDEN_MODULES = (
    "src.query_parser",
    "src.llm_query_parser",
    "src.assistant",
)

EXPECTED_CODES = {
    "unknown_tool", "arguments_not_dict", "missing_required_argument", "unexpected_argument",
    "invalid_argument_type", "invalid_parser_mode", "unknown_team", "ambiguous_team",
    "invalid_special_team", "invalid_window", "invalid_n", "invalid_season_id",
    "same_team_head_to_head",
}


def _warning(code: str = "canonicalised_team") -> ValidationError:
    return ValidationError(code=code, message="m", severity="warning")


def _error(code: str = "unknown_team") -> ValidationError:
    return ValidationError(code=code, message="m", severity="error")


# --- Group 1: parser modes --------------------------------------------------

def test_parser_modes() -> None:
    assert PARSER_MODE_RULE == "rule"
    assert PARSER_MODE_LLM == "llm"
    assert PARSER_MODES == ("rule", "llm")
    assert all(isinstance(m, str) for m in PARSER_MODES)


def test_invalid_parser_mode_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ParsedIntent("team_average_points", {"team": "x"}, "magic")


# --- Group 2: error codes ---------------------------------------------------

def test_error_codes_are_unique_non_empty_strings() -> None:
    assert all(isinstance(c, str) and c for c in ERROR_CODES)
    assert len(ERROR_CODES) == len(set(ERROR_CODES))


def test_error_codes_exact_catalogue() -> None:
    assert set(ERROR_CODES) == EXPECTED_CODES


# --- Group 3: ParsedIntent --------------------------------------------------

def test_parsed_intent_valid_modes_construct_and_serialise() -> None:
    for mode in (PARSER_MODE_RULE, PARSER_MODE_LLM):
        intent = ParsedIntent("team_average_points", {"team": "x", "window": 5}, mode)
        json.dumps(intent.to_dict())


def test_parsed_intent_empty_tool_name_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ParsedIntent("", {"team": "x"}, "rule")


def test_parsed_intent_non_dict_arguments_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ParsedIntent("team_average_points", ["not", "a", "dict"], "rule")


def test_parsed_intent_defensively_copies_arguments() -> None:
    args = {"team": "x", "window": 5}
    intent = ParsedIntent("team_average_points", args, "rule")
    args["window"] = 999
    assert intent.arguments["window"] == 5
    dumped = intent.to_dict()
    dumped["arguments"]["window"] = 123
    assert intent.arguments["window"] == 5


def test_parsed_intent_raw_query() -> None:
    assert ParsedIntent("t", {"a": 1}, "rule", raw_query=None).raw_query is None
    assert ParsedIntent("t", {"a": 1}, "rule", raw_query="hi").raw_query == "hi"


def test_parsed_intent_raw_query_non_string_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ParsedIntent("t", {"a": 1}, "rule", raw_query=123)


def test_parsed_intent_arguments_non_json_value_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ParsedIntent("t", {"bad": {1, 2}}, "rule")  # a set is not JSON-serialisable


def test_parsed_intent_arguments_cannot_be_mutated_directly() -> None:
    intent = ParsedIntent("t", {"team": "x"}, "rule")
    with pytest.raises(TypeError):
        intent.arguments["team"] = "Changed Team"


@pytest.mark.parametrize("conf", [None, 0, 0.5, 1.0])
def test_parsed_intent_confidence_accepted(conf) -> None:
    ParsedIntent("t", {"a": 1}, "rule", confidence=conf)


@pytest.mark.parametrize("conf", [True, -0.1, 1.1, "high"])
def test_parsed_intent_confidence_rejected(conf) -> None:
    with pytest.raises((TypeError, ValueError)):
        ParsedIntent("t", {"a": 1}, "rule", confidence=conf)


def test_parsed_intent_confidence_is_plain_metadata_in_dict() -> None:
    d = ParsedIntent("t", {"a": 1}, "rule", confidence=0.9).to_dict()
    assert d["confidence"] == 0.9
    json.dumps(d)


# --- Group 4: ValidationError -----------------------------------------------

def test_validation_error_and_warning_construct() -> None:
    assert _error().severity == "error"
    assert _warning().severity == "warning"


def test_validation_error_invalid_severity_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidationError(code="c", message="m", severity="fatal")


def test_validation_error_empty_code_or_message_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidationError(code="", message="m")
    with pytest.raises((TypeError, ValueError)):
        ValidationError(code="c", message="")


def test_validation_error_suggestions_immutable_tuple() -> None:
    err = ValidationError(code="c", message="m", suggestions=["Boston Celtics"])
    assert isinstance(err.suggestions, tuple)
    assert err.suggestions == ("Boston Celtics",)
    json.dumps(err.to_dict())


def test_validation_error_string_suggestions_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidationError(code="c", message="m", suggestions="Boston Celtics")


def test_validation_error_non_string_suggestion_items_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidationError(code="c", message="m", suggestions=("Boston Celtics", 123))


def test_validation_error_field_non_string_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidationError(code="c", message="m", field=123)


def test_validation_error_non_json_value_is_safely_represented() -> None:
    err = ValidationError(code="c", message="m", value={1, 2, 3})  # a set is not JSON
    d = err.to_dict()
    json.dumps(d)  # must not raise
    assert isinstance(d["value"], str)


# --- Group 5: ValidatedIntent -----------------------------------------------

def test_validated_intent_constructs_and_copies() -> None:
    args = {"team": "Golden State Warriors", "window": 5}
    vi = ValidatedIntent("team_average_points", args, "llm", warnings=(_warning(),))
    args["window"] = 99
    assert vi.arguments["window"] == 5
    json.dumps(vi.to_dict())


def test_validated_intent_rejects_error_severity_in_warnings() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidatedIntent("t", {"a": 1}, "rule", warnings=(_error(),))


def test_validated_intent_invalid_parser_mode_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidatedIntent("t", {"a": 1}, "nope")


def test_validated_intent_arguments_non_json_value_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidatedIntent("t", {"bad": {1, 2}}, "rule")


def test_validated_intent_arguments_cannot_be_mutated_directly() -> None:
    vi = ValidatedIntent("t", {"team": "Golden State Warriors"}, "rule")
    with pytest.raises(TypeError):
        vi.arguments["team"] = "Changed Team"


# --- Group 6: ValidationResult ----------------------------------------------

def test_validation_result_valid_requires_intent_and_no_errors() -> None:
    vi = ValidatedIntent("t", {"a": 1}, "rule")
    res = ValidationResult.valid(vi)
    assert res.is_valid is True
    assert res.validated_intent is vi
    json.dumps(res.to_dict())
    with pytest.raises((TypeError, ValueError)):
        ValidationResult(is_valid=True, validated_intent=None)
    with pytest.raises((TypeError, ValueError)):
        ValidationResult(is_valid=True, validated_intent=vi, errors=(_error(),))


def test_validation_result_invalid_has_errors_and_no_intent() -> None:
    res = ValidationResult.invalid(errors=(_error(),))
    assert res.is_valid is False
    assert res.validated_intent is None
    assert res.errors
    json.dumps(res.to_dict())
    with pytest.raises((TypeError, ValueError)):
        ValidationResult(is_valid=False, validated_intent=ValidatedIntent("t", {}, "rule"))
    with pytest.raises((TypeError, ValueError)):
        ValidationResult.invalid(errors=())  # invalid must have at least one error


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_validation_result_is_valid_must_be_bool(bad) -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidationResult(is_valid=bad, validated_intent=None, errors=(_error(),))


def test_validation_result_warnings_must_be_warning_severity() -> None:
    with pytest.raises((TypeError, ValueError)):
        ValidationResult.invalid(errors=(_error(),), warnings=(_error(),))


# --- Group 7: import / scope safety -----------------------------------------

def test_forbidden_modules_absent() -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


def test_intent_types_import_is_lightweight() -> None:
    code = (
        "import sys; import src.intent_types;"
        "assert 'pandas' not in sys.modules, 'pandas imported';"
        "assert 'src.tool_registry' not in sys.modules, 'registry imported';"
        "assert 'src.tools' not in sys.modules, 'tools imported';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
