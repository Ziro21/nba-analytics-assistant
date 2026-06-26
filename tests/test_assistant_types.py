"""Phase 9A tests: assistant response contracts only."""

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
    ASSISTANT_ISSUE_CODES,
    ASSISTANT_STATUS_ANSWER,
    ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    ASSISTANT_STATUS_ERROR,
    ASSISTANT_STATUS_UNSUPPORTED,
    ASSISTANT_STATUSES,
    EXECUTION_FAILED,
    INTERNAL_ERROR,
    INVALID_SPECIAL_TEAM,
    MISSING_INFORMATION,
    NO_DATA,
    PARSE_FAILED,
    SAME_TEAM_HEAD_TO_HEAD,
    UNKNOWN_TEAM,
    UNSUPPORTED_QUERY,
    VALIDATION_FAILED,
    AssistantIssue,
    AssistantResult,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_STATUSES = {
    "answer", "clarification_needed", "unsupported", "error",
}
EXPECTED_STATUS_ORDER = (
    "answer", "clarification_needed", "unsupported", "error",
)
EXPECTED_ISSUE_CODES = {
    "parse_failed", "unsupported_query", "ambiguous_intent", "missing_information",
    "validation_failed", "ambiguous_team", "unknown_team", "invalid_special_team",
    "same_team_head_to_head", "no_data", "execution_failed", "internal_error",
}
EXPECTED_ISSUE_CODE_ORDER = (
    "parse_failed", "unsupported_query", "ambiguous_intent", "missing_information",
    "validation_failed", "ambiguous_team", "unknown_team", "invalid_special_team",
    "same_team_head_to_head", "no_data", "execution_failed", "internal_error",
)


def _issue(code: str = AMBIGUOUS_TEAM) -> AssistantIssue:
    return AssistantIssue(code=code, message="Issue message.", field="team", value="LA")


# --- status constants -------------------------------------------------------

def test_status_constants() -> None:
    assert ASSISTANT_STATUS_ANSWER == "answer"
    assert ASSISTANT_STATUS_CLARIFICATION_NEEDED == "clarification_needed"
    assert ASSISTANT_STATUS_UNSUPPORTED == "unsupported"
    assert ASSISTANT_STATUS_ERROR == "error"
    assert set(ASSISTANT_STATUSES) == EXPECTED_STATUSES
    assert ASSISTANT_STATUSES == EXPECTED_STATUS_ORDER
    assert all(isinstance(status, str) and status for status in ASSISTANT_STATUSES)
    assert len(ASSISTANT_STATUSES) == len(set(ASSISTANT_STATUSES))


# --- issue-code constants ---------------------------------------------------

def test_issue_code_constants() -> None:
    constants = (
        PARSE_FAILED, UNSUPPORTED_QUERY, AMBIGUOUS_INTENT, MISSING_INFORMATION,
        VALIDATION_FAILED, AMBIGUOUS_TEAM, UNKNOWN_TEAM, INVALID_SPECIAL_TEAM,
        SAME_TEAM_HEAD_TO_HEAD, NO_DATA, EXECUTION_FAILED, INTERNAL_ERROR,
    )
    assert set(constants) == EXPECTED_ISSUE_CODES
    assert set(ASSISTANT_ISSUE_CODES) == EXPECTED_ISSUE_CODES
    assert ASSISTANT_ISSUE_CODES == EXPECTED_ISSUE_CODE_ORDER
    assert all(isinstance(code, str) and code for code in ASSISTANT_ISSUE_CODES)
    assert len(ASSISTANT_ISSUE_CODES) == len(set(ASSISTANT_ISSUE_CODES))


# --- AssistantIssue ---------------------------------------------------------

def test_assistant_issue_valid_and_serialisable() -> None:
    issue = AssistantIssue(
        code=AMBIGUOUS_TEAM,
        message="The team name is ambiguous.",
        field="team",
        value="LA",
        suggestions=["Los Angeles Lakers", "Los Angeles Clippers"],
    )
    assert issue.suggestions == ("Los Angeles Lakers", "Los Angeles Clippers")
    assert issue.to_dict() == {
        "code": AMBIGUOUS_TEAM,
        "message": "The team name is ambiguous.",
        "field": "team",
        "value": "LA",
        "suggestions": ["Los Angeles Lakers", "Los Angeles Clippers"],
    }
    json.dumps(issue.to_dict())


@pytest.mark.parametrize("code", ["", None, 123])
def test_assistant_issue_invalid_code_rejected(code) -> None:
    with pytest.raises((TypeError, ValueError)):
        AssistantIssue(code=code, message="message")


@pytest.mark.parametrize("message", ["", None, 123])
def test_assistant_issue_invalid_message_rejected(message) -> None:
    with pytest.raises((TypeError, ValueError)):
        AssistantIssue(code=INTERNAL_ERROR, message=message)


def test_assistant_issue_field_must_be_none_or_string() -> None:
    AssistantIssue(code=INTERNAL_ERROR, message="m", field=None)
    AssistantIssue(code=INTERNAL_ERROR, message="m", field="team")
    with pytest.raises(TypeError):
        AssistantIssue(code=INTERNAL_ERROR, message="m", field=123)


def test_assistant_issue_suggestions_validation() -> None:
    issue = AssistantIssue(code=UNKNOWN_TEAM, message="m", suggestions=("Boston Celtics",))
    assert issue.suggestions == ("Boston Celtics",)
    with pytest.raises(TypeError):
        AssistantIssue(code=UNKNOWN_TEAM, message="m", suggestions="Boston Celtics")
    with pytest.raises(TypeError):
        AssistantIssue(code=UNKNOWN_TEAM, message="m", suggestions=("Boston Celtics", 123))


def test_assistant_issue_non_json_value_safely_represented() -> None:
    issue = AssistantIssue(code=INTERNAL_ERROR, message="m", value={1, 2, 3})
    out = issue.to_dict()
    assert isinstance(out["value"], str)
    json.dumps(out)


def test_assistant_issue_to_dict_mutation_does_not_affect_issue() -> None:
    issue = AssistantIssue(code=UNKNOWN_TEAM, message="m", suggestions=("Boston Celtics",))
    out = issue.to_dict()
    out["suggestions"].append("Hacked")
    out["code"] = "hacked"
    assert issue.code == UNKNOWN_TEAM
    assert issue.suggestions == ("Boston Celtics",)


def test_assistant_issue_value_is_isolated_and_immutable() -> None:
    value = {"raw": ["LA"]}
    issue = AssistantIssue(code=AMBIGUOUS_TEAM, message="m", value=value)
    value["raw"].append("Hacked")
    assert issue.value["raw"] == ("LA",)
    with pytest.raises(TypeError):
        issue.value["raw"] = ()
    out = issue.to_dict()
    out["value"]["raw"].append("Changed")
    assert issue.value["raw"] == ("LA",)


# --- AssistantResult --------------------------------------------------------

def test_answer_result_valid_and_serialisable() -> None:
    result = AssistantResult.answer(
        "Golden State Warriors average 114.4 points.",
        query="q",
        tool_name="team_average_points",
        data={"team": "Golden State Warriors", "values": [114.4]},
        meta={"source": "tool"},
    )
    assert result.status == ASSISTANT_STATUS_ANSWER
    assert result.errors == ()
    assert result.data["team"] == "Golden State Warriors"
    with pytest.raises(TypeError):
        result.data["team"] = "Hacked"
    json.dumps(result.to_dict())


def test_clarification_needed_result_valid() -> None:
    result = AssistantResult.clarification_needed(
        "Please clarify which team you mean.",
        (_issue(AMBIGUOUS_TEAM),),
        query="How many points do LA average?",
    )
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert result.errors[0].code == AMBIGUOUS_TEAM
    json.dumps(result.to_dict())


def test_unsupported_result_valid() -> None:
    result = AssistantResult.unsupported(
        "I can only answer supported NBA analytics questions.",
        (_issue(UNSUPPORTED_QUERY),),
        query="Who is better?",
    )
    assert result.status == ASSISTANT_STATUS_UNSUPPORTED


def test_error_result_valid() -> None:
    result = AssistantResult.error(
        "Something went wrong.",
        (_issue(INTERNAL_ERROR),),
        query="q",
        tool_name="team_average_points",
    )
    assert result.status == ASSISTANT_STATUS_ERROR


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValueError):
        AssistantResult("ok", "message")


@pytest.mark.parametrize("message", ["", None, 123])
def test_result_message_must_be_non_empty_string(message) -> None:
    with pytest.raises((TypeError, ValueError)):
        AssistantResult.answer(message)


def test_result_query_and_tool_name_types() -> None:
    with pytest.raises(TypeError):
        AssistantResult.answer("message", query=123)
    with pytest.raises(TypeError):
        AssistantResult.answer("message", tool_name=123)


def test_answer_result_cannot_contain_errors() -> None:
    with pytest.raises(ValueError):
        AssistantResult(ASSISTANT_STATUS_ANSWER, "message", errors=(_issue(),))


@pytest.mark.parametrize("status", [
    ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    ASSISTANT_STATUS_UNSUPPORTED,
    ASSISTANT_STATUS_ERROR,
])
def test_non_answer_statuses_require_at_least_one_issue(status) -> None:
    with pytest.raises(ValueError):
        AssistantResult(status, "message")


def test_errors_and_warnings_must_be_assistant_issues() -> None:
    with pytest.raises(TypeError):
        AssistantResult.clarification_needed("message", ("not an issue",))
    with pytest.raises(TypeError):
        AssistantResult.answer("message", warnings=("not an issue",))


def test_data_and_meta_defensive_copy() -> None:
    data = {"team": "Warriors", "nested": {"games": [1, 2, 3]}}
    meta = {"trace": {"steps": ["parse"]}}
    result = AssistantResult.answer("message", data=data, meta=meta)
    data["team"] = "Hacked"
    data["nested"]["games"].append(4)
    meta["trace"]["steps"].append("hacked")
    assert result.data["team"] == "Warriors"
    assert result.data["nested"]["games"] == (1, 2, 3)
    assert result.meta["trace"]["steps"] == ("parse",)
    with pytest.raises(TypeError):
        result.meta["trace"] = {}


def test_data_and_meta_must_be_dict_or_none() -> None:
    with pytest.raises(TypeError):
        AssistantResult.answer("message", data=["bad"])
    with pytest.raises(TypeError):
        AssistantResult.answer("message", meta=["bad"])


def test_data_and_meta_non_string_keys_are_json_safe() -> None:
    result = AssistantResult.answer("message", data={1: "one"}, meta={("a", "b"): 2})
    out = result.to_dict()
    assert out["data"] == {"1": "one"}
    assert out["meta"] == {"('a', 'b')": 2}
    json.dumps(out)


def test_data_and_meta_reject_json_key_collisions() -> None:
    with pytest.raises(ValueError):
        AssistantResult.answer("message", data={1: "one", "1": "string-one"})
    with pytest.raises(ValueError):
        AssistantResult.answer("message", meta={1: "one", "1": "string-one"})


def test_result_to_dict_json_serialisable_and_isolated() -> None:
    warning = AssistantIssue(code=NO_DATA, message="No rows found.")
    result = AssistantResult.answer(
        "message",
        data={"items": [1, object()]},
        warnings=(warning,),
        meta={"note": {"x", "y"}},
    )
    out = result.to_dict()
    json.dumps(out)
    assert isinstance(out["data"]["items"][1], str)
    assert isinstance(out["meta"]["note"], str)
    out["data"]["items"].append("hacked")
    out["warnings"].append({"code": "hacked"})
    assert result.to_dict()["data"]["items"] != out["data"]["items"]
    assert len(result.warnings) == 1


def test_warning_to_dict_mutation_does_not_affect_result() -> None:
    warning = AssistantIssue(code=NO_DATA, message="No data warning.")
    result = AssistantResult.answer("message", warnings=(warning,))
    out = result.to_dict()
    out["warnings"][0]["code"] = "hacked"
    out["warnings"].append({"code": "extra"})
    assert result.warnings == (warning,)
    assert result.to_dict()["warnings"][0]["code"] == NO_DATA


def test_convenience_constructors_enforce_invariants() -> None:
    with pytest.raises(ValueError):
        AssistantResult.clarification_needed("message", ())
    with pytest.raises(ValueError):
        AssistantResult.unsupported("message", ())
    with pytest.raises(ValueError):
        AssistantResult.error("message", ())


# --- import / scope safety --------------------------------------------------

def test_assistant_types_import_is_lightweight() -> None:
    code = (
        "import sys; import src.assistant_types;"
        "forbidden = ['pandas', 'src.data_loader', 'src.data_model', 'src.data_validation',"
        " 'src.tool_registry', 'src.tools', 'src.validation_context', 'src.intent_validator',"
        " 'src.team_resolution', 'src.rule_parser', 'src.rule_slot_extractor',"
        " 'src.response_formatter', 'src.assistant', 'src.llm_query_parser'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_future_production_modules_absent() -> None:
    for module in ("src.llm_query_parser", "src.parse_validate_execute"):
        assert importlib.util.find_spec(module) is None
