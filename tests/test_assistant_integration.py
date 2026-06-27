"""Phase 9D: assistant integration and safety review tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

import src.assistant as assistant_module
from src.assistant import answer_query
from src.assistant_types import (
    AMBIGUOUS_TEAM,
    ASSISTANT_STATUS_ANSWER,
    ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    ASSISTANT_STATUS_ERROR,
    ASSISTANT_STATUS_UNSUPPORTED,
    EXECUTION_FAILED,
    INTERNAL_ERROR,
    NO_DATA,
    SAME_TEAM_HEAD_TO_HEAD,
    UNKNOWN_TEAM,
    AssistantResult,
)
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.intent_types import PARSER_MODE_RULE, ParsedIntent, ValidatedIntent, ValidationResult
from src.rule_parser_types import ParseError, RuleParseResult, UNSUPPORTED_QUERY
from src.tool_registry import DEFAULT_REGISTRY
from src.tool_results import build_meta, error_result, no_data_result, ok_result
from src.validation_context import build_validation_context

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def assistant_dependencies():
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    context = build_validation_context(clean, registry=DEFAULT_REGISTRY)
    return clean, context, DEFAULT_REGISTRY


class SpyRegistry:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def execute(self, name, args=None, *, clean_df):
        self.calls.append((name, dict(args or {}), clean_df))
        return self.result


class ExplodingRegistry:
    def __init__(self):
        self.calls = []

    def execute(self, name, args=None, *, clean_df):
        self.calls.append((name, args, clean_df))
        raise RuntimeError("registry exploded")


def _codes(result) -> list[str]:
    return [issue.code for issue in result.errors]


def _json_safe(result) -> dict:
    payload = result.to_dict()
    json.dumps(payload)
    return payload


def _assert_safe_internal_error(result, query: str = "") -> None:
    payload = _json_safe(result)
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == INTERNAL_ERROR
    assert result.query == query
    dumped = json.dumps(payload)
    assert "Traceback" not in dumped
    assert "RuntimeError" not in dumped
    assert "exploded" not in dumped


def _average_points_result(*, warnings=None):
    return ok_result(
        "team_average_points",
        {"team": "Golden State Warriors", "average_points": 114.4, "games_used": 5},
        meta=build_meta(team="Golden State Warriors", games_used=5, window_requested=5),
        warnings=warnings,
    )


# --- real full-chain supported queries -------------------------------------

@pytest.mark.parametrize("query,tool_name,message_parts", [
    (
        "How many points do the Warriors average over the last 5 games?",
        "team_average_points",
        ("Golden State Warriors", "114.4"),
    ),
    (
        "How many points do GSW allow over the last 5 games?",
        "average_points_allowed",
        ("Golden State Warriors", "117.0"),
    ),
    (
        "What is the Warriors record?",
        "team_record",
        ("Golden State Warriors", "289-223", "512"),
    ),
    (
        "Top 5 scoring teams",
        "top_scoring_teams",
        ("Atlanta Hawks", "116.13"),
    ),
    (
        "Celtics vs Heat head to head",
        "head_to_head",
        ("Boston Celtics", "Miami Heat", "25-14", "39"),
    ),
    (
        "Boston Celtics efficiency last 10 games",
        "team_efficiency_summary",
        ("Boston Celtics", "106.98", "101.93"),
    ),
])
def test_real_supported_queries_return_answers(
    query, tool_name, message_parts, assistant_dependencies
) -> None:
    clean, context, registry = assistant_dependencies
    result = answer_query(query, clean_df=clean, validation_context=context, registry=registry)
    assert result.status == ASSISTANT_STATUS_ANSWER
    assert result.tool_name == tool_name
    assert result.query == query
    assert result.errors == ()
    assert result.data is not None
    assert result.meta is not None
    for part in message_parts:
        assert part in result.message
    _json_safe(result)


# --- v1.1.0-B: a tuned typo suggestion must NOT execute a tool --------------

def test_typo_team_suggestion_does_not_execute_registry(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    spy = SpyRegistry(_average_points_result())  # would return a GSW answer IF wrongly executed
    result = answer_query(
        "How many points do Celics average?",
        clean_df=clean, validation_context=context, registry=spy,
    )
    # validation fails (unknown_team) BEFORE execution: the registry is never called.
    assert spy.calls == []
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert UNKNOWN_TEAM in _codes(result)
    # the metric may be identified, but nothing is computed and the registry is never called.
    assert result.data is None
    # the tuned suggestion is the single correct team, surfaced but never auto-applied.
    suggestions = tuple(s for issue in result.errors for s in (issue.suggestions or ()))
    assert "Boston Celtics" in suggestions
    assert "New Orleans Pelicans" not in suggestions
    _json_safe(result)


# --- parse failure paths ----------------------------------------------------

@pytest.mark.parametrize("query", [
    "Who is better?",
    "Compare Lakers and Celtics",
    "Warriors last few games",
    "What happened last night?",
])
def test_real_parse_failures_are_safe_and_do_not_answer(query, monkeypatch) -> None:
    def fail_validate(*args, **kwargs):
        raise AssertionError("validator must not run after parse failure")

    monkeypatch.setattr(assistant_module, "validate_intent", fail_validate)
    registry = ExplodingRegistry()
    result = answer_query(
        query,
        clean_df=object(),
        validation_context=object(),
        registry=registry,
    )
    assert result.status in {
        ASSISTANT_STATUS_UNSUPPORTED,
        ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    }
    assert result.status != ASSISTANT_STATUS_ANSWER
    assert registry.calls == []
    assert result.query == query
    _json_safe(result)


# --- validation failure paths ----------------------------------------------

@pytest.mark.parametrize("query,tool_name,issue_code", [
    ("How many points do LA average?", "team_average_points", AMBIGUOUS_TEAM),
    ("How many points do Celics average?", "team_average_points", UNKNOWN_TEAM),
    ("Celtics vs Celtics head to head", "head_to_head", SAME_TEAM_HEAD_TO_HEAD),
])
def test_real_validation_failures_skip_registry(
    query, tool_name, issue_code, assistant_dependencies
) -> None:
    clean, context, _ = assistant_dependencies
    registry = ExplodingRegistry()
    result = answer_query(query, clean_df=clean, validation_context=context, registry=registry)
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert result.tool_name == tool_name
    assert issue_code in _codes(result)
    assert registry.calls == []
    assert result.query == query
    _json_safe(result)


# --- registry execution gating ---------------------------------------------

def test_valid_query_executes_registry_once_after_validation(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    registry = SpyRegistry(_average_points_result())
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_ANSWER
    assert registry.calls == [
        (
            "team_average_points",
            {"team": "Golden State Warriors", "window": 5},
            clean,
        )
    ]


def test_parse_failure_executes_registry_zero_times() -> None:
    registry = SpyRegistry(_average_points_result())
    result = answer_query(
        "Who is better?",
        clean_df=object(),
        validation_context=object(),
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_UNSUPPORTED
    assert registry.calls == []


def test_validation_failure_executes_registry_zero_times(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    registry = SpyRegistry(_average_points_result())
    result = answer_query(
        "How many points do LA average?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert AMBIGUOUS_TEAM in _codes(result)
    assert registry.calls == []


# --- tool-result handling through assistant --------------------------------

def test_fake_registry_ok_result_flows_through_formatter(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    registry = SpyRegistry(_average_points_result())
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_ANSWER
    assert "114.4" in result.message
    assert result.to_dict()["data"]["average_points"] == 114.4


def test_fake_registry_no_data_result_flows_through_formatter(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    registry = SpyRegistry(
        no_data_result(
            "team_average_points",
            result={"team": "Golden State Warriors", "average_points": None, "games_used": 0},
            meta=build_meta(team="Golden State Warriors", games_used=0, window_requested=5),
            warnings=["No games found for team 'Golden State Warriors'."],
        )
    )
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert result.errors[0].code == NO_DATA
    _json_safe(result)


def test_fake_registry_error_result_flows_through_formatter(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    registry = SpyRegistry(
        error_result(
            "team_average_points",
            "Tool execution failed for test.",
            meta=build_meta(team="Golden State Warriors"),
        )
    )
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == EXECUTION_FAILED
    _json_safe(result)


def test_fake_registry_malformed_result_fails_closed(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    registry = SpyRegistry({
        "status": "ok",
        "tool": "team_average_points",
        "result": {"team": "Golden State Warriors"},
        "meta": {},
        "warnings": [],
    })
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == INTERNAL_ERROR
    _json_safe(result)


def test_fake_registry_malformed_warnings_fails_closed(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    registry = SpyRegistry({
        "status": "ok",
        "tool": "team_average_points",
        "result": {"team": "Golden State Warriors", "average_points": 114.4, "games_used": 5},
        "meta": build_meta(team="Golden State Warriors", games_used=5, window_requested=5),
        "warnings": {"message": "bad warning shape"},
    })
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == INTERNAL_ERROR
    assert "malformed tool warnings" in result.message
    _json_safe(result)


# --- internal failure handling ---------------------------------------------

def test_parser_exception_returns_safe_internal_error(monkeypatch) -> None:
    monkeypatch.setattr(
        assistant_module,
        "parse_rule_query",
        lambda query: (_ for _ in ()).throw(RuntimeError("parser exploded")),
    )
    result = answer_query(
        "What is the Warriors record?",
        clean_df=object(),
        validation_context=object(),
        registry=SpyRegistry(_average_points_result()),
    )
    _assert_safe_internal_error(result, "What is the Warriors record?")


def test_validator_exception_returns_safe_internal_error(monkeypatch) -> None:
    monkeypatch.setattr(
        assistant_module,
        "validate_intent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("validator exploded")),
    )
    registry = SpyRegistry(_average_points_result())
    result = answer_query(
        "What is the Warriors record?",
        clean_df=object(),
        validation_context=object(),
        registry=registry,
    )
    _assert_safe_internal_error(result, "What is the Warriors record?")
    assert registry.calls == []


def test_registry_exception_returns_safe_internal_error(assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    result = answer_query(
        "What is the Warriors record?",
        clean_df=clean,
        validation_context=context,
        registry=ExplodingRegistry(),
    )
    _assert_safe_internal_error(result, "What is the Warriors record?")


def test_formatter_exception_returns_safe_internal_error(monkeypatch, assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    monkeypatch.setattr(
        assistant_module,
        "format_tool_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("formatter exploded")),
    )
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=SpyRegistry(_average_points_result()),
    )
    _assert_safe_internal_error(
        result,
        "How many points do the Warriors average over the last 5 games?",
    )


@pytest.mark.parametrize("kwargs,expected_query", [
    (
        {
            "query": 123,
            "clean_df": object(),
            "validation_context": object(),
            "registry": SpyRegistry(_average_points_result()),
        },
        "",
    ),
    (
        {
            "query": "What is the Warriors record?",
            "clean_df": None,
            "validation_context": object(),
            "registry": SpyRegistry(_average_points_result()),
        },
        "What is the Warriors record?",
    ),
    (
        {
            "query": "What is the Warriors record?",
            "clean_df": object(),
            "validation_context": None,
            "registry": SpyRegistry(_average_points_result()),
        },
        "What is the Warriors record?",
    ),
    (
        {
            "query": "What is the Warriors record?",
            "clean_df": object(),
            "validation_context": object(),
            "registry": None,
        },
        "What is the Warriors record?",
    ),
    (
        {
            "query": "What is the Warriors record?",
            "clean_df": object(),
            "validation_context": object(),
            "registry": object(),
        },
        "What is the Warriors record?",
    ),
])
def test_bad_dependencies_return_safe_internal_error(kwargs, expected_query) -> None:
    _assert_safe_internal_error(answer_query(**kwargs), expected_query)


# --- determinism and serialisation -----------------------------------------

@pytest.mark.parametrize("query", [
    "How many points do the Warriors average over the last 5 games?",
    "Celtics vs Heat head to head",
    "Who is better?",
    "How many points do LA average?",
])
def test_real_outputs_are_deterministic_and_json_safe(query, assistant_dependencies) -> None:
    clean, context, registry = assistant_dependencies
    first = answer_query(query, clean_df=clean, validation_context=context, registry=registry).to_dict()
    json.dumps(first)
    mutated = json.loads(json.dumps(first))
    if isinstance(mutated.get("data"), dict):
        mutated["data"]["mutated"] = True
    for _ in range(3):
        assert answer_query(
            query,
            clean_df=clean,
            validation_context=context,
            registry=registry,
        ).to_dict() == first


@pytest.mark.parametrize("tool_result", [
    no_data_result(
        "team_average_points",
        result={"team": "Golden State Warriors", "average_points": None, "games_used": 0},
        meta=build_meta(team="Golden State Warriors", games_used=0, window_requested=5),
        warnings=["No games found for team 'Golden State Warriors'."],
    ),
    {
        "status": "ok",
        "tool": "team_average_points",
        "result": {"team": "Golden State Warriors"},
        "meta": {},
        "warnings": [],
    },
])
def test_fake_registry_outputs_are_deterministic(tool_result, assistant_dependencies) -> None:
    clean, context, _ = assistant_dependencies
    query = "How many points do the Warriors average over the last 5 games?"
    first = answer_query(
        query,
        clean_df=clean,
        validation_context=context,
        registry=SpyRegistry(tool_result),
    ).to_dict()
    json.dumps(first)
    for _ in range(3):
        assert answer_query(
            query,
            clean_df=clean,
            validation_context=context,
            registry=SpyRegistry(tool_result),
        ).to_dict() == first


def test_no_raw_exception_objects_in_assistant_results() -> None:
    result = answer_query(
        "What is the Warriors record?",
        clean_df=None,
        validation_context=object(),
        registry=object(),
    )
    payload = _json_safe(result)
    dumped = json.dumps(payload)
    assert "<" not in dumped
    assert "object at 0x" not in dumped
    assert "Traceback" not in dumped


# --- import and architecture guardrails ------------------------------------

def test_assistant_import_scope_is_lightweight() -> None:
    code = (
        "import sys; import src.assistant;"
        "forbidden = ['pandas', 'numpy', 'src.data_loader', 'src.data_model',"
        " 'src.data_validation', 'src.tools', 'src.llm_query_parser',"
        " 'src.response_formatter_llm', 'src.web', 'src.api', 'src.database',"
        " 'src.rag', 'src.agent'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_assistant_source_has_no_hidden_data_setup_or_tool_logic() -> None:
    source = (REPO_ROOT / "src" / "assistant.py").read_text()
    forbidden = (
        "load_raw_dataset",
        "build_clean_view",
        "build_validation_context",
        "DEFAULT_REGISTRY",
        "src.tools",
        "from src.tools",
        "team_average_points(",
        "average_points_allowed(",
        "team_record(",
        "top_scoring_teams(",
        "head_to_head(",
        "team_efficiency_summary(",
        "resolve_team_name",
        "groupby",
        ".mean(",
        "points_for",
        "points_against",
        "win_flag",
        "import pandas",
        "import numpy",
        "src.llm_query_parser",
        "src.api",
        "src.web",
        "src.database",
        "src.rag",
        "src.agent",
    )
    for text in forbidden:
        assert text not in source


def test_architecture_call_order_is_parse_validate_execute_format(monkeypatch) -> None:
    calls = []
    parsed_intent = ParsedIntent(
        "team_average_points",
        {"team": "Warriors", "window": 5},
        PARSER_MODE_RULE,
        raw_query="q",
    )
    validated_intent = ValidatedIntent(
        "team_average_points",
        {"team": "Golden State Warriors", "window": 5},
        PARSER_MODE_RULE,
        raw_query="q",
    )

    def fake_parse(query):
        calls.append(("parse", query))
        return RuleParseResult.parsed(parsed_intent, raw_query=query)

    def fake_validate(intent, *, context):
        calls.append(("validate", intent.tool_name, context))
        return ValidationResult.valid(validated_intent)

    class OrderedRegistry:
        def execute(self, name, args=None, *, clean_df):
            calls.append(("execute", name, dict(args or {}), clean_df))
            return _average_points_result()

    def fake_format(tool_result, *, query):
        calls.append(("format_tool", tool_result["tool"], query))
        return AssistantResult.answer("formatted", query=query, tool_name=tool_result["tool"])

    monkeypatch.setattr(assistant_module, "parse_rule_query", fake_parse)
    monkeypatch.setattr(assistant_module, "validate_intent", fake_validate)
    monkeypatch.setattr(assistant_module, "format_tool_result", fake_format)
    clean = object()
    context = object()
    result = answer_query("q", clean_df=clean, validation_context=context, registry=OrderedRegistry())
    assert result.status == ASSISTANT_STATUS_ANSWER
    assert calls == [
        ("parse", "q"),
        ("validate", "team_average_points", context),
        ("execute", "team_average_points", {"team": "Golden State Warriors", "window": 5}, clean),
        ("format_tool", "team_average_points", "q"),
    ]


def test_layer_source_boundaries_remain_separated() -> None:
    parser_source = (REPO_ROOT / "src" / "rule_parser.py").read_text()
    validator_source = (REPO_ROOT / "src" / "intent_validator.py").read_text()
    registry_source = (REPO_ROOT / "src" / "tool_registry.py").read_text()
    formatter_source = (REPO_ROOT / "src" / "response_formatter.py").read_text()
    assistant_source = (REPO_ROOT / "src" / "assistant.py").read_text()

    assert "validate_intent" not in parser_source
    assert ".execute(" not in parser_source
    assert "format_tool_result" not in parser_source
    assert "registry.execute" not in validator_source
    assert "format_tool_result" not in validator_source
    assert "parse_rule_query" not in registry_source
    assert "parse_rule_query" not in formatter_source
    assert "validate_intent" not in formatter_source
    assert "registry.execute" not in formatter_source
    assert "parse_rule_query" in assistant_source
    assert "validate_intent" in assistant_source
    assert "format_tool_result" in assistant_source
