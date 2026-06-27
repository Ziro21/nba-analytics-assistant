"""Phase 9C tests: production assistant orchestrator."""

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
)
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.intent_types import PARSER_MODE_RULE, ValidatedIntent, ValidationResult
from src.tool_registry import DEFAULT_REGISTRY
from src.tool_results import build_meta, error_result, no_data_result, ok_result
from src.validation_context import build_validation_context

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pipeline():
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    context = build_validation_context(clean, registry=DEFAULT_REGISTRY)
    return clean, context


class StaticRegistry:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def execute(self, name, args=None, *, clean_df):
        self.calls.append((name, dict(args or {}), clean_df))
        return self.result


class RaisingRegistry:
    def __init__(self):
        self.calls = []

    def execute(self, name, args=None, *, clean_df):
        self.calls.append((name, args, clean_df))
        raise AssertionError("registry.execute should not have been called")


def _codes(result) -> list[str]:
    return [issue.code for issue in result.errors]


def _assert_json_safe(result) -> None:
    json.dumps(result.to_dict())


def _assert_internal_error(result, *, query: str = "") -> None:
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors
    assert result.errors[0].code == INTERNAL_ERROR
    assert result.query == query
    out = json.dumps(result.to_dict())
    assert "Traceback" not in out
    assert "RuntimeError" not in out


# --- import / scope safety --------------------------------------------------

def test_assistant_import_is_lightweight() -> None:
    code = (
        "import sys; import src.assistant;"
        "forbidden = ['pandas', 'numpy', 'src.data_loader', 'src.data_model',"
        " 'src.data_validation', 'src.tool_registry', 'src.tools', 'src.llm_query_parser',"
        " 'src.response_formatter_llm', 'src.web', 'src.api', 'src.database', 'src.rag',"
        " 'src.agent'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --- successful full-chain answers -----------------------------------------

@pytest.mark.parametrize("query,tool_name,message_parts,data_checks", [
    (
        "How many points do the Warriors average over the last 5 games?",
        "team_average_points",
        ("Golden State Warriors", "114.4", "last 5 games"),
        {"team": "Golden State Warriors", "average_points": 114.4, "games_used": 5},
    ),
    (
        "How many points do GSW allow over the last 5 games?",
        "average_points_allowed",
        ("Golden State Warriors", "117.0", "last 5 games"),
        {"team": "Golden State Warriors", "average_points_allowed": 117.0, "games_used": 5},
    ),
    (
        "What is the Warriors record?",
        "team_record",
        ("Golden State Warriors", "289-223", "512"),
        {"team": "Golden State Warriors", "record": "289-223", "games_used": 512},
    ),
    (
        "Top 5 scoring teams",
        "top_scoring_teams",
        ("Atlanta Hawks", "116.13"),
        {"teams_returned": 5, "n_requested": 5},
    ),
    (
        "Celtics vs Heat head to head",
        "head_to_head",
        ("Boston Celtics", "Miami Heat", "25-14", "39"),
        {"team_a": "Boston Celtics", "team_b": "Miami Heat", "record": "25-14", "meetings": 39},
    ),
    (
        "Boston Celtics efficiency last 10 games",
        "team_efficiency_summary",
        ("Boston Celtics", "106.98", "101.93"),
        {"team": "Boston Celtics", "games_used": 10},
    ),
])
def test_supported_queries_answer_end_to_end(
    query, tool_name, message_parts, data_checks, pipeline
) -> None:
    clean, context = pipeline
    result = answer_query(query, clean_df=clean, validation_context=context, registry=DEFAULT_REGISTRY)
    assert result.status == ASSISTANT_STATUS_ANSWER
    assert result.tool_name == tool_name
    assert result.query == query
    assert result.errors == ()
    for text in message_parts:
        assert text in result.message
    for key, expected in data_checks.items():
        assert result.to_dict()["data"][key] == expected
    _assert_json_safe(result)


def test_valid_path_calls_validator_and_registry(monkeypatch) -> None:
    calls = []
    clean = object()
    context = object()

    def fake_validate(parsed_intent, *, context):
        calls.append(("validate", parsed_intent.tool_name, context))
        return ValidationResult.valid(
            ValidatedIntent(
                "team_average_points",
                {"team": "Golden State Warriors", "window": 5},
                PARSER_MODE_RULE,
                raw_query=parsed_intent.raw_query,
            )
        )

    class SpyRegistry:
        def execute(self, name, args=None, *, clean_df):
            calls.append(("execute", name, dict(args or {}), clean_df))
            return ok_result(
                "team_average_points",
                {"team": "Golden State Warriors", "average_points": 114.4, "games_used": 5},
                meta=build_meta(
                    team="Golden State Warriors", games_used=5, window_requested=5
                ),
            )

    monkeypatch.setattr(assistant_module, "validate_intent", fake_validate)
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=SpyRegistry(),
    )
    assert result.status == ASSISTANT_STATUS_ANSWER
    assert calls == [
        ("validate", "team_average_points", context),
        (
            "execute",
            "team_average_points",
            {"team": "Golden State Warriors", "window": 5},
            clean,
        ),
    ]


# --- parse failure path -----------------------------------------------------

@pytest.mark.parametrize("query", [
    "Who is better?",
    "Compare Lakers and Celtics",
    "Warriors last few games",
])
def test_parse_failures_skip_validation_and_execution(query, monkeypatch) -> None:
    def fail_validate(*args, **kwargs):
        raise AssertionError("validate_intent should not run after parse failure")

    monkeypatch.setattr(assistant_module, "validate_intent", fail_validate)
    result = answer_query(
        query,
        clean_df=object(),
        validation_context=object(),
        registry=RaisingRegistry(),
    )
    assert result.status in {
        ASSISTANT_STATUS_UNSUPPORTED,
        ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    }
    assert result.query == query
    _assert_json_safe(result)


# --- validation failure path ------------------------------------------------

@pytest.mark.parametrize("query,issue_code", [
    ("How many points do LA average?", AMBIGUOUS_TEAM),
    ("How many points do Celics average?", UNKNOWN_TEAM),
    ("Celtics vs Celtics head to head", SAME_TEAM_HEAD_TO_HEAD),
])
def test_validation_failures_skip_registry_execution(query, issue_code, pipeline) -> None:
    clean, context = pipeline
    registry = RaisingRegistry()
    result = answer_query(query, clean_df=clean, validation_context=context, registry=registry)
    assert result.status == ASSISTANT_STATUS_CLARIFICATION_NEEDED
    assert issue_code in _codes(result)
    assert registry.calls == []
    assert result.query == query
    _assert_json_safe(result)


# --- tool-result path -------------------------------------------------------

def test_registry_no_data_result_formats_as_clarification(pipeline) -> None:
    clean, context = pipeline
    registry = StaticRegistry(
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
    assert registry.calls
    _assert_json_safe(result)


def test_registry_error_result_formats_as_assistant_error(pipeline) -> None:
    clean, context = pipeline
    registry = StaticRegistry(
        error_result(
            "team_record",
            "window must be a positive integer, got 0",
            meta=build_meta(team="Golden State Warriors"),
        )
    )
    result = answer_query(
        "What is the Warriors record?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == EXECUTION_FAILED
    assert registry.calls
    _assert_json_safe(result)


def test_malformed_registry_result_formats_as_internal_error(pipeline) -> None:
    clean, context = pipeline
    registry = StaticRegistry(
        {
            "status": "ok",
            "tool": "team_average_points",
            "result": {"team": "Golden State Warriors"},
            "meta": {},
            "warnings": [],
        }
    )
    result = answer_query(
        "How many points do the Warriors average over the last 5 games?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    assert result.status == ASSISTANT_STATUS_ERROR
    assert result.errors[0].code == INTERNAL_ERROR
    assert registry.calls
    _assert_json_safe(result)


# --- internal failure handling ---------------------------------------------

def test_parser_exception_fails_closed(monkeypatch) -> None:
    def boom(query):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(assistant_module, "parse_rule_query", boom)
    result = answer_query(
        "What is the Warriors record?",
        clean_df=object(),
        validation_context=object(),
        registry=StaticRegistry({}),
    )
    _assert_internal_error(result, query="What is the Warriors record?")


def test_validator_exception_fails_closed(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(assistant_module, "validate_intent", boom)
    result = answer_query(
        "What is the Warriors record?",
        clean_df=object(),
        validation_context=object(),
        registry=StaticRegistry({}),
    )
    _assert_internal_error(result, query="What is the Warriors record?")


def test_registry_exception_fails_closed(pipeline) -> None:
    clean, context = pipeline

    class ExplodingRegistry:
        def execute(self, *args, **kwargs):
            raise RuntimeError("registry exploded")

    result = answer_query(
        "What is the Warriors record?",
        clean_df=clean,
        validation_context=context,
        registry=ExplodingRegistry(),
    )
    _assert_internal_error(result, query="What is the Warriors record?")


def test_parse_failure_formatter_exception_fails_closed(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("parse formatter exploded")

    monkeypatch.setattr(assistant_module, "format_parse_failure", boom)
    result = answer_query(
        "Who is better?",
        clean_df=object(),
        validation_context=object(),
        registry=StaticRegistry({}),
    )
    _assert_internal_error(result, query="Who is better?")


def test_validation_failure_formatter_exception_fails_closed(monkeypatch, pipeline) -> None:
    clean, context = pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("validation formatter exploded")

    monkeypatch.setattr(assistant_module, "format_validation_failure", boom)
    result = answer_query(
        "How many points do LA average?",
        clean_df=clean,
        validation_context=context,
        registry=RaisingRegistry(),
    )
    _assert_internal_error(result, query="How many points do LA average?")


def test_tool_result_formatter_exception_fails_closed(monkeypatch, pipeline) -> None:
    clean, context = pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("tool formatter exploded")

    monkeypatch.setattr(assistant_module, "format_tool_result", boom)
    registry = StaticRegistry(
        ok_result(
            "team_record",
            {
                "team": "Golden State Warriors",
                "wins": 289,
                "losses": 223,
                "record": "289-223",
                "games_used": 512,
                "win_percentage": 0.564453125,
            },
            meta=build_meta(team="Golden State Warriors", games_used=512),
        )
    )
    result = answer_query(
        "What is the Warriors record?",
        clean_df=clean,
        validation_context=context,
        registry=registry,
    )
    _assert_internal_error(result, query="What is the Warriors record?")


# --- dependency validation --------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"query": 123, "clean_df": object(), "validation_context": object(), "registry": StaticRegistry({})},
    {"query": "What is the Warriors record?", "clean_df": None, "validation_context": object(), "registry": StaticRegistry({})},
    {"query": "What is the Warriors record?", "clean_df": object(), "validation_context": None, "registry": StaticRegistry({})},
    {"query": "What is the Warriors record?", "clean_df": object(), "validation_context": object(), "registry": object()},
])
def test_bad_dependencies_fail_closed(kwargs) -> None:
    query = kwargs["query"] if isinstance(kwargs["query"], str) else ""
    result = answer_query(**kwargs)
    _assert_internal_error(result, query=query)
    _assert_json_safe(result)


# --- determinism ------------------------------------------------------------

def test_repeated_calls_are_deterministic_for_real_paths(pipeline) -> None:
    clean, context = pipeline
    for query in (
        "How many points do the Warriors average over the last 5 games?",
        "Who is better?",
        "How many points do LA average?",
    ):
        first = answer_query(
            query, clean_df=clean, validation_context=context, registry=DEFAULT_REGISTRY
        ).to_dict()
        for _ in range(3):
            assert answer_query(
                query, clean_df=clean, validation_context=context, registry=DEFAULT_REGISTRY
            ).to_dict() == first


def test_repeated_calls_are_deterministic_for_registry_no_data(pipeline) -> None:
    clean, context = pipeline
    registry = StaticRegistry(
        no_data_result(
            "team_average_points",
            result={"team": "Golden State Warriors", "average_points": None, "games_used": 0},
            meta=build_meta(team="Golden State Warriors", games_used=0, window_requested=5),
            warnings=["No games found for team 'Golden State Warriors'."],
        )
    )
    query = "How many points do the Warriors average over the last 5 games?"
    first = answer_query(query, clean_df=clean, validation_context=context, registry=registry).to_dict()
    for _ in range(3):
        assert answer_query(query, clean_df=clean, validation_context=context, registry=registry).to_dict() == first


# --- no hidden statistics / direct tool calls -------------------------------

def test_assistant_source_has_no_data_loading_or_direct_tool_logic() -> None:
    source = (REPO_ROOT / "src" / "assistant.py").read_text()
    for forbidden in (
        "import pandas",
        "import numpy",
        "src.tools",
        "from src.tools",
        "load_raw_dataset",
        "build_clean_view",
        "validate_dataset",
        "resolve_team_name",
        "groupby",
        ".mean(",
        "points_for",
        "points_against",
        "win_flag",
        "ortg",
        "drtg",
        "src.llm_query_parser",
        "src.api",
        "src.web",
        "src.database",
        "src.rag",
        "src.agent",
    ):
        assert forbidden not in source


def test_no_future_llm_or_pipeline_modules_exist() -> None:
    for module in (
        "src.response_formatter_llm",  # LLM response generation stays out of scope
        "src.parse_validate_execute",
        "src.api",
        "src.web",
        "src.database",
        "src.rag",
        "src.agent",
        "src.server",
    ):
        assert importlib.util.find_spec(module) is None
