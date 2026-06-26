"""Phase 8E: parser -> validator integration GATE (tests only).

Proves the deterministic parser hands off correctly to the Phase 7 validator (and that validated
intents are registry-ready), WITHOUT introducing a production orchestration module — that belongs
to a later assistant layer. The integration is exercised via a private test helper that calls
``parse_rule_query`` then ``validate_intent`` directly.
"""

from __future__ import annotations

import importlib.util

import pytest

from src.intent_types import (
    AMBIGUOUS_TEAM,
    INVALID_SPECIAL_TEAM,
    SAME_TEAM_HEAD_TO_HEAD,
    UNKNOWN_TEAM,
)
import tests.test_rule_parser_validation_integration as self_module
from src.intent_validator import validate_intent
from src.rule_parser import parse_rule_query
from src.rule_query_catalogue import SUPPORTED_QUERY_EXAMPLES, UNSUPPORTED_QUERY_EXAMPLES
from src.team_surface_catalogue import CANONICAL_TEAM_NAMES
from src.tool_registry import DEFAULT_REGISTRY

CANONICAL = set(CANONICAL_TEAM_NAMES)
TEAM_ARG_NAMES = ("team", "team_a", "team_b")


@pytest.fixture(scope="module")
def env():
    from src.data_loader import load_raw_dataset
    from src.data_model import build_clean_view, validate_clean_view
    from src.data_validation import validate_dataset
    from src.validation_context import build_validation_context

    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    context = build_validation_context(clean, registry=DEFAULT_REGISTRY)
    return context, clean


# --- private integration helpers (no production orchestration module exists) -------------

def _parse_then_validate(query, *, context):
    """parse -> (only if parsed) validate. Returns (parse_result, validation_result|None)."""
    parse_result = parse_rule_query(query)
    if parse_result.status != "parsed":
        return parse_result, None
    return parse_result, validate_intent(parse_result.parsed_intent, context=context)


def _run_pipeline(query, *, context, clean):
    """parse -> validate -> (only if valid) execute. Returns (parse, validation, exec|None)."""
    parse_result, validation_result = _parse_then_validate(query, context=context)
    if validation_result is None or not validation_result.is_valid:
        return parse_result, validation_result, None
    vi = validation_result.validated_intent
    exec_result = DEFAULT_REGISTRY.execute(vi.tool_name, dict(vi.arguments), clean_df=clean)
    return parse_result, validation_result, exec_result


def _codes(validation_result):
    return [e.code for e in validation_result.errors]


# --- supported catalogue: parse + validate end-to-end -----------------------

@pytest.mark.parametrize("example", SUPPORTED_QUERY_EXAMPLES, ids=lambda e: e.query)
def test_supported_examples_validate(example, env) -> None:
    context, _ = env
    parse_result, validation = _parse_then_validate(example.query, context=context)
    assert parse_result.status == "parsed"
    assert validation.is_valid is True, validation.to_dict()
    args = dict(validation.validated_intent.arguments)
    assert validation.validated_intent.tool_name == example.expected_tool
    for name in TEAM_ARG_NAMES:                      # raw team -> canonical franchise
        if name in args:
            assert args[name] in CANONICAL
    for name in ("window", "n", "season_id"):        # numeric args preserved verbatim
        if name in dict(example.expected_arguments):
            assert args[name] == dict(example.expected_arguments)[name]


# --- canonicalisation specifics + the closed bare-city gap ------------------

@pytest.mark.parametrize("query,tool,team", [
    ("What is the Warriors record?", "team_record", "Golden State Warriors"),
    ("What is GSW averaging over the last 5 games?", "team_average_points", "Golden State Warriors"),
    ("How many points are Boston allowing?", "average_points_allowed", "Boston Celtics"),
    ("Denver record last 10 games", "team_record", "Denver Nuggets"),
])
def test_team_is_canonicalised(query, tool, team, env) -> None:
    context, _ = env
    _, validation = _parse_then_validate(query, context=context)
    assert validation.is_valid and validation.validated_intent.tool_name == tool
    assert dict(validation.validated_intent.arguments)["team"] == team


def test_bare_city_examples_resolve_end_to_end(env) -> None:
    context, _ = env
    _, validation = _parse_then_validate("Boston against Miami", context=context)
    assert validation.is_valid
    assert dict(validation.validated_intent.arguments) == {
        "team_a": "Boston Celtics", "team_b": "Miami Heat"}


# --- validation failures on parsed intents (validator owns these) -----------

@pytest.mark.parametrize("query,code", [
    ("How many points do LA average?", AMBIGUOUS_TEAM),
    ("How many points do Celics average?", UNKNOWN_TEAM),
    ("How many points do Team World average?", INVALID_SPECIAL_TEAM),
    ("Celtics vs Celtics head to head", SAME_TEAM_HEAD_TO_HEAD),
])
def test_parsed_but_invalid(query, code, env) -> None:
    context, _ = env
    parse_result, validation = _parse_then_validate(query, context=context)
    assert parse_result.status == "parsed"          # parser emitted raw candidate
    assert validation.is_valid is False
    assert code in _codes(validation)


# --- parse failures never reach the validator -------------------------------

@pytest.mark.parametrize("query", [
    "Who is better?", "Compare Lakers and Celtics", "Celtics vs", "Warriors recent form",
])
def test_parse_failures_have_no_validation(query, env) -> None:
    context, _ = env
    parse_result, validation = _parse_then_validate(query, context=context)
    assert parse_result.status != "parsed" and validation is None


def test_no_unsupported_example_validates(env) -> None:
    context, _ = env
    for ex in UNSUPPORTED_QUERY_EXAMPLES:
        _, validation = _parse_then_validate(ex.query, context=context)
        assert validation is None or validation.is_valid is False


# --- boundary safety: spies prove what is and isn't called ------------------

def test_parse_failure_does_not_call_validate_intent(env, monkeypatch) -> None:
    context, _ = env
    calls = []
    monkeypatch.setattr(self_module, "validate_intent",
                        lambda *a, **k: calls.append(1))
    _parse_then_validate("Who is better?", context=context)  # parse fails
    assert calls == []  # validator never invoked


def test_invalid_validation_does_not_call_registry_execute(env, monkeypatch) -> None:
    context, clean = env
    calls = []

    def _spy_execute(*a, **k):
        calls.append((a, k))
        raise AssertionError("registry.execute must not run for an invalid intent")

    monkeypatch.setattr(DEFAULT_REGISTRY, "execute", _spy_execute)
    parse_result, validation, exec_result = _run_pipeline(
        "How many points do LA average?", context=context, clean=clean)  # ambiguous -> invalid
    assert validation is not None and not validation.is_valid
    assert exec_result is None and calls == []


# --- closed loop: validated intents are registry-ready ----------------------

@pytest.mark.parametrize("example", SUPPORTED_QUERY_EXAMPLES, ids=lambda e: e.query)
def test_validated_intents_execute_without_error(example, env) -> None:
    context, clean = env
    _, validation, exec_result = _run_pipeline(example.query, context=context, clean=clean)
    assert validation.is_valid                       # parsed + validated
    assert exec_result["status"] in {"ok", "no_data"}  # accepted by the registry, never "error"
    assert exec_result["tool"] == example.expected_tool


# --- determinism + module-boundary guard ------------------------------------

def test_pipeline_is_deterministic(env) -> None:
    context, _ = env
    for query in ("What is GSW averaging over the last 5 games?", "How many points do LA average?"):
        first = _parse_then_validate(query, context=context)
        for _ in range(3):
            again = _parse_then_validate(query, context=context)
            assert again[0].to_dict() == first[0].to_dict()
            assert (again[1].to_dict() if again[1] else None) == (
                first[1].to_dict() if first[1] else None)


def test_no_production_parser_validator_integration_module_exists() -> None:
    # Phase 8E stays a tests-only gate; production orchestration is deferred to the assistant layer.
    assert importlib.util.find_spec("src.rule_parser_validation_integration") is None
