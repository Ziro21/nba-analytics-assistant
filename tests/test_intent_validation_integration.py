"""Phase 7D integration review: the full validation layer end to end.

Proves: parser-mode invariance, the closed loop validate -> registry.execute -> oracle,
that invalid intents never execute, that the validator stays execution-free and data-free,
JSON-serialisability, and scope. Registry execution happens ONLY here (in tests), never in
`validate_intent`. No production orchestrator/parser/formatter/assistant is created.
"""

from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest

from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
import src.tool_registry as registry_module
from src.intent_types import ParsedIntent
from src.intent_validator import validate_intent
from src.tool_registry import DEFAULT_REGISTRY, execute
from src.validation_context import build_validation_context

FORBIDDEN_MODULES = ("src.query_parser",)


@pytest.fixture(scope="module")
def pipeline():
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    context = build_validation_context(clean, registry=DEFAULT_REGISTRY)
    return clean, context


def _outcome_key(result):
    """A parser-mode-agnostic fingerprint of a ValidationResult."""
    return (
        result.is_valid,
        tuple(sorted(e.code for e in result.errors)),
        tuple(sorted(w.code for w in result.warnings)),
        dict(result.validated_intent.arguments) if result.is_valid else None,
    )


# --- 2. Parser-mode invariance ---------------------------------------------

@pytest.mark.parametrize("tool,args", [
    ("team_average_points", {"team": "golden state warriors", "window": 5}),  # valid
    ("team_average_points", {"team": "Celics", "window": 5}),                 # unknown
    ("team_average_points", {"team": "LA", "window": 5}),                     # ambiguous
    ("team_average_points", {"team": "Boston Celtics", "window": "5"}),       # invalid type
    ("head_to_head", {"team_a": "Boston Celtics", "team_b": "boston celtics"}),  # same team
])
def test_parser_mode_invariance(pipeline, tool, args) -> None:
    _, context = pipeline
    rule = validate_intent(ParsedIntent(tool, args, "rule"), context=context)
    llm = validate_intent(ParsedIntent(tool, args, "llm"), context=context)
    assert _outcome_key(rule) == _outcome_key(llm)
    if rule.is_valid:
        assert (rule.validated_intent.parser_mode, llm.validated_intent.parser_mode) == ("rule", "llm")


# --- 3. Closed loop: validate -> execute -> oracle --------------------------

def _validate_then_execute(intent, clean, context):
    result = validate_intent(intent, context=context)
    assert result.is_valid, [e.code for e in result.errors]
    tool_result = execute(
        result.validated_intent.tool_name, result.validated_intent.arguments, clean_df=clean
    )
    json.dumps(tool_result)  # registry output serialises
    return result, tool_result


def test_closed_loop_team_average_points(pipeline) -> None:
    clean, context = pipeline
    result, out = _validate_then_execute(
        ParsedIntent("team_average_points", {"team": "golden state warriors", "window": 5}, "llm"),
        clean, context,
    )
    assert dict(result.validated_intent.arguments) == {"team": "Golden State Warriors", "window": 5}
    assert out["status"] == "ok" and out["tool"] == "team_average_points"
    assert out["result"]["average_points"] == pytest.approx(114.4, abs=1e-2)


def test_closed_loop_average_points_allowed(pipeline) -> None:
    clean, context = pipeline
    result, out = _validate_then_execute(
        ParsedIntent("average_points_allowed", {"team": "gsw", "window": 5}, "rule"), clean, context,
    )
    assert result.validated_intent.arguments["team"] == "Golden State Warriors"
    assert out["status"] == "ok"
    assert out["result"]["average_points_allowed"] == pytest.approx(117.0, abs=1e-2)


def test_closed_loop_team_record(pipeline) -> None:
    clean, context = pipeline
    _, out = _validate_then_execute(
        ParsedIntent("team_record", {"team": "Golden State Warriors"}, "llm"), clean, context,
    )
    assert (out["result"]["wins"], out["result"]["losses"]) == (289, 223)


def test_closed_loop_top_scoring_teams(pipeline) -> None:
    clean, context = pipeline
    _, out = _validate_then_execute(
        ParsedIntent("top_scoring_teams", {"n": 5}, "rule"), clean, context,
    )
    first = out["result"]["teams"][0]
    assert out["status"] == "ok"
    assert first["team"] == "Atlanta Hawks" and round(first["average_points"], 2) == 116.13


def test_closed_loop_head_to_head(pipeline) -> None:
    clean, context = pipeline
    result, out = _validate_then_execute(
        ParsedIntent("head_to_head", {"team_a": "celtics", "team_b": "heat"}, "llm"), clean, context,
    )
    assert dict(result.validated_intent.arguments) == {"team_a": "Boston Celtics", "team_b": "Miami Heat"}
    assert (out["result"]["meetings"], out["result"]["team_a_wins"], out["result"]["team_b_wins"]) == (39, 25, 14)
    assert out["result"]["record"] == "25-14"


def test_closed_loop_efficiency_summary(pipeline) -> None:
    clean, context = pipeline
    _, out = _validate_then_execute(
        ParsedIntent("team_efficiency_summary", {"team": "Boston Celtics", "window": 10}, "rule"),
        clean, context,
    )
    assert out["status"] == "ok"
    assert round(out["result"]["average_ortg"], 2) == 106.98
    assert round(out["result"]["average_drtg"], 2) == 101.93


# --- 4. Invalid intents must not execute ------------------------------------

def _gated_execute(result, clean):
    """The caller gate: execute (through the registry module attribute) only if valid."""
    if result.is_valid:
        return registry_module.execute(
            result.validated_intent.tool_name, result.validated_intent.arguments, clean_df=clean
        )
    return None


@pytest.mark.parametrize("tool,args,code", [
    ("team_average_points", {"team": "Celics", "window": 5}, "unknown_team"),
    ("team_average_points", {"team": "LA", "window": 5}, "ambiguous_team"),
    ("head_to_head", {"team_a": "Boston Celtics", "team_b": "boston celtics"}, "same_team_head_to_head"),
])
def test_invalid_intents_do_not_execute(pipeline, monkeypatch, tool, args, code) -> None:
    clean, context = pipeline
    calls: list = []
    monkeypatch.setattr(registry_module, "execute", lambda *a, **k: calls.append((a, k)))
    result = validate_intent(ParsedIntent(tool, args, "rule"), context=context)
    assert not result.is_valid and code in {e.code for e in result.errors}
    _gated_execute(result, clean)  # gate calls through the monkeypatched attribute
    assert calls == []  # invalid -> never executed
    json.dumps(result.to_dict())


def test_valid_gate_executes_through_monkeypatched_module(pipeline, monkeypatch) -> None:
    # Positive control: a valid intent's gate DOES reach the (spied) execute.
    clean, context = pipeline
    calls: list = []
    monkeypatch.setattr(registry_module, "execute", lambda *a, **k: calls.append((a, k)))
    result = validate_intent(ParsedIntent("team_average_points", {"team": "gsw", "window": 5}, "llm"), context=context)
    assert result.is_valid
    _gated_execute(result, clean)
    assert len(calls) == 1


def test_registry_execute_accepts_validated_intent_arguments_directly(pipeline) -> None:
    from types import MappingProxyType

    clean, context = pipeline
    result = validate_intent(ParsedIntent("team_record", {"team": "Golden State Warriors"}, "rule"), context=context)
    assert isinstance(result.validated_intent.arguments, MappingProxyType)
    out = execute(result.validated_intent.tool_name, result.validated_intent.arguments, clean_df=clean)
    assert out["status"] == "ok" and out["result"]["record"] == "289-223"


# --- 5/6. Validator stays execution-free and data-free ----------------------

def test_validate_intent_does_not_call_execute(pipeline, monkeypatch) -> None:
    _, context = pipeline
    monkeypatch.setattr(
        "src.tool_registry.execute",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("validate must not execute")),
    )
    result = validate_intent(
        ParsedIntent("team_average_points", {"team": "gsw", "window": 5}, "llm"), context=context
    )
    assert result.is_valid  # validated without executing anything


def test_validate_intent_does_not_load_data(pipeline, monkeypatch) -> None:
    _, context = pipeline
    monkeypatch.setattr(
        pd, "read_csv",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("validate must not read the CSV")),
    )
    result = validate_intent(
        ParsedIntent("team_record", {"team": "lakers"}, "rule"), context=context
    )
    assert result.is_valid and result.validated_intent.arguments["team"] == "Los Angeles Lakers"


def test_context_requires_clean_df_and_does_not_store_it(pipeline) -> None:
    clean, context = pipeline
    with pytest.raises(TypeError):
        build_validation_context(registry=DEFAULT_REGISTRY)  # clean_df is required
    for value in vars(context).values():
        assert not isinstance(value, pd.DataFrame)


# --- 7. JSON serialisation (valid + invalid) --------------------------------

def test_validation_results_serialise(pipeline) -> None:
    _, context = pipeline
    valid = validate_intent(ParsedIntent("team_average_points", {"team": "gsw", "window": 5}, "llm"), context=context)
    invalid = validate_intent(ParsedIntent("team_average_points", {"team": "Celics", "window": 5}, "llm"), context=context)
    json.dumps(valid.to_dict())
    json.dumps(invalid.to_dict())


# --- 8. Scope guard ---------------------------------------------------------

def test_no_query_or_llm_parser_modules(pipeline) -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"
