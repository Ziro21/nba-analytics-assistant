"""Phase 9E: final assistant-layer acceptance gate.

Cross-cutting proof that the assistant layer (9A contracts, 9B formatter, 9C orchestrator,
9D integration) is complete, deterministic, safe, and correctly scoped. This is an ACCEPTANCE
layer — it asserts whole-of-Phase-9 invariants in one place rather than re-duplicating per-module
unit tests. Oracle values are treated as locked project facts already proven by the tools and
integration tests (never recomputed here).
"""

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
    EXECUTION_FAILED,
    INTERNAL_ERROR,
    NO_DATA,
    SAME_TEAM_HEAD_TO_HEAD,
    UNKNOWN_TEAM,
    AssistantIssue,
    AssistantResult,
)
from src.intent_validator import validate_intent
from src.response_formatter import (
    format_parse_failure,
    format_tool_result,
    format_validation_failure,
)
from src.rule_parser import parse_rule_query
from src.tool_registry import DEFAULT_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSISTANT_SRC = (REPO_ROOT / "src" / "assistant.py").read_text()


# --- fixtures + fakes -------------------------------------------------------

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
    return clean, build_validation_context(clean, registry=DEFAULT_REGISTRY)


def _answer(query, env):
    clean, context = env
    return answer_query(query, clean_df=clean, validation_context=context, registry=DEFAULT_REGISTRY)


class _SpyRegistry:
    def __init__(self, inner):
        self._inner = inner
        self.calls: list[str] = []

    def execute(self, name, args, *, clean_df):
        self.calls.append(name)
        return self._inner.execute(name, args, clean_df=clean_df)


class _CannedRegistry:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def execute(self, name, args, *, clean_df):
        self.calls += 1
        return self._result


class _RaisingRegistry:
    def execute(self, name, args, *, clean_df):
        raise RuntimeError("registry boom")


def _ok_result():
    return {"status": "ok", "tool": "team_record",
            "result": {"team": "Boston Celtics", "record": "10-5", "games_used": 15},
            "meta": {}, "warnings": []}


_NO_DATA_RESULT = {"status": "no_data", "tool": "team_record", "result": {}, "meta": {}, "warnings": []}
_ERROR_RESULT = {"status": "error", "tool": "team_record",
                 "result": {"message": "tool failed"}, "meta": {}, "warnings": []}
_MALFORMED_RESULT = {"status": "ok", "tool": "team_record"}  # missing result/meta/warnings
_MALFORMED_WARNINGS_RESULT = {"status": "ok", "tool": "team_record",
                              "result": {"team": "Boston Celtics", "record": "10-5", "games_used": 15},
                              "meta": {}, "warnings": [{"bad": "dict"}]}  # non-string warning item


# --- 1. public API surface --------------------------------------------------

def test_phase9_public_api_surface_is_locked() -> None:
    assert callable(answer_query)
    assert callable(format_tool_result) and callable(format_parse_failure)
    assert callable(format_validation_failure)
    assert isinstance(AssistantIssue("c", "m"), AssistantIssue)
    assert AssistantResult.answer("ok").status == "answer"
    # no production orchestration shortcut leaked onto the assistant module
    assert not hasattr(assistant_module, "parse_validate_execute")
    for module in ("src.response_formatter_llm", "src.web",
                   "src.api", "src.server", "src.parse_validate_execute"):
        assert importlib.util.find_spec(module) is None


# --- 2. assistant contract acceptance ---------------------------------------

def test_phase9_assistant_result_contract_acceptance() -> None:
    with pytest.raises((TypeError, ValueError)):
        AssistantResult("answer", "m", errors=(AssistantIssue("c", "m"),))  # answer + errors
    with pytest.raises((TypeError, ValueError)):
        AssistantResult("error", "m", errors=())  # non-answer requires an error
    issue = AssistantIssue("c", "m", suggestions=["a", "b"])
    assert issue.suggestions == ("a", "b")  # stored as an immutable tuple


def test_phase9_assistant_result_to_dict_is_json_safe_and_mutation_safe() -> None:
    res = AssistantResult.error(
        "internal", (AssistantIssue(INTERNAL_ERROR, "bad", value={1, 2, 3}),),
        query="q", meta={"x": 1})
    d = res.to_dict()
    json.dumps(d)                                   # non-serialisable set safely represented
    d["errors"].append("hax")
    d["meta"]["x"] = 999
    assert len(res.errors) == 1 and res.to_dict()["meta"] == {"x": 1}


# --- 3. formatter acceptance ------------------------------------------------

def test_phase9_formatter_status_mapping_acceptance() -> None:
    assert format_tool_result(_ok_result()).status == "answer"
    no_data = format_tool_result(_NO_DATA_RESULT)
    assert no_data.status == "clarification_needed" and no_data.errors[0].code == NO_DATA
    err = format_tool_result(_ERROR_RESULT)
    assert err.status == "error" and err.errors[0].code == EXECUTION_FAILED
    malformed = format_tool_result(_MALFORMED_RESULT)
    assert malformed.status == "error" and malformed.errors[0].code == INTERNAL_ERROR


def test_phase9_formatter_rejects_malformed_tool_warnings_final_lock() -> None:
    # the previous Amber: warnings must be a list/tuple of strings; anything else fails closed.
    for bad in (
        {"status": "ok", "tool": "team_record", "result": {}, "meta": {}, "warnings": "oops"},
        _MALFORMED_WARNINGS_RESULT,
        {"status": "ok", "tool": "team_record", "result": {}, "meta": {}, "warnings": [object()]},
    ):
        res = format_tool_result(bad)
        assert res.status == "error" and res.errors[0].code == INTERNAL_ERROR


def test_phase9_formatter_parse_and_validation_failures(env) -> None:
    _, context = env
    pf = format_parse_failure(parse_rule_query("Who is better?"))
    assert pf.status == "unsupported"
    vr = validate_intent(parse_rule_query("How many points do LA average?").parsed_intent, context=context)
    vf = format_validation_failure(vr, tool_name="team_average_points")
    assert vf.status == "clarification_needed" and vf.errors[0].code == AMBIGUOUS_TEAM


# --- 4. production chain reaches registry then formatter --------------------

def test_phase9_production_chain_valid_query_reaches_registry_and_formatter(env) -> None:
    clean, context = env
    spy = _SpyRegistry(DEFAULT_REGISTRY)
    res = answer_query("What is the Warriors record?", clean_df=clean,
                       validation_context=context, registry=spy)
    assert spy.calls == ["team_record"]              # executed exactly once, the validated tool
    assert res.status == "answer"
    json.dumps(res.to_dict())


# --- 5. supported full-chain query acceptance (locked oracles) --------------

@pytest.mark.parametrize("query,tool,must_contain", [
    ("How many points do the Warriors average over the last 5 games?",
     "team_average_points", ["Golden State Warriors", "114.4"]),
    ("How many points do GSW allow over the last 5 games?",
     "average_points_allowed", ["Golden State Warriors", "117.0"]),
    ("What is the Warriors record?", "team_record", ["289-223", "512"]),
    ("Top 5 scoring teams", "top_scoring_teams", ["Atlanta Hawks", "116.13"]),
    ("Celtics vs Heat head to head", "head_to_head",
     ["Boston Celtics", "Miami Heat", "25-14", "39"]),
    ("Boston Celtics efficiency last 10 games", "team_efficiency_summary",
     ["Boston Celtics", "106.98", "101.93"]),
])
def test_phase9_final_supported_queries_return_expected_answers(query, tool, must_contain, env) -> None:
    res = _answer(query, env)
    assert res.status == "answer"
    assert res.tool_name == tool
    for fragment in must_contain:
        assert fragment in res.message, (query, res.message)
    json.dumps(res.to_dict())


# --- 6. failure-path acceptance ---------------------------------------------

@pytest.mark.parametrize("query,status,code", [
    ("Who is better?", "unsupported", None),
    ("Compare LA teams", "clarification_needed", None),    # comparison, no clear second team
    ("Warriors last few games", None, None),               # safe failure; not an answer
    ("How many points do LA average?", "clarification_needed", AMBIGUOUS_TEAM),
    ("How many points do Celics average?", "clarification_needed", UNKNOWN_TEAM),
    ("Celtics vs Celtics head to head", "clarification_needed", SAME_TEAM_HEAD_TO_HEAD),
])
def test_phase9_final_failure_paths_are_safe_and_structured(query, status, code, env) -> None:
    res = _answer(query, env)
    assert res.status != "answer"
    if status is not None:
        assert res.status == status
    if code is not None:
        assert code in [e.code for e in res.errors]
    json.dumps(res.to_dict())


# --- 7. registry execution gating -------------------------------------------

def test_phase9_final_registry_executes_only_after_parse_and_validation_success(env) -> None:
    clean, context = env
    valid = _SpyRegistry(DEFAULT_REGISTRY)
    answer_query("What is the Warriors record?", clean_df=clean, validation_context=context, registry=valid)
    assert len(valid.calls) == 1

    parse_fail = _SpyRegistry(DEFAULT_REGISTRY)
    answer_query("Who is better?", clean_df=clean, validation_context=context, registry=parse_fail)
    assert parse_fail.calls == []

    validation_fail = _SpyRegistry(DEFAULT_REGISTRY)
    answer_query("How many points do LA average?", clean_df=clean,
                 validation_context=context, registry=validation_fail)
    assert validation_fail.calls == []


# --- 8. tool-result handling delegated to the formatter ---------------------

@pytest.mark.parametrize("canned,status,code", [
    (_ok_result(), "answer", None),
    (_NO_DATA_RESULT, "clarification_needed", NO_DATA),
    (_ERROR_RESULT, "error", EXECUTION_FAILED),
    (_MALFORMED_RESULT, "error", INTERNAL_ERROR),
    (_MALFORMED_WARNINGS_RESULT, "error", INTERNAL_ERROR),
])
def test_phase9_final_assistant_delegates_tool_results_to_formatter(canned, status, code, env) -> None:
    clean, context = env
    registry = _CannedRegistry(canned)
    res = answer_query("What is the Warriors record?", clean_df=clean,
                       validation_context=context, registry=registry)
    assert registry.calls == 1
    assert res.status == status
    if code is not None:
        assert any(e.code == code for e in res.errors)


# --- 9. internal fail-closed ------------------------------------------------

@pytest.mark.parametrize("attr", ["parse_rule_query", "validate_intent", "format_tool_result"])
def test_phase9_final_internal_exceptions_fail_closed(attr, env, monkeypatch) -> None:
    clean, context = env

    def _raise(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(assistant_module, attr, _raise)
    res = answer_query("What is the Warriors record?", clean_df=clean,
                       validation_context=context, registry=DEFAULT_REGISTRY)
    assert res.status == "error" and res.errors[0].code == INTERNAL_ERROR
    json.dumps(res.to_dict())


def test_phase9_final_registry_exception_fails_closed(env) -> None:
    clean, context = env
    res = answer_query("What is the Warriors record?", clean_df=clean,
                       validation_context=context, registry=_RaisingRegistry())
    assert res.status == "error" and res.errors[0].code == INTERNAL_ERROR


@pytest.mark.parametrize("query,clean_df,context,registry", [
    (123, "df", "ctx", DEFAULT_REGISTRY),                 # non-string query
    ("q", None, "ctx", DEFAULT_REGISTRY),                 # missing clean_df
    ("q", "df", None, DEFAULT_REGISTRY),                  # missing context
    ("q", "df", "ctx", None),                             # missing registry
    ("q", "df", "ctx", object()),                         # registry missing execute
])
def test_phase9_final_bad_dependencies_fail_closed(query, clean_df, context, registry) -> None:
    res = answer_query(query, clean_df=clean_df, validation_context=context, registry=registry)
    assert res.status == "error" and res.errors[0].code == INTERNAL_ERROR
    json.dumps(res.to_dict())


# --- 10. determinism + serialisation ----------------------------------------

@pytest.mark.parametrize("query", [
    "How many points do the Warriors average over the last 5 games?",
    "Celtics vs Heat head to head", "Who is better?", "How many points do LA average?",
])
def test_phase9_final_outputs_are_deterministic_and_json_serialisable(query, env) -> None:
    first = _answer(query, env).to_dict()
    json.dumps(first)
    for _ in range(3):
        assert _answer(query, env).to_dict() == first


def test_phase9_final_fake_registry_outputs_are_deterministic(env) -> None:
    clean, context = env
    for canned in (_NO_DATA_RESULT, _MALFORMED_RESULT):
        first = answer_query("What is the Warriors record?", clean_df=clean,
                             validation_context=context, registry=_CannedRegistry(canned)).to_dict()
        again = answer_query("What is the Warriors record?", clean_df=clean,
                             validation_context=context, registry=_CannedRegistry(canned)).to_dict()
        assert first == again


# --- 11. import / source scope safety ---------------------------------------

def test_phase9_final_assistant_import_scope_is_safe() -> None:
    code = (
        "import sys; import src.assistant;"
        "forbidden = ['pandas', 'numpy', 'src.data_loader', 'src.data_model', 'src.data_validation',"
        " 'src.tools', 'src.llm_query_parser', 'src.response_formatter_llm', 'src.web', 'src.api',"
        " 'src.database', 'src.rag', 'src.agent'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_phase9_final_assistant_source_has_no_data_loading_or_direct_tool_calls() -> None:
    for forbidden in ("load_raw_dataset", "build_clean_view", "build_validation_context",
                      "DEFAULT_REGISTRY", "import pandas", "from src.tools", "resolve_team_name",
                      ".mean(", ".sum(", "groupby"):
        assert forbidden not in ASSISTANT_SRC, f"assistant.py must not contain {forbidden!r}"


# --- 12. out-of-scope module absence ----------------------------------------

def test_phase9_final_no_out_of_scope_production_modules_exist() -> None:
    for module in ("src.response_formatter_llm", "src.web", "src.api",
                   "src.database", "src.rag", "src.agent", "src.server",
                   "src.parse_validate_execute", "src.rule_parser_validation_integration"):
        assert importlib.util.find_spec(module) is None, f"{module} must not exist in Phase 9"
