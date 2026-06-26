"""Phase 10A tests: assistant runtime / bootstrap layer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import src.assistant_runtime as runtime_module
from src.assistant_runtime import AssistantRuntime, build_default_runtime
from src.assistant_types import AssistantResult
from src.config import DATASET_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_SRC = (REPO_ROOT / "src" / "assistant_runtime.py").read_text()

BOOTSTRAP_STEPS = (
    "load_raw_dataset", "validate_dataset", "build_clean_view",
    "validate_clean_view", "build_validation_context",
)


@pytest.fixture(scope="module")
def runtime():
    return build_default_runtime()


# --- 10.1 runtime builds successfully ---------------------------------------

def test_runtime_builds_with_dependencies(runtime) -> None:
    assert isinstance(runtime, AssistantRuntime)
    assert runtime.clean_df is not None
    assert runtime.validation_context is not None
    assert runtime.registry is not None
    assert len(runtime.clean_df) > 0
    assert set(runtime.validation_context.canonical_teams)  # 30 franchises derived from data


def test_runtime_registry_can_execute(runtime) -> None:
    result = runtime.registry.execute(
        "team_record", {"team": "Boston Celtics"}, clean_df=runtime.clean_df)
    assert result["status"] in {"ok", "no_data"}
    assert result["tool"] == "team_record"


# --- 10.2 runtime answer method works ---------------------------------------

def test_runtime_answer_supported_query(runtime) -> None:
    result = runtime.answer("How many points do the Warriors average over the last 5 games?")
    assert isinstance(result, AssistantResult)
    assert result.status == "answer"
    assert result.tool_name == "team_average_points"
    assert "Golden State Warriors" in result.message
    assert "114.4" in result.message
    json.dumps(result.to_dict())


# --- 10.3 representative safe failures --------------------------------------

@pytest.mark.parametrize("query,status,code", [
    ("Who is better?", "unsupported", None),
    ("How many points do LA average?", "clarification_needed", "ambiguous_team"),
    ("How many points do Celics average?", "clarification_needed", "unknown_team"),
    ("Celtics vs Celtics head to head", "clarification_needed", "same_team_head_to_head"),
])
def test_runtime_safe_failures(runtime, query, status, code) -> None:
    result = runtime.answer(query)
    assert result.status == status
    if code is not None:
        assert code in [e.code for e in result.errors]
    json.dumps(result.to_dict())


# --- 10.4 answer delegates to answer_query ----------------------------------

def test_answer_delegates_to_answer_query(runtime, monkeypatch) -> None:
    captured = {}
    sentinel = AssistantResult.answer("delegated")

    def _spy(query, *, clean_df, validation_context, registry):
        captured.update(query=query, clean_df=clean_df,
                        validation_context=validation_context, registry=registry)
        return sentinel

    monkeypatch.setattr(runtime_module, "answer_query", _spy)
    out = runtime.answer("any query")
    assert out is sentinel
    assert captured["query"] == "any query"
    assert captured["clean_df"] is runtime.clean_df
    assert captured["validation_context"] is runtime.validation_context
    assert captured["registry"] is runtime.registry


def test_runtime_does_not_expose_pipeline_internals() -> None:
    # the runtime imports only answer_query, never the parser/validator/formatter functions.
    for name in ("parse_rule_query", "validate_intent", "format_tool_result",
                 "format_parse_failure", "format_validation_failure"):
        assert not hasattr(runtime_module, name)


# --- 10.5 bootstrap uses the existing project pipeline ----------------------

def test_build_default_runtime_uses_project_pipeline(monkeypatch) -> None:
    calls: list[str] = []
    for name in BOOTSTRAP_STEPS:
        real = getattr(runtime_module, name)

        def _make(step, fn):
            def _spy(*args, **kwargs):
                calls.append(step)
                return fn(*args, **kwargs)
            return _spy

        monkeypatch.setattr(runtime_module, name, _make(name, real))

    rt = build_default_runtime()
    assert isinstance(rt, AssistantRuntime)
    assert calls == list(BOOTSTRAP_STEPS)  # called once each, in pipeline order


# --- 10.6 setup failure raises (no partial runtime, no AssistantResult) -----

def test_build_default_runtime_setup_failure_raises(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("dataset is invalid")

    monkeypatch.setattr(runtime_module, "validate_dataset", _boom)
    with pytest.raises(RuntimeError):
        build_default_runtime()


# --- optional dataset_path parameter ----------------------------------------

def test_build_default_runtime_with_explicit_dataset_path() -> None:
    rt = build_default_runtime(dataset_path=DATASET_PATH)
    assert isinstance(rt, AssistantRuntime)
    result = rt.answer("What is the Warriors record?")
    assert result.status == "answer" and result.tool_name == "team_record"


def test_build_default_runtime_with_missing_dataset_path_raises(tmp_path) -> None:
    # a real failure mode: the path IS routed to the loader, and a setup failure raises loudly.
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises((FileNotFoundError, OSError)):
        build_default_runtime(dataset_path=missing)


# --- 10.7 assistant module remains lightweight ------------------------------

def test_assistant_import_remains_lightweight() -> None:
    code = (
        "import sys; import src.assistant;"
        "forbidden = ['pandas', 'numpy', 'src.data_loader', 'src.data_model', 'src.data_validation',"
        " 'src.tools', 'src.assistant_runtime', 'src.llm_query_parser', 'src.web', 'src.api',"
        " 'src.database', 'src.rag', 'src.agent'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_runtime_module_may_import_bootstrap_dependencies() -> None:
    # the bootstrap layer is ALLOWED to pull data modules (the separation is the point).
    code = (
        "import sys; import src.assistant_runtime;"
        "allowed = ['src.data_loader', 'src.data_model', 'src.data_validation',"
        " 'src.validation_context', 'src.tool_registry'];"
        "missing = [m for m in allowed if m not in sys.modules];"
        "assert not missing, missing; print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_importing_runtime_does_not_load_data_eagerly() -> None:
    # importing the module must not build a runtime / read the CSV at import time.
    code = (
        "import sys; import src.assistant_runtime as rt;"
        "assert hasattr(rt, 'build_default_runtime');"
        "assert not any('nba_dataset' in str(getattr(m, '__file__', '')) for m in sys.modules.values());"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --- 10.8 runtime module scope ----------------------------------------------

def test_runtime_source_has_no_out_of_scope_logic() -> None:
    for forbidden in ("argparse", "input(", "flask", "fastapi", "uvicorn", "sqlite", "requests.",
                      ".mean(", ".sum(", "groupby", "llm", "openai"):
        assert forbidden not in RUNTIME_SRC, f"assistant_runtime.py must not contain {forbidden!r}"
