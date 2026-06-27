"""Phase 10D: final delivery acceptance gate.

A consolidated, practical proof that the project is ready to ship: the public surface imports,
the real CLI runs end to end, the runtime answers and fails safely, the docs are present and
honest, packaging is minimal, no out-of-scope module exists, and the import boundaries hold.
It is a delivery gate, not a re-run of every unit test.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

README = REPO_ROOT / "README.md"
DOC_FILES = (
    README,
    REPO_ROOT / "docs" / "usage_examples.md",
    REPO_ROOT / "docs" / "architecture.md",
    REPO_ROOT / "docs" / "testing_and_quality.md",
)

# AI-vendor/authorship markers stored reversed so this file holds no literal AI token.
_REVERSED_AI_MARKERS = (
    "edualc", "ciporhtna", "tpgtahc", "tolipoc", "xedoc", "ianepo", "inimeg",
    "derohtua-oc", "htiw detareneg", "yb-detareneg", "detareneg ia",
)
AI_MARKERS = tuple(m[::-1] for m in _REVERSED_AI_MARKERS)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-m", "src.cli", *args], capture_output=True, text=True,
                          cwd=str(REPO_ROOT))


def _import_probe(module: str, forbidden: list[str]) -> subprocess.CompletedProcess:
    code = (
        f"import sys; import {module};"
        f"forbidden = {forbidden!r};"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))


# --- 1. public delivery surface ---------------------------------------------

def test_delivery_public_surface_is_available() -> None:
    from src.assistant import answer_query
    from src.assistant_runtime import AssistantRuntime, build_default_runtime
    from src.assistant_types import AssistantIssue, AssistantResult
    from src.cli import main
    from src.response_formatter import (
        format_parse_failure, format_tool_result, format_validation_failure,
    )
    assert callable(main) and callable(build_default_runtime) and callable(answer_query)
    assert callable(format_tool_result) and callable(format_parse_failure)
    assert callable(format_validation_failure)
    assert AssistantRuntime.__name__ == "AssistantRuntime"
    assert AssistantResult.answer("ok").status == "answer"
    assert AssistantIssue("c", "m").code == "c"


# --- 2. CLI delivery readiness (real subprocess) ----------------------------

def test_delivery_cli_import_is_lightweight() -> None:
    result = _import_probe("src.cli", [
        "pandas", "numpy", "src.tools", "src.llm_query_parser", "src.web", "src.api",
        "src.database", "src.rag", "src.agent",
    ])
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_delivery_cli_human_readable_smoke() -> None:
    result = _run_cli("How many points do the Warriors average over the last 5 games?")
    assert result.returncode == 0, result.stderr
    assert "Golden State Warriors" in result.stdout
    assert "114.4" in result.stdout
    assert "Traceback" not in result.stderr


def test_delivery_cli_json_smoke() -> None:
    result = _run_cli("--json", "Celtics vs Heat head to head")
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "answer"
    assert payload["tool_name"] == "head_to_head"
    assert "Boston Celtics" in payload["message"] and "Miami Heat" in payload["message"]


# --- 3. runtime delivery readiness ------------------------------------------

@pytest.fixture(scope="module")
def runtime():
    from src.assistant_runtime import build_default_runtime
    return build_default_runtime()


def test_delivery_runtime_supported_query_smoke(runtime) -> None:
    result = runtime.answer("How many points do the Warriors average over the last 5 games?")
    assert result.status == "answer" and result.tool_name == "team_average_points"
    assert "Golden State Warriors" in result.message and "114.4" in result.message
    json.dumps(result.to_dict())


@pytest.mark.parametrize("query,status,code", [
    ("How many points do LA average?", "clarification_needed", "ambiguous_team"),
    ("How many points do Celics average?", "clarification_needed", "unknown_team"),
    ("Celtics vs Celtics head to head", "clarification_needed", "same_team_head_to_head"),
    ("Who is better?", "unsupported", None),
])
def test_delivery_runtime_safe_failure_smoke(runtime, query, status, code) -> None:
    result = runtime.answer(query)
    assert result.status == status
    if code is not None:
        assert code in [e.code for e in result.errors]
    json.dumps(result.to_dict())


# --- 4. documentation delivery readiness ------------------------------------

def test_delivery_required_documentation_exists() -> None:
    for path in DOC_FILES:
        assert path.exists(), f"missing {path.relative_to(REPO_ROOT)}"


def test_delivery_readme_contains_real_usage_commands() -> None:
    text = README.read_text().lower()
    for fragment in ("python -m src.cli", "--json", "python -m pytest tests/ -q",
                     "average points", "points allowed", "record",
                     "top scoring teams", "head-to-head", "efficiency"):
        assert fragment in text, f"README should mention {fragment!r}"


def test_delivery_documentation_declares_key_limitations() -> None:
    text = "\n".join(p.read_text().lower() for p in DOC_FILES if p.exists())
    for limitation in ("no live data", "no betting", "no llm", "no web", "arbitrary"):
        assert limitation in text, f"docs should state the limitation {limitation!r}"


# --- 5. required test layers exist ------------------------------------------

def test_delivery_key_test_layers_exist() -> None:
    for name in ("test_tools.py", "test_tool_registry.py", "test_intent_validator.py",
                 "test_rule_parser_phase8_final.py", "test_assistant_phase9_final.py",
                 "test_assistant_runtime.py", "test_cli.py", "test_documentation.py",
                 "test_delivery_final.py"):
        assert (REPO_ROOT / "tests" / name).exists(), f"missing tests/{name}"


# --- 6. packaging / configuration -------------------------------------------

def _packages(path: Path) -> set:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return {ln.split("==")[0].split(">=")[0].strip().lower()
            for ln in lines if ln and not ln.startswith("#")}


def test_delivery_requirements_are_minimal() -> None:
    # the exact-set check guarantees no extra (LLM/web/API/rich/...) dependency was added to the CORE.
    assert _packages(REPO_ROOT / "requirements.txt") == {"pandas", "pytest"}


def test_rich_is_an_optional_dependency_only() -> None:
    # Rich (pretty terminal mode) must never become a core dependency — it lives in its own optional
    # file, so `pip install -r requirements.txt` keeps the assistant minimal and deterministic.
    assert "rich" not in _packages(REPO_ROOT / "requirements.txt")
    rich_file = REPO_ROOT / "requirements-rich.txt"
    assert rich_file.exists(), "requirements-rich.txt (optional pretty-mode dependency) is missing"
    assert "rich" in _packages(rich_file)


def test_delivery_packaging_entry_point_is_consistent_if_present() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return  # acceptable: `python -m src.cli` is the supported command
    text = pyproject.read_text()
    if "[project.scripts]" in text:
        assert "src.cli:main" in text, "a script entry point must target src.cli:main"


# --- 7. scope guard: no out-of-scope production modules ---------------------

def test_delivery_no_out_of_scope_production_modules_exist() -> None:
    for module in ("src.response_formatter_llm", "src.web", "src.api",
                   "src.database", "src.rag", "src.agent", "src.server", "src.live_data",
                   "src.betting_model", "src.odds_model", "src.parse_validate_execute"):
        assert importlib.util.find_spec(module) is None, f"{module} must not exist"


# --- 8. import / scope safety -----------------------------------------------

def test_delivery_assistant_import_scope_is_safe() -> None:
    result = _import_probe("src.assistant", [
        "pandas", "numpy", "src.data_loader", "src.data_model", "src.data_validation",
        "src.tools", "src.llm_query_parser", "src.web", "src.api", "src.database",
        "src.rag", "src.agent",
    ])
    assert result.returncode == 0, result.stderr


def test_delivery_cli_import_scope_is_safe() -> None:
    result = _import_probe("src.cli", [
        "pandas", "numpy", "src.tools", "src.rule_parser", "src.intent_validator",
        "src.response_formatter", "src.llm_query_parser", "src.web", "src.api",
        "src.database", "src.rag", "src.agent",
    ])
    assert result.returncode == 0, result.stderr


# --- 9. AI-authorship / provenance sweep ------------------------------------

def test_delivery_public_files_have_no_ai_authorship_provenance_language() -> None:
    targets = (
        sorted((REPO_ROOT / "src").glob("*.py"))
        + sorted((REPO_ROOT / "tests").glob("*.py"))
        + sorted((REPO_ROOT / "docs").glob("*.md"))
        + [README, REPO_ROOT / "main.py", REPO_ROOT / "requirements.txt"]
    )
    blob = "\n".join(p.read_text() for p in targets if p.exists()).lower()
    found = [marker for marker in AI_MARKERS if marker in blob]
    assert not found, f"public files contain AI-authorship/provenance language: {found}"


# --- 10. full-suite command is documented -----------------------------------

def test_delivery_full_suite_command_is_documented() -> None:
    blob = "\n".join(p.read_text() for p in DOC_FILES if p.exists())
    assert "python -m pytest tests/ -q" in blob


# --- Pre-11B UX patch: clarification messages name the options ---------------

def test_delivery_ambiguous_team_message_names_both_options(runtime) -> None:
    ny = runtime.answer("New York record")
    assert ny.status == "clarification_needed"
    assert "New York Knicks" in ny.message and "Brooklyn Nets" in ny.message
    la = runtime.answer("How many points do LA average?")
    assert "Los Angeles Lakers" in la.message and "Los Angeles Clippers" in la.message


def test_delivery_cli_ambiguous_prints_both_team_options() -> None:
    result = _run_cli("New York record")
    assert result.returncode == 1
    assert "New York Knicks" in result.stdout and "Brooklyn Nets" in result.stdout
    assert "Traceback" not in result.stderr
