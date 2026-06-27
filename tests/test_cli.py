"""Phase 10B tests: command-line demo interface (fake runtime; no dataset load)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import src.assistant_runtime as runtime_module
import src.cli as cli
from src import __version__
from src.assistant_types import AssistantIssue, AssistantResult
from src.cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeRuntime:
    """Stand-in for AssistantRuntime that builds a query-preserving result of a chosen status."""

    def __init__(self, *, status="answer", message="ok", tool_name=None, code=None):
        self._status, self._message = status, message
        self._tool, self._code = tool_name, code
        self.queries: list[str] = []

    def answer(self, query):
        self.queries.append(query)
        if self._status == "answer":
            return AssistantResult.answer(self._message, query=query, tool_name=self._tool,
                                          data={"value": 1}, meta={"team": "Golden State Warriors"})
        issue = AssistantIssue(self._code or "internal_error", self._message)
        if self._status == "clarification_needed":
            return AssistantResult.clarification_needed(self._message, (issue,), query=query,
                                                        tool_name=self._tool)
        if self._status == "unsupported":
            return AssistantResult.unsupported(self._message, (issue,), query=query)
        return AssistantResult.error(self._message, (issue,), query=query)


def _install(monkeypatch, fake):
    monkeypatch.setattr(runtime_module, "build_default_runtime", lambda: fake)
    return fake


# --- 11.1 parser / help / missing query -------------------------------------

def test_help_returns_zero(capsys) -> None:
    assert main(["--help"]) == 0
    assert "usage" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("argv", [[], ["   "], ["", " "]])
def test_missing_query_returns_error(argv, capsys) -> None:
    assert main(argv) == 2
    assert capsys.readouterr().err.strip()  # safe message on stderr


def test_unknown_flag_returns_argparse_error(capsys) -> None:
    # the last branch of the exit-code contract: argparse errors -> exit 2, no traceback.
    code = main(["--definitely-not-a-flag", "some query"])
    err = capsys.readouterr().err
    assert code == 2
    assert err.strip() and "Traceback" not in err  # argparse usage, not a crash


# --- 11.1b --version --------------------------------------------------------

def test_cli_version_exits_zero_and_prints_version(capsys) -> None:
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "sporting-risk-nba-assistant" in out
    assert __version__ in out  # e.g. 1.2.0


def test_cli_version_does_not_build_runtime(monkeypatch, capsys) -> None:
    built: list[int] = []

    def _builder():
        built.append(1)
        return FakeRuntime()

    monkeypatch.setattr(runtime_module, "build_default_runtime", _builder)
    assert main(["--version"]) == 0
    assert built == []  # --version must not bootstrap or load the dataset


def test_cli_version_flag_does_not_break_normal_query(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeRuntime(status="answer", message="ok"))
    assert main(["Warriors record"]) == 0  # normal query path still works with the flag present
    assert fake.queries == ["Warriors record"]


# --- 11.2 successful human-readable output ----------------------------------

def test_answer_human_readable(monkeypatch, capsys) -> None:
    _install(monkeypatch, FakeRuntime(
        status="answer", tool_name="team_average_points",
        message="Golden State Warriors averaged 114.4 points over the last 5 games."))
    code = main(["How many points do the Warriors average over the last 5 games?"])
    out = capsys.readouterr()
    assert code == 0
    assert "Golden State Warriors averaged 114.4 points" in out.out
    assert out.err == ""


# --- 11.3 successful JSON output --------------------------------------------

def test_answer_json_output(monkeypatch, capsys) -> None:
    _install(monkeypatch, FakeRuntime(status="answer", tool_name="head_to_head", message="ok"))
    code = main(["--json", "Celtics vs Heat head to head"])
    out = capsys.readouterr()
    assert code == 0 and out.err == ""
    payload = json.loads(out.out)
    assert payload["status"] == "answer"
    assert payload["tool_name"] == "head_to_head"
    assert payload["query"] == "Celtics vs Heat head to head"


# --- 11.4 / 11.5 / 11.6 status -> exit code ---------------------------------

@pytest.mark.parametrize("status,code,exit_code", [
    ("clarification_needed", "ambiguous_team", 1),
    ("unsupported", "unsupported_query", 1),
    ("error", "internal_error", 2),
])
def test_non_answer_statuses(status, code, exit_code, monkeypatch, capsys) -> None:
    _install(monkeypatch, FakeRuntime(status=status, code=code, message="safe message"))
    rc = main(["some query"])
    out = capsys.readouterr()
    assert rc == exit_code
    assert "safe message" in out.out         # safe user-facing message, not a traceback
    assert "Traceback" not in out.out and "Traceback" not in out.err

    _install(monkeypatch, FakeRuntime(status=status, code=code, message="safe message"))
    assert main(["--json", "some query"]) == exit_code
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == status
    assert any(e["code"] == code for e in payload["errors"])


# --- 11.7 runtime bootstrap failure -----------------------------------------

def test_bootstrap_failure_returns_error(monkeypatch, capsys) -> None:
    def _raise():
        raise RuntimeError("dataset missing")

    monkeypatch.setattr(runtime_module, "build_default_runtime", _raise)
    code = main(["What is the Warriors record?"])
    out = capsys.readouterr()
    assert code == 2
    assert out.out == ""                     # nothing on stdout
    assert out.err.strip() and "Traceback" not in out.err  # safe stderr message, no traceback


# --- 11.8 query joining -----------------------------------------------------

def test_query_words_are_joined(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeRuntime(status="answer", message="ok"))
    main(["How", "many", "points?"])
    assert fake.queries == ["How many points?"]


def test_quoted_and_unquoted_are_equivalent(monkeypatch) -> None:
    fake = _install(monkeypatch, FakeRuntime(status="answer", message="ok"))
    main(["Warriors", "record"])
    main(["Warriors record"])
    assert fake.queries == ["Warriors record", "Warriors record"]


# --- 11.9 CLI uses runtime only ---------------------------------------------

def test_cli_uses_runtime_only(monkeypatch) -> None:
    built, fake = [], FakeRuntime(status="answer", message="ok")

    def _builder():
        built.append(1)
        return fake

    monkeypatch.setattr(runtime_module, "build_default_runtime", _builder)
    main(["Warriors record"])
    assert built == [1] and fake.queries == ["Warriors record"]
    # the CLI module never imports the lower layers directly
    for name in ("parse_rule_query", "validate_intent", "answer_query",
                 "format_tool_result", "registry", "DEFAULT_REGISTRY"):
        assert not hasattr(cli, name)


# --- 11.10 import / scope safety --------------------------------------------

def test_cli_import_is_lightweight() -> None:
    code = (
        "import sys; import src.cli;"
        "forbidden = ['pandas', 'numpy', 'src.data_loader', 'src.data_model', 'src.data_validation',"
        " 'src.tools', 'src.rule_parser', 'src.intent_validator', 'src.response_formatter',"
        " 'src.tool_registry', 'src.assistant_runtime', 'src.assistant', 'src.llm_query_parser',"
        " 'src.web', 'src.api', 'src.database', 'src.rag', 'src.agent'];"
        "bad = [m for m in forbidden if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
