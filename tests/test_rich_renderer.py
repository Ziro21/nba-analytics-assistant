"""Optional Rich renderer (view layer) unit tests.

These run only where the optional ``rich`` dependency is installed (``pip install -r
requirements-rich.txt``); otherwise the whole module skips cleanly. The renderer is a pure view: it
reads an already-produced ``AssistantResult`` and must never compute, mutate, or import analytics.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rich")  # optional dependency; skip the whole module if Rich is absent

from rich.console import Console  # noqa: E402 - imported after the optional-dependency guard

from src.assistant_types import AssistantIssue, AssistantResult  # noqa: E402
from src.rich_renderer import render_result  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _capture(result: AssistantResult) -> str:
    """Render to a captured, non-terminal Console (fixed width) and return clean text.

    A non-terminal Console emits the layout (panels/tables/box characters) without ANSI styling, so
    assertions read the actual words deterministically. Production uses a real auto-detecting Console
    that adds colour.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=100)
    render_result(result, console=console)
    return buffer.getvalue()


def _simple_answer() -> AssistantResult:
    return AssistantResult.answer(
        "Golden State Warriors averaged 114.4 points over the last 5 games.",
        query="q", tool_name="team_average_points",
        data={"team": "Golden State Warriors", "average_points": 114.4, "games_used": 5},
    )


def _comparison_answer() -> AssistantResult:
    def profile(team, record, pf, pa, ortg, drtg, net):
        return {"team": team, "games": 10, "wins": 0, "losses": 0, "record": record,
                "win_pct": 0.5, "average_points_for": pf, "average_points_against": pa,
                "average_plus_minus": pf - pa, "average_ortg": ortg, "average_drtg": drtg,
                "average_net_rating": net, "date_start": None, "date_end": None}
    return AssistantResult.answer(
        "Boston Celtics had the stronger profile over this selected sample based on net rating.",
        query="q", tool_name="compare_team_profiles",
        data={
            "team_a": "Golden State Warriors", "team_b": "Boston Celtics",
            "window": 10, "location": None,
            "team_a_profile": profile("Golden State Warriors", "4-6", 110.9, 114.7, 113.0, 116.6, -3.6),
            "team_b_profile": profile("Boston Celtics", "7-3", 108.5, 100.7, 116.0, 107.8, 5.8),
            "comparison": {"stronger_profile_team": "Boston Celtics",
                           "profile_strength_summary": "Boston Celtics had the stronger profile "
                                                       "over this selected sample based on net rating."},
        },
    )


def _top_scoring_answer() -> AssistantResult:
    return AssistantResult.answer(
        "Top scoring teams: 1. Atlanta Hawks - 116.1 points per game.",
        query="q", tool_name="top_scoring_teams",
        data={"teams": [{"rank": 1, "team": "Atlanta Hawks", "average_points": 116.13, "games_used": 82},
                        {"rank": 2, "team": "Indiana Pacers", "average_points": 115.9, "games_used": 82}],
              "teams_returned": 2, "n_requested": 5},
    )


# --- panels -----------------------------------------------------------------

def test_simple_answer_renders_answer_panel() -> None:
    text = _capture(_simple_answer())
    assert "ANSWER" in text
    assert "Golden State Warriors averaged 114.4 points" in text


def test_clarification_renders_clarification_panel() -> None:
    issue = AssistantIssue("ambiguous_team", "amb", suggestions=("Los Angeles Lakers",))
    result = AssistantResult.clarification_needed(
        '"LA" is ambiguous. Do you mean Los Angeles Lakers or Los Angeles Clippers?',
        (issue,), query="q")
    text = _capture(result)
    assert "CLARIFICATION NEEDED" in text and "ambiguous" in text


def test_unsupported_renders_unsupported_panel() -> None:
    issue = AssistantIssue("unsupported_query", "no")
    result = AssistantResult.unsupported("This assistant supports dataset-based NBA analytics only.",
                                         (issue,), query="q")
    assert "UNSUPPORTED QUERY" in _capture(result)


def test_unsupported_betting_query_still_refuses_without_footer() -> None:
    issue = AssistantIssue("unsupported_query", "no betting advice")
    result = AssistantResult.unsupported(
        "I can only answer supported NBA analytics questions.",
        (issue,),
        query="Should I bet on Warriors or Celtics?",
    )
    text = _capture(result)
    assert "UNSUPPORTED QUERY" in text
    assert "supported NBA analytics" in text
    assert "Static dataset" not in text


def test_error_renders_error_panel() -> None:
    issue = AssistantIssue("internal_error", "boom")
    result = AssistantResult.error("Something went wrong.", (issue,), query="q")
    assert "ERROR" in _capture(result)


# --- tables -----------------------------------------------------------------

def test_comparison_renders_table_with_key_content() -> None:
    text = _capture(_comparison_answer())
    assert "Golden State Warriors" in text and "Boston Celtics" in text
    assert "PPG" in text and "Net" in text              # column headers
    assert "+5.8" in text and "-3.6" in text            # signed net rating, from data
    assert "4-6" in text and "7-3" in text              # records, from data
    assert "stronger profile" in text                   # summary verdict (not recomputed)


def test_top_scoring_renders_table_with_ranks_and_teams() -> None:
    text = _capture(_top_scoring_answer())
    assert "Atlanta Hawks" in text and "Indiana Pacers" in text
    assert "Rank" in text


# --- footer, fallback, safety -----------------------------------------------

def test_static_dataset_footer_is_not_rendered_in_pretty_output() -> None:
    text = _capture(_simple_answer())
    assert "Golden State Warriors averaged 114.4 points" in text
    assert "Static dataset" not in text
    assert "no live scores" not in text and "betting recommendations" not in text


def test_falls_back_to_message_panel_on_unexpected_data() -> None:
    # compare tool_name but missing the profile fields -> must not crash; renders a panel of message.
    result = AssistantResult.answer("Fallback message body.", query="q",
                                    tool_name="compare_team_profiles", data={"team_a": "X"})
    text = _capture(result)
    assert "Fallback message body." in text and "ANSWER" in text


def test_render_is_deterministic() -> None:
    result = _comparison_answer()
    assert _capture(result) == _capture(result)


def test_render_does_not_mutate_result() -> None:
    result = _comparison_answer()
    before = result.to_dict()
    _capture(result)
    assert result.to_dict() == before


def test_cli_pretty_end_to_end_renders_table(monkeypatch, capsys) -> None:
    # the full --pretty path: CLI -> real Rich renderer -> stdout (fake runtime; no dataset load).
    import src.assistant_runtime as runtime_module
    from src import cli

    result = _comparison_answer()
    monkeypatch.setattr(runtime_module, "build_default_runtime",
                        lambda: type("RT", (), {"answer": lambda self, q: result})())
    code = cli.main(["--pretty", "compare warriors and celtics"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Golden State Warriors" in out and "Boston Celtics" in out
    assert "+5.8" in out
    assert "Static dataset" not in out


def test_renderer_imports_no_analytics_modules() -> None:
    code = (
        "import sys, src.rich_renderer;"
        "bad=[m for m in ['pandas','numpy','src.tools','src.tool_registry','src.data_loader',"
        "'src.intent_validator','src.rule_parser','src.rag','src.agent','src.web','src.api',"
        "'src.database'] if m in sys.modules];"
        "assert not bad, bad; print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO_ROOT))
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout
