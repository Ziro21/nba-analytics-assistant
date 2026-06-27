"""Optional Rich terminal presentation for an ``AssistantResult`` (a VIEW layer only).

Renders an already-produced ``AssistantResult`` into a professional terminal layout using the
optional ``rich`` library. It is strictly a view: it never parses, validates, resolves teams,
executes a tool, calls the registry, touches pandas, computes a statistic, or mutates the result —
it only reads the result's fields and the structured values the tools already produced.

Requires the optional ``rich`` dependency (``requirements-rich.txt``). The plain CLI and ``--json``
modes do not need it; ``--pretty`` imports this module lazily and fails closed if rich is absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src import __version__
from src.assistant_types import (
    ASSISTANT_STATUS_ANSWER,
    ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    ASSISTANT_STATUS_ERROR,
    ASSISTANT_STATUS_UNSUPPORTED,
    AssistantResult,
)

# One central, restrained style map — no scattered inline colour tags, no emoji, no backgrounds.
STATUS_STYLES = {
    "answer": "bold green",
    "clarification": "bold yellow",
    "unsupported": "bold magenta",
    "error": "bold red",
    "warning": "bold yellow",
    "muted": "dim",
    "positive": "green",
    "negative": "red",
}

# Status words are always shown as text (panel titles), so the output is readable without colour.
_PANELS = {
    ASSISTANT_STATUS_ANSWER: ("ANSWER", "answer"),
    ASSISTANT_STATUS_CLARIFICATION_NEEDED: ("CLARIFICATION NEEDED", "clarification"),
    ASSISTANT_STATUS_UNSUPPORTED: ("UNSUPPORTED QUERY", "unsupported"),
    ASSISTANT_STATUS_ERROR: ("ERROR", "error"),
}

COMPARE_TOOL = "compare_team_profiles"
TOP_SCORING_TOOL = "top_scoring_teams"

DATASET_FOOTER = (
    "Static dataset mode — no live scores, odds, injuries, or betting recommendations."
)


def _num(value: object, decimals: int = 1) -> str:
    """Display a numeric cell (rounded for readability) — pure formatting, never a calculation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def _signed(value: object, decimals: int = 1) -> str:
    """A signed numeric display (e.g. ``+5.8`` / ``-2.1``) so positives/negatives read without colour."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{'+' if value >= 0 else ''}{_num(value, decimals)}"
    return str(value)


def _panel(message: str, *, title: str, style: str) -> Panel:
    return Panel(Text(message), title=title, title_align="left", border_style=style, box=box.ROUNDED)


def _status_panel(result: AssistantResult) -> Panel:
    title, style_key = _PANELS.get(result.status, ("RESULT", "muted"))
    return _panel(result.message, title=title, style=STATUS_STYLES[style_key])


def _comparison_table(data: dict) -> Table:
    """Two-team table built from the already-computed profiles in ``data`` (no recomputation)."""
    profile_a = data["team_a_profile"]
    profile_b = data["team_b_profile"]
    table = Table(box=box.SIMPLE_HEAVY, title="Team comparison", title_style="bold",
                  title_justify="left")
    table.add_column("Team", justify="left", no_wrap=True)
    # Compact, conventional column headers (PPG = points scored, OPP = points allowed, Net = net
    # rating) so the eight columns fit narrow ~80-column terminals without truncation.
    for header in ("Games", "W-L", "PPG", "OPP", "ORTG", "DRTG", "Net"):
        table.add_column(header, justify="right")
    for profile in (profile_a, profile_b):
        net = profile["average_net_rating"]
        net_style = STATUS_STYLES["positive"] if isinstance(net, (int, float)) and net >= 0 \
            else STATUS_STYLES["negative"]
        table.add_row(
            str(profile["team"]),
            _num(profile["games"]),
            str(profile["record"]),
            _num(profile["average_points_for"]),
            _num(profile["average_points_against"]),
            _num(profile["average_ortg"]),
            _num(profile["average_drtg"]),
            Text(_signed(net), style=net_style),
        )
    return table


def _top_scoring_table(data: dict) -> Table:
    """Ranking table built from the already-ranked ``teams`` list in ``data`` (no recomputation)."""
    table = Table(box=box.SIMPLE_HEAVY, title="Top scoring teams", title_style="bold",
                  title_justify="left")
    table.add_column("Rank", justify="right")
    table.add_column("Team", justify="left", no_wrap=True)
    table.add_column("Points Per Game", justify="right")
    for row in data["teams"]:
        table.add_row(str(row["rank"]), str(row["team"]), _num(row["average_points"]))
    return table


def _footer() -> Text:
    return Text(f"{DATASET_FOOTER}  (assistant v{__version__})", style=STATUS_STYLES["muted"])


def render_result(result: AssistantResult, *, console: Optional[Console] = None) -> None:
    """Render an ``AssistantResult`` to the terminal. View only — never mutates ``result``.

    Dispatch is status-first (so clarification/unsupported/error never tabularise), then tool-name for
    answers. Any unexpected ``data`` shape falls back to a plain status panel — a rendering problem
    must never turn a valid result into a traceback.
    """
    console = console or Console()
    # AssistantResult.data is a frozen, JSON-safe structure (mappings → mappingproxy, lists → tuples),
    # so check against the abstract Mapping/Sequence types, not the concrete dict/list.
    data = result.data if isinstance(result.data, Mapping) else {}
    teams = data.get("teams")
    try:
        if result.status == ASSISTANT_STATUS_ANSWER and result.tool_name == COMPARE_TOOL \
                and "team_a_profile" in data and "team_b_profile" in data:
            console.print(_comparison_table(data))
            summary = (data.get("comparison") or {}).get("profile_strength_summary") or result.message
            console.print(_panel(summary, title="SUMMARY", style=STATUS_STYLES["answer"]))
        elif result.status == ASSISTANT_STATUS_ANSWER and result.tool_name == TOP_SCORING_TOOL \
                and isinstance(teams, Sequence) and not isinstance(teams, (str, bytes)) and teams:
            console.print(_top_scoring_table(data))
        else:
            console.print(_status_panel(result))
    except Exception:  # noqa: BLE001 - a render failure must never break a valid result
        console.print(_status_panel(result))

    for warning in result.warnings:
        console.print(Text(f"note: {warning.message}", style=STATUS_STYLES["muted"]))
    console.print(_footer())
