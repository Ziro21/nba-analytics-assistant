"""Shared dataframe helpers for the analytical tools.

Phase 5A provides ONLY the reusable building blocks every tool will share — franchise
filtering, team filtering, windowing, and date-range extraction. The six analytical
tools themselves are implemented later (Phases 5B–5G) and are intentionally absent here.

Rules: pandas is the only source of truth; no helper prints, mutates its input, or
rounds. Exhibition (All-Star) rows are excluded by default for franchise-level use.
"""

from __future__ import annotations

import pandas as pd

from src.tool_results import (
    ToolResult,
    build_meta,
    error_result,
    no_data_result,
    ok_result,
)


def filter_franchise_games(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Return franchise games only: exclude exhibition rows (and rows whose opponent is
    an exhibition team). Does not mutate the input or re-sort; returns a copy."""
    mask = ~clean_df["is_exhibition"]
    if "opponent_is_exhibition" in clean_df.columns:
        mask &= ~clean_df["opponent_is_exhibition"]
    return clean_df.loc[mask].copy()


def filter_team_games(clean_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return one franchise's games by EXACT canonical ``team_name`` match.

    Applies franchise filtering first. No alias handling, no fuzzy matching, no guessing
    (those belong to the parser/validator later). An unknown team yields an empty frame.
    Order is preserved; the input is not mutated.
    """
    franchise = filter_franchise_games(clean_df)
    return franchise.loc[franchise["team_name"] == team].copy()


def apply_window(df: pd.DataFrame, window: int | None) -> tuple[pd.DataFrame, list[str]]:
    """Return the last ``window`` rows (in the frame's existing order) plus any warnings.

    - ``window is None`` → all rows, no warning.
    - positive int → last N rows.
    - ``window`` larger than available → all rows + a clear warning.
    - ``window <= 0``, non-int, or bool → ``ValueError``.

    Tools later catch ``ValueError`` and return ``status="error"``. The input is never
    mutated; a copy is returned.
    """
    if window is None:
        return df.copy(), []
    if isinstance(window, bool) or not isinstance(window, int):
        raise ValueError(f"window must be a positive integer or None, got {window!r}")
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")

    available = len(df)
    if window > available:
        return df.copy(), [
            f"Requested last {window} games but only {available} available; using all {available}."
        ]
    return df.tail(window).copy(), []


def date_range_for(df: pd.DataFrame) -> list[str] | None:
    """Return ``[start_date, end_date]`` as ISO date strings, or ``None`` for an empty frame."""
    if df.empty:
        return None
    return [
        df["game_date"].min().date().isoformat(),
        df["game_date"].max().date().isoformat(),
    ]


# --- Analytical tools (Phase 5B+) -------------------------------------------

def team_average_points(
    clean_df: pd.DataFrame, team: str, window: int | None = None
) -> ToolResult:
    """Average points scored (``points_for``) by a team over its most recent games.

    Args:
        clean_df: The clean per-team-game view.
        team: Exact canonical ``team_name``.
        window: Number of most recent games to average; ``None`` uses all games.

    Status semantics: an invalid window (``<= 0``, non-int) yields ``status="error"``;
    a team with no matching games yields ``status="no_data"`` (the tool ran but found
    nothing — distinguishing a genuinely unknown team is the validator's job upstream).
    An over-long window uses all available games with a warning. pandas computes the
    mean — the value is returned unrounded.
    """
    tool = "team_average_points"
    team_games = filter_team_games(clean_df, team)
    # Validate the window first, so an invalid argument always errors (even for an
    # unknown team) before the no-data check.
    try:
        windowed, warnings = apply_window(team_games, window)
    except ValueError as exc:
        return error_result(tool, str(exc), meta=build_meta(team=team))

    if windowed.empty:
        return no_data_result(
            tool,
            meta=build_meta(team=team, window_requested=window),
            warnings=[f"No games found for team {team!r}."],
        )

    average_points = float(windowed["points_for"].mean())
    meta = build_meta(
        team=team,
        games_used=len(windowed),
        date_range=date_range_for(windowed),
        window_requested=window,
    )
    return ok_result(tool, {"average_points": average_points}, meta=meta, warnings=warnings)
