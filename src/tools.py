"""Shared dataframe helpers and the analytical tools.

Provides the reusable building blocks every tool shares — franchise filtering, team
filtering, windowing, and date-range extraction — plus the analytical tools as they are
implemented one per Phase 5 sub-step. Implemented so far: ``team_average_points``,
``average_points_allowed``, ``team_record``. Pending: ``top_scoring_teams``,
``head_to_head``, ``team_efficiency_summary``.

Rules: pandas is the only source of truth; no helper or tool prints, mutates its input,
or rounds. Exhibition (All-Star) rows are excluded by default for franchise-level use.
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

def _team_average_metric(
    clean_df: pd.DataFrame,
    team: str,
    metric_column: str,
    result_key: str,
    tool_name: str,
    window: int | None = None,
) -> ToolResult:
    """Shared logic for a single-metric team average over recent games.

    Used by ``team_average_points`` (points_for) and ``average_points_allowed``
    (points_against). Status semantics: an invalid window (``<= 0``, non-int, bool)
    yields ``status="error"`` (checked first, so it wins even for an unknown team); a
    team with no matching games yields ``status="no_data"`` (distinguishing a genuinely
    unknown team is the validator's job upstream). An over-long window uses all games
    with a warning. pandas computes the mean — the value is returned unrounded.
    """
    team_games = filter_team_games(clean_df, team)
    try:
        windowed, warnings = apply_window(team_games, window)
    except ValueError as exc:
        return error_result(tool_name, str(exc), meta=build_meta(team=team))

    if windowed.empty:
        return no_data_result(
            tool_name,
            result={"team": team, result_key: None, "games_used": 0},
            meta=build_meta(team=team, games_used=0, window_requested=window),
            warnings=[f"No games found for team {team!r}."],
        )

    value = float(windowed[metric_column].mean())
    games_used = len(windowed)
    meta = build_meta(
        team=team,
        games_used=games_used,
        date_range=date_range_for(windowed),
        window_requested=window,
    )
    return ok_result(
        tool_name,
        {"team": team, result_key: value, "games_used": games_used},
        meta=meta,
        warnings=warnings,
    )


def team_average_points(
    clean_df: pd.DataFrame, team: str, window: int | None = None
) -> ToolResult:
    """Average points scored (``points_for``) by a team over its most recent games."""
    return _team_average_metric(
        clean_df, team, "points_for", "average_points", "team_average_points", window
    )


def average_points_allowed(
    clean_df: pd.DataFrame, team: str, window: int | None = None
) -> ToolResult:
    """Average points conceded (``points_against``) by a team over its most recent games."""
    return _team_average_metric(
        clean_df,
        team,
        "points_against",
        "average_points_allowed",
        "average_points_allowed",
        window,
    )


def team_record(
    clean_df: pd.DataFrame, team: str, window: int | None = None
) -> ToolResult:
    """Win/loss record for a team over its most recent games (from ``win_flag``).

    Args:
        clean_df: The clean per-team-game view.
        team: Exact canonical ``team_name``.
        window: Number of most recent games; ``None`` uses all games.

    Returns the §4.1 contract. ``wins`` counts ``win_flag``; ``losses`` is the remainder
    (NBA games have no draws); ``win_percentage`` is ``wins / games_used`` (unrounded).
    Invalid window → ``status="error"`` (checked first); a team with no games →
    ``status="no_data"``; an over-long window uses all games with a warning.
    """
    tool = "team_record"
    team_games = filter_team_games(clean_df, team)
    try:
        windowed, warnings = apply_window(team_games, window)
    except ValueError as exc:
        return error_result(tool, str(exc), meta=build_meta(team=team))

    if windowed.empty:
        return no_data_result(
            tool,
            result={"team": team, "wins": 0, "losses": 0, "record": "0-0",
                    "games_used": 0, "win_percentage": None},
            meta=build_meta(team=team, games_used=0, window_requested=window),
            warnings=[f"No games found for team {team!r}."],
        )

    games_used = len(windowed)
    wins = int(windowed["win_flag"].sum())
    losses = games_used - wins
    result = {
        "team": team,
        "wins": wins,
        "losses": losses,
        "record": f"{wins}-{losses}",
        "games_used": games_used,
        "win_percentage": wins / games_used,
    }
    meta = build_meta(
        team=team,
        games_used=games_used,
        date_range=date_range_for(windowed),
        window_requested=window,
    )
    return ok_result(tool, result, meta=meta, warnings=warnings)
