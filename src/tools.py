"""Shared dataframe helpers and the analytical tools.

Provides the reusable building blocks every tool shares — franchise filtering, team
filtering, windowing, and date-range extraction — plus the analytical tools as they are
implemented one per Phase 5 sub-step. All six are now implemented:
``team_average_points``, ``average_points_allowed``, ``team_record``,
``top_scoring_teams``, ``head_to_head``, ``team_efficiency_summary``.

Rules: pandas is the only source of truth; no helper or tool prints, mutates its input,
or rounds. Exhibition (All-Star) rows are excluded by default for franchise-level use.
"""

from __future__ import annotations

import pandas as pd

from src.config import DEFAULT_TOP_N
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


def _validate_window_value(window: int | None) -> None:
    """Raise ``ValueError`` unless ``window`` is ``None`` or a positive, non-bool int."""
    if window is None:
        return
    if isinstance(window, bool) or not isinstance(window, int):
        raise ValueError(f"window must be a positive integer or None, got {window!r}")
    if window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")


def apply_window(df: pd.DataFrame, window: int | None) -> tuple[pd.DataFrame, list[str]]:
    """Return the last ``window`` rows (in the frame's existing order) plus any warnings.

    - ``window is None`` → all rows, no warning.
    - positive int → last N rows.
    - ``window`` larger than available → all rows + a clear warning.
    - ``window <= 0``, non-int, or bool → ``ValueError``.

    Tools later catch ``ValueError`` and return ``status="error"``. The input is never
    mutated; a copy is returned.
    """
    _validate_window_value(window)
    if window is None:
        return df.copy(), []

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


def _require_positive_int(value: object, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a positive, non-bool integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def top_scoring_teams(
    clean_df: pd.DataFrame, n: int = DEFAULT_TOP_N, season_id: int | None = None
) -> ToolResult:
    """Rank franchises by mean ``points_for`` (exhibition rows excluded by default).

    Args:
        clean_df: The clean per-team-game view.
        n: Number of top teams to return.
        season_id: Optional opaque season filter (never decoded into a calendar season).

    Ranking: mean ``points_for`` descending, ``team_name`` ascending as a deterministic
    tie-break. Returns the §4.1 contract with a ranked ``teams`` list. Invalid ``n`` or a
    non-int ``season_id`` → ``status="error"``; a valid season with no rows → ``status="no_data"``;
    ``n`` larger than the available teams returns all with a warning. Means are unrounded.
    """
    tool = "top_scoring_teams"
    try:
        _require_positive_int(n, "n")
        if season_id is not None and (
            isinstance(season_id, bool) or not isinstance(season_id, int)
        ):
            raise ValueError(f"season_id must be an integer or None, got {season_id!r}")
    except ValueError as exc:
        return error_result(tool, str(exc), meta=build_meta())

    games = filter_franchise_games(clean_df)
    if season_id is not None:
        games = games[games["season_id"] == season_id]

    if games.empty:
        return no_data_result(
            tool,
            result={"teams": [], "teams_returned": 0, "n_requested": n},
            meta=build_meta(season_id=season_id, games_used=0),
            warnings=[f"No games found for season_id {season_id!r}."],
        )

    ranked = (
        games.groupby("team_name")["points_for"]
        .agg(avg="mean", games="count")
        .reset_index()
        .sort_values(by=["avg", "team_name"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    total_teams = len(ranked)
    warnings: list[str] = []
    if n > total_teams:
        warnings.append(
            f"Requested top {n} but only {total_teams} teams available; returning all {total_teams}."
        )

    teams = [
        {
            "rank": rank,
            "team": row["team_name"],
            "average_points": float(row["avg"]),
            "games_used": int(row["games"]),
        }
        for rank, (_, row) in enumerate(ranked.head(n).iterrows(), start=1)
    ]
    meta = build_meta(
        games_used=len(games),
        date_range=date_range_for(games),
        season_id=season_id,
    )
    return ok_result(
        tool,
        {"teams": teams, "teams_returned": len(teams), "n_requested": n},
        meta=meta,
        warnings=warnings,
    )


def head_to_head(
    clean_df: pd.DataFrame, team_a: str, team_b: str, window: int | None = None
) -> ToolResult:
    """Head-to-head record and scoring summary from ``team_a``'s perspective.

    Counts each meeting ONCE, from ``team_a``'s row (``team_name == team_a`` and
    ``opponent_team_name == team_b``), so meetings are not doubled. Returns the §4.1
    contract with both team names in ``result`` and ``team_a`` in ``meta["team"]``.

    Validation order: window first (invalid window → ``status="error"``), then
    ``team_a == team_b`` → ``error``. A pair with no meetings (incl. an unknown team)
    → ``no_data``. An over-long window uses all meetings with a warning. Means unrounded.
    """
    tool = "head_to_head"
    try:
        _validate_window_value(window)
    except ValueError as exc:
        return error_result(tool, str(exc), meta=build_meta(team=team_a))

    if team_a == team_b:
        return error_result(
            tool,
            f"team_a and team_b must differ; both were {team_a!r}.",
            meta=build_meta(team=team_a),
        )

    meetings = filter_team_games(clean_df, team_a)
    meetings = meetings[meetings["opponent_team_name"] == team_b]
    windowed, warnings = apply_window(meetings, window)  # window already validated

    if windowed.empty:
        return no_data_result(
            tool,
            result={
                "team_a": team_a, "team_b": team_b, "meetings": 0,
                "team_a_wins": 0, "team_b_wins": 0, "record": "0-0",
                "average_points_for": None, "average_points_against": None,
                "average_point_differential": None,
            },
            meta=build_meta(team=team_a, games_used=0, window_requested=window),
            warnings=[f"No head-to-head games found for {team_a!r} vs {team_b!r}."],
        )

    meetings_count = len(windowed)
    team_a_wins = int(windowed["win_flag"].sum())
    team_b_wins = meetings_count - team_a_wins
    avg_points_for = float(windowed["points_for"].mean())
    avg_points_against = float(windowed["points_against"].mean())
    result = {
        "team_a": team_a,
        "team_b": team_b,
        "meetings": meetings_count,
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "record": f"{team_a_wins}-{team_b_wins}",
        "average_points_for": avg_points_for,
        "average_points_against": avg_points_against,
        "average_point_differential": avg_points_for - avg_points_against,
    }
    meta = build_meta(
        team=team_a,
        games_used=meetings_count,
        date_range=date_range_for(windowed),
        window_requested=window,
    )
    return ok_result(tool, result, meta=meta, warnings=warnings)


def team_efficiency_summary(
    clean_df: pd.DataFrame, team: str, window: int | None = None
) -> ToolResult:
    """Descriptive recent-form efficiency summary for a team.

    Reports the **average per-game** offensive and defensive rating (mean of ``ortg`` /
    ``drtg``), net rating (mean of ``net_rating``), and possessions over the selected
    games. This is a per-game mean, NOT a possession-weighted season-level aggregate.
    The tool does not judge whether a team is "good" or "bad" — that is the formatter's job.

    Returns the §4.1 contract. Invalid window → ``status="error"`` (checked first); a team
    with no games → ``status="no_data"``; an over-long window uses all games with a warning.
    Values are returned unrounded. ``ortg``/``drtg``/``net_rating``/``possessions`` are core,
    null-free columns, so the means need no NaN handling.
    """
    tool = "team_efficiency_summary"
    team_games = filter_team_games(clean_df, team)
    try:
        windowed, warnings = apply_window(team_games, window)
    except ValueError as exc:
        return error_result(tool, str(exc), meta=build_meta(team=team))

    if windowed.empty:
        return no_data_result(
            tool,
            result={
                "team": team, "average_ortg": None, "average_drtg": None,
                "average_net_rating": None, "average_possessions": None, "games_used": 0,
            },
            meta=build_meta(team=team, games_used=0, window_requested=window),
            warnings=[f"No games found for team {team!r}."],
        )

    games_used = len(windowed)
    result = {
        "team": team,
        "average_ortg": float(windowed["ortg"].mean()),
        "average_drtg": float(windowed["drtg"].mean()),
        "average_net_rating": float(windowed["net_rating"].mean()),
        "average_possessions": float(windowed["possessions"].mean()),
        "games_used": games_used,
    }
    meta = build_meta(
        team=team,
        games_used=games_used,
        date_range=date_range_for(windowed),
        window_requested=window,
    )
    return ok_result(tool, result, meta=meta, warnings=warnings)
