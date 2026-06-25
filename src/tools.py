"""Shared dataframe helpers for the analytical tools.

Phase 5A provides ONLY the reusable building blocks every tool will share — franchise
filtering, team filtering, windowing, and date-range extraction. The six analytical
tools themselves are implemented later (Phases 5B–5G) and are intentionally absent here.

Rules: pandas is the only source of truth; no helper prints, mutates its input, or
rounds. Exhibition (All-Star) rows are excluded by default for franchise-level use.
"""

from __future__ import annotations

import pandas as pd


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
