"""Clean internal data model: turn the validated raw frame into a tool-ready view.

`build_clean_view` converts the raw analytical dataframe (as returned by
`load_raw_dataset` and proven by Phase 3 validation) into a clean, sorted, per-team-game
view that the later analytical tools consume. It:

  - parses `match_date` into a datetime `game_date` (date parsing lives here, not the loader);
  - derives `opponent_team_name` via a one-to-one self-merge on `game_id`;
  - maps/derives the tool-ready columns (points_for, points_against, win_flag, net_rating, …);
  - flags exhibition rows (`is_exhibition`) without removing them;
  - sorts once by `team_name, game_date, game_id` for deterministic "last N games" windows.

It does NOT implement analytical tools, registries, parsers, an LLM, a formatter, or any
orchestration. `opponent_points` is NOT re-derived — Phase 3 already verified it.
"""

from __future__ import annotations

import pandas as pd

from src.config import SPECIAL_TEAMS
from src.data_validation import (
    EXPECTED_ROW_COUNT,
    EXPECTED_SPECIAL_TEAM_ROWS,
    DataValidationError,
    parse_match_date,
)

# Final clean schema (column order is part of the contract).
CLEAN_COLUMNS: tuple[str, ...] = (
    "game_id", "game_date", "season_id", "team_id", "team_name", "opponent_team_name",
    "is_home", "points_for", "points_against", "plus_minus", "win_flag",
    "possessions", "ortg", "drtg", "net_rating", "is_exhibition", "opponent_is_exhibition",
)

SORT_KEYS: tuple[str, ...] = ("team_name", "game_date", "game_id")


def _derive_opponent_team_name(raw_df: pd.DataFrame) -> pd.Series:
    """Derive each row's opponent team name via a one-to-one self-merge on ``game_id``.

    Pairs rows whose ``team_id`` differs. Raises if the pairing is not one-to-one or if
    any opponent is missing. Does not pre-filter exhibition rows.
    """
    pairs = raw_df[["game_id", "team_id", "team_name"]]
    merged = pairs.merge(pairs, on="game_id", suffixes=("", "_opp"))
    merged = merged[merged["team_id"] != merged["team_id_opp"]]
    if len(merged) != len(raw_df):
        raise DataValidationError(
            f"Opponent derivation is not one-to-one: {len(merged)} paired rows "
            f"vs {len(raw_df)} rows (broken game-pair structure?)."
        )
    opponent_by_key = merged.set_index(["game_id", "team_id"])["team_name_opp"]
    keys = pd.MultiIndex.from_arrays([raw_df["game_id"], raw_df["team_id"]])
    opponent = pd.Series(
        opponent_by_key.reindex(keys).to_numpy(),
        index=raw_df.index,
        name="opponent_team_name",
    )
    if opponent.isna().any():
        raise DataValidationError("opponent_team_name contains nulls after derivation.")
    return opponent


def build_clean_view(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the clean, sorted, tool-ready per-team-game view.

    Assumes the caller has run Phase 3 validation (`validate_dataset`) on ``raw_df``.
    The input frame is never mutated; a new dataframe is returned.

    Note: this does not call `validate_clean_view` itself — building does not imply
    post-transform validation. It does self-guard the opponent pairing (raising on a
    broken game-pair structure). Run `validate_clean_view` separately for full checks.
    """
    game_date = parse_match_date(raw_df)  # explicit, fixed-format parse (Phase 3 helper)
    opponent_team_name = _derive_opponent_team_name(raw_df)

    clean = pd.DataFrame(index=raw_df.index)
    clean["game_id"] = raw_df["game_id"]
    clean["game_date"] = game_date
    clean["season_id"] = raw_df["season_id"]
    clean["team_id"] = raw_df["team_id"]
    clean["team_name"] = raw_df["team_name"]
    clean["opponent_team_name"] = opponent_team_name
    clean["is_home"] = raw_df["is_home"]
    clean["points_for"] = raw_df["team_points"]
    clean["points_against"] = raw_df["opponent_points"]
    clean["plus_minus"] = raw_df["plus_minus"]
    clean["win_flag"] = clean["points_for"] > clean["points_against"]
    clean["possessions"] = raw_df["possessions"]
    clean["ortg"] = raw_df["ORTG"]
    clean["drtg"] = raw_df["DRTG"]
    clean["net_rating"] = clean["ortg"] - clean["drtg"]
    clean["is_exhibition"] = clean["team_name"].isin(SPECIAL_TEAMS)
    clean["opponent_is_exhibition"] = clean["opponent_team_name"].isin(SPECIAL_TEAMS)

    clean = clean[list(CLEAN_COLUMNS)]
    clean = clean.sort_values(list(SORT_KEYS), kind="mergesort").reset_index(drop=True)
    return clean


def validate_clean_view(
    clean_df: pd.DataFrame, raw_df: pd.DataFrame | None = None
) -> dict[str, object]:
    """Post-transform checks on the clean view. Raises ``DataValidationError`` on failure.

    If ``raw_df`` is supplied, also cross-checks the points mappings against the raw frame.
    """
    if tuple(clean_df.columns) != CLEAN_COLUMNS:
        raise DataValidationError(
            "Clean schema/order does not match the expected contract.\n"
            f"  expected: {list(CLEAN_COLUMNS)}\n"
            f"  found:    {list(clean_df.columns)}"
        )
    if len(clean_df) != EXPECTED_ROW_COUNT:
        raise DataValidationError(
            f"Clean view row count {len(clean_df)} != expected {EXPECTED_ROW_COUNT}."
        )
    if not pd.api.types.is_datetime64_any_dtype(clean_df["game_date"]):
        raise DataValidationError("game_date is not datetime-typed.")
    null_cols = [c for c in CLEAN_COLUMNS if clean_df[c].isna().any()]
    if null_cols:
        raise DataValidationError(f"Clean view has nulls in: {null_cols}")
    if (clean_df["opponent_team_name"] == clean_df["team_name"]).any():
        raise DataValidationError("opponent_team_name equals team_name on some rows.")
    if not (clean_df["win_flag"] == (clean_df["points_for"] > clean_df["points_against"])).all():
        raise DataValidationError("win_flag inconsistent with points_for > points_against.")
    if not (clean_df["net_rating"] == clean_df["ortg"] - clean_df["drtg"]).all():
        raise DataValidationError("net_rating inconsistent with ortg - drtg.")
    if not (clean_df["is_exhibition"] == clean_df["team_name"].isin(SPECIAL_TEAMS)).all():
        raise DataValidationError("is_exhibition does not flag exactly the special-team rows.")
    if not (
        clean_df["opponent_is_exhibition"]
        == clean_df["opponent_team_name"].isin(SPECIAL_TEAMS)
    ).all():
        raise DataValidationError(
            "opponent_is_exhibition does not flag exactly the special-team opponents."
        )
    exhibition_rows = int(clean_df["is_exhibition"].sum())
    if exhibition_rows != EXPECTED_SPECIAL_TEAM_ROWS:
        raise DataValidationError(
            f"Expected {EXPECTED_SPECIAL_TEAM_ROWS} exhibition rows, found {exhibition_rows}."
        )

    keys = clean_df[list(SORT_KEYS)].reset_index(drop=True)
    expected_order = keys.sort_values(list(SORT_KEYS), kind="mergesort").reset_index(drop=True)
    if not keys.equals(expected_order):
        raise DataValidationError(f"Clean view is not sorted by {list(SORT_KEYS)}.")

    if raw_df is not None:
        check = clean_df.merge(
            raw_df[["game_id", "team_id", "team_points", "opponent_points"]],
            on=["game_id", "team_id"],
            how="left",
            validate="one_to_one",
        )
        if not (check["points_for"] == check["team_points"]).all():
            raise DataValidationError("points_for does not map from team_points.")
        if not (check["points_against"] == check["opponent_points"]).all():
            raise DataValidationError("points_against does not map from opponent_points.")

    return {
        "rows": len(clean_df),
        "columns": list(clean_df.columns),
        "exhibition_rows": exhibition_rows,
        "date_min": clean_df["game_date"].min().date().isoformat(),
        "date_max": clean_df["game_date"].max().date().isoformat(),
        "sorted_by": list(SORT_KEYS),
    }
