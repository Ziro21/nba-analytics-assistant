"""Phase 4A tests: the clean internal data model.

Happy paths run on the real file (build + validate + the GSW last-5 oracle). Targeted
behaviour and failure cases use a tiny hand-built raw frame. No network, no LLM, no
guessed expected values (114.4 is the verified §3 oracle).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import load_raw_dataset
from src.data_model import (
    CLEAN_COLUMNS,
    SORT_KEYS,
    build_clean_view,
    validate_clean_view,
)
from src.data_validation import DataValidationError


def make_raw_frame() -> pd.DataFrame:
    """A tiny valid raw frame (2 games, 4 rows) with the columns the clean model reads."""
    return pd.DataFrame(
        {
            "game_id": [100, 100, 101, 101],
            "match_date": [
                "2020-12-23 03:00:00",
                "2021-01-05 03:00:00",
                "2026-03-05 03:00:00",
                "2025-02-10 03:00:00",
            ],
            "season_id": [26, 26, 26, 26],
            "team_id": [1, 2, 1, 3],
            "team_name": ["Atlanta Hawks", "Boston Celtics", "Atlanta Hawks", "Chicago Bulls"],
            "is_home": [1, 0, 0, 1],
            "team_points": [110, 100, 120, 115],
            "opponent_points": [100, 110, 115, 120],
            "plus_minus": [10, -10, 5, -5],
            "possessions": [100, 100, 102, 102],
            "ORTG": [110, 100, 118, 113],
            "DRTG": [100, 110, 113, 118],
        }
    )


# --- Build: shape, schema, immutability -------------------------------------

def test_build_clean_view_row_count_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    assert len(clean) == 14_746


def test_clean_required_columns_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    assert tuple(clean.columns) == CLEAN_COLUMNS


def test_build_does_not_mutate_raw() -> None:
    raw = load_raw_dataset()
    before = raw.copy(deep=True)
    build_clean_view(raw)
    assert "game_date" not in raw.columns
    assert raw.equals(before)


# --- Dates ------------------------------------------------------------------

def test_game_date_parsed_with_expected_range_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    assert pd.api.types.is_datetime64_any_dtype(clean["game_date"])
    # game_date retains the parsed time component; compare on the date part.
    assert clean["game_date"].min().date().isoformat() == "2020-12-23"
    assert clean["game_date"].max().date().isoformat() == "2026-03-05"


# --- Opponent derivation ----------------------------------------------------

def test_opponent_derived_for_every_row_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    assert clean["opponent_team_name"].notna().all()


def test_opponent_never_equals_team_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    assert (clean["opponent_team_name"] != clean["team_name"]).all()


def test_opponent_symmetric_within_game() -> None:
    clean = build_clean_view(make_raw_frame())
    for _, pair in clean.groupby("game_id"):
        assert len(pair) == 2
        a, b = pair.iloc[0], pair.iloc[1]
        assert a["opponent_team_name"] == b["team_name"]
        assert b["opponent_team_name"] == a["team_name"]


def test_broken_opponent_pair_fails() -> None:
    broken = make_raw_frame()
    broken.loc[1, "team_id"] = 1  # game 100 now has two identical team_ids
    with pytest.raises(DataValidationError):
        build_clean_view(broken)


# --- Derived columns --------------------------------------------------------

def test_points_mappings() -> None:
    raw = make_raw_frame()
    clean = build_clean_view(raw)
    check = clean.merge(
        raw[["game_id", "team_id", "team_points", "opponent_points"]],
        on=["game_id", "team_id"], validate="one_to_one",
    )
    assert (check["points_for"] == check["team_points"]).all()
    assert (check["points_against"] == check["opponent_points"]).all()


def test_win_flag_correct() -> None:
    clean = build_clean_view(make_raw_frame())
    assert (clean["win_flag"] == (clean["points_for"] > clean["points_against"])).all()


def test_net_rating_correct() -> None:
    clean = build_clean_view(make_raw_frame())
    assert (clean["net_rating"] == clean["ortg"] - clean["drtg"]).all()


# --- Exhibition flags -------------------------------------------------------

def test_is_exhibition_flags_eight_rows_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    assert int(clean["is_exhibition"].sum()) == 8


def test_opponent_is_exhibition_correct_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    # Every exhibition row's opponent is also an exhibition team (All-Star games).
    assert clean.loc[clean["is_exhibition"], "opponent_is_exhibition"].all()
    assert int(clean["opponent_is_exhibition"].sum()) == 8


# --- Sorting ----------------------------------------------------------------

def test_clean_view_is_sorted_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    keys = clean[list(SORT_KEYS)].reset_index(drop=True)
    assert keys.equals(keys.sort_values(list(SORT_KEYS), kind="mergesort").reset_index(drop=True))


# --- Spot-check oracle (clean-model validation, NOT a tool) -----------------

def test_gsw_last5_points_for_oracle_real() -> None:
    clean = build_clean_view(load_raw_dataset())
    gsw = clean[clean["team_name"] == "Golden State Warriors"]
    assert gsw.tail(5)["points_for"].mean() == pytest.approx(114.4, abs=1e-2)


# --- Post-transform validation ---------------------------------------------

def test_validate_clean_view_passes_real() -> None:
    raw = load_raw_dataset()
    clean = build_clean_view(raw)
    summary = validate_clean_view(clean, raw_df=raw)
    assert summary["rows"] == 14_746
    assert summary["exhibition_rows"] == 8
    assert summary["sorted_by"] == list(SORT_KEYS)


def test_validate_clean_view_detects_unsorted() -> None:
    clean = build_clean_view(load_raw_dataset())
    shuffled = clean.iloc[::-1].reset_index(drop=True)  # reverse order
    with pytest.raises(DataValidationError):
        validate_clean_view(shuffled)


def test_validate_clean_view_detects_bad_opponent_is_exhibition() -> None:
    clean = build_clean_view(load_raw_dataset())
    clean.loc[0, "opponent_is_exhibition"] = not bool(clean.loc[0, "opponent_is_exhibition"])
    with pytest.raises(DataValidationError):
        validate_clean_view(clean)


def test_validate_clean_view_rejects_extra_or_reordered_columns() -> None:
    clean = build_clean_view(load_raw_dataset())
    reordered = clean[list(reversed(CLEAN_COLUMNS))]
    with pytest.raises(DataValidationError):
        validate_clean_view(reordered)
    extra = clean.assign(unexpected=1)
    with pytest.raises(DataValidationError):
        validate_clean_view(extra)
    missing = clean.drop(columns=["net_rating"])
    with pytest.raises(DataValidationError):
        validate_clean_view(missing)


def test_validate_clean_view_detects_bad_points_for_mapping() -> None:
    raw = load_raw_dataset()
    clean = build_clean_view(raw)
    clean.loc[0, "points_for"] = clean.loc[0, "points_for"] + 1
    clean.loc[0, "win_flag"] = clean.loc[0, "points_for"] > clean.loc[0, "points_against"]
    with pytest.raises(DataValidationError):
        validate_clean_view(clean, raw_df=raw)


def test_validate_clean_view_detects_bad_points_against_mapping() -> None:
    raw = load_raw_dataset()
    clean = build_clean_view(raw)
    clean.loc[0, "points_against"] = clean.loc[0, "points_against"] + 1
    clean.loc[0, "win_flag"] = clean.loc[0, "points_for"] > clean.loc[0, "points_against"]
    with pytest.raises(DataValidationError):
        validate_clean_view(clean, raw_df=raw)
