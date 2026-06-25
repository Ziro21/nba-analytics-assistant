"""Phase 3A tests: dataset loader and core validators.

Strategy:
  - the happy path runs on the REAL file (the loader + full validator suite);
  - each failure path uses a tiny hand-built frame deliberately corrupted in one way.

No network, no LLM, no guessed expected values — every expected number was computed
from the dataset before being asserted.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import load_raw_dataset
from src.data_validation import (
    EXPECTED_COLUMN_COUNT,
    EXPECTED_ROW_COUNT,
    EXPECTED_SPECIAL_TEAM_ROWS,
    EXPECTED_UNIQUE_GAMES,
    TOOL_COLUMN_DEPENDENCIES,
    DataValidationError,
    build_dataset_profile,
    profile_missingness,
    summarise_tool_column_readiness,
    validate_core_completeness,
    validate_dataset,
    validate_dates,
    validate_game_pair_structure,
    validate_known_identities,
    validate_no_duplicate_ids,
    validate_no_duplicate_rows,
    validate_opponent_points_consistency,
    validate_points_identities,
    validate_required_columns,
    validate_season_ids,
    validate_shape,
    validate_special_teams,
    validate_three_point_trap,
)


def make_valid_frame() -> pd.DataFrame:
    """A tiny, internally consistent 2-game (4-row) long-format frame.

    Satisfies every structural/identity validator (but NOT the dataset-size checks,
    which are specific to the real 14,746-row file). Designed so that:
      - dates span exactly the real dataset bounds (2020-12-23 .. 2026-03-05);
      - the three-point pairs disagree on exactly one row each.
    """
    return pd.DataFrame(
        {
            "_id": [1, 2, 3, 4],
            "season_id": [26, 26, 26, 26],
            "game_id": [100, 100, 101, 101],
            "match_date": [
                "2020-12-23 03:00:00",  # dataset min
                "2021-01-05 03:00:00",
                "2026-03-05 03:00:00",  # dataset max
                "2025-02-10 03:00:00",
            ],
            "team_id": [1, 2, 1, 3],
            "team_name": ["Atlanta Hawks", "Boston Celtics", "Atlanta Hawks", "Chicago Bulls"],
            "is_home": [1, 0, 0, 1],
            "team_points": [110, 100, 120, 115],
            "opponent_points": [100, 110, 115, 120],
            "total_points": [110, 100, 120, 115],
            "plus_minus": [10, -10, 5, -5],
            "possessions": [100, 100, 102, 102],
            "ORTG": [110, 100, 118, 113],
            "DRTG": [100, 110, 113, 118],
            "fg_points": [88, 80, 90, 85],
            "ft_points": [22, 20, 30, 30],
            "FTA": [25, 20, 33, 32],
            "ft_attempts": [25, 20, 33, 32],
            "FTM": [22, 20, 30, 30],
            "ft_makes": [22, 20, 30, 30],
            "3FGA": [40, 35, 42, 38],
            "three_attempts": [39, 35, 42, 38],  # differs on row 0
            "3FGM": [15, 12, 16, 14],
            "three_makes": [14, 12, 16, 14],      # differs on row 0
            "our_fixture_id": [5000, 5000, 5001, 5001],
        }
    )


# --- Loader -----------------------------------------------------------------

def test_loader_drops_unnamed_and_keeps_id() -> None:
    df = load_raw_dataset()
    assert "Unnamed: 0" not in df.columns
    assert "_id" in df.columns


def test_loader_expected_shape() -> None:
    df = load_raw_dataset()
    assert df.shape == (EXPECTED_ROW_COUNT, EXPECTED_COLUMN_COUNT)


# --- Required columns -------------------------------------------------------

def test_required_columns_present_on_real_file() -> None:
    assert validate_required_columns(load_raw_dataset()) > 0


def test_missing_required_column_fails() -> None:
    broken = make_valid_frame().drop(columns=["team_points"])
    with pytest.raises(DataValidationError):
        validate_required_columns(broken)


# --- Duplicates -------------------------------------------------------------

def test_duplicate_row_fails() -> None:
    broken = pd.concat([make_valid_frame(), make_valid_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError):
        validate_no_duplicate_rows(broken)


def test_duplicate_id_fails() -> None:
    broken = make_valid_frame()
    broken.loc[3, "_id"] = broken.loc[0, "_id"]
    with pytest.raises(DataValidationError):
        validate_no_duplicate_ids(broken)


# --- Game-pair structure ----------------------------------------------------

def test_game_pair_structure_passes_on_valid_frame() -> None:
    validate_game_pair_structure(make_valid_frame())  # must not raise


def test_one_row_game_fails() -> None:
    broken = make_valid_frame().drop(index=1)  # game 100 now has a single row
    with pytest.raises(DataValidationError):
        validate_game_pair_structure(broken)


def test_two_home_rows_fails() -> None:
    broken = make_valid_frame()
    broken.loc[1, "is_home"] = 1  # game 100 now has two home rows
    with pytest.raises(DataValidationError):
        validate_game_pair_structure(broken)


# --- Points identities ------------------------------------------------------

def test_corrupted_plus_minus_fails() -> None:
    broken = make_valid_frame()
    broken.loc[0, "plus_minus"] = 999
    with pytest.raises(DataValidationError):
        validate_points_identities(broken)


def test_corrupted_total_points_fails() -> None:
    broken = make_valid_frame()
    broken.loc[0, "total_points"] = 999
    with pytest.raises(DataValidationError):
        validate_points_identities(broken)


def test_corrupted_opponent_points_fails() -> None:
    broken = make_valid_frame()
    broken.loc[0, "opponent_points"] = 999  # no longer equals paired team_points
    with pytest.raises(DataValidationError):
        validate_opponent_points_consistency(broken)


# --- Dates ------------------------------------------------------------------

def test_dates_pass_on_valid_frame() -> None:
    result = validate_dates(make_valid_frame())
    assert result == {"date_min": "2020-12-23", "date_max": "2026-03-05"}


def test_out_of_range_date_fails() -> None:
    broken = make_valid_frame()
    broken.loc[2, "match_date"] = "2030-01-01 03:00:00"
    with pytest.raises(DataValidationError):
        validate_dates(broken)


def test_unparseable_date_fails() -> None:
    broken = make_valid_frame()
    broken.loc[0, "match_date"] = "not-a-real-date"
    with pytest.raises(DataValidationError):
        validate_dates(broken)


# --- Season ids -------------------------------------------------------------

def test_season_ids_pass_on_valid_frame() -> None:
    assert validate_season_ids(make_valid_frame()) == [26]


def test_unexpected_season_id_fails() -> None:
    broken = make_valid_frame()
    broken.loc[0, "season_id"] = 99
    with pytest.raises(DataValidationError):
        validate_season_ids(broken)


# --- Special teams ----------------------------------------------------------

def test_special_teams_report_on_real_file() -> None:
    summary = validate_special_teams(load_raw_dataset())
    assert summary["special_team_rows"] == EXPECTED_SPECIAL_TEAM_ROWS
    assert summary["team_name_count"] == 33
    assert summary["fixture_null_alignment_ok"] is True


def test_special_team_fixture_misalignment_fails() -> None:
    broken = make_valid_frame()
    broken.loc[0, "team_name"] = "Team Stars"  # special row, but fixture id not null
    with pytest.raises(DataValidationError):
        validate_special_teams(broken)


# --- Known identities & three-point trap ------------------------------------

def test_known_identities_pass_on_valid_frame() -> None:
    results = validate_known_identities(make_valid_frame())
    assert all(v == 0 for v in results.values())


def test_corrupted_identity_fails() -> None:
    broken = make_valid_frame()
    broken.loc[0, "fg_points"] = 1  # team_points no longer equals fg_points + ft_points
    with pytest.raises(DataValidationError):
        validate_known_identities(broken)


def test_three_point_trap_counts_disagreements() -> None:
    result = validate_three_point_trap(make_valid_frame())
    assert result["3FGA vs three_attempts"] == 1
    assert result["3FGM vs three_makes"] == 1


# --- Shape failure & full happy path ---------------------------------------

def test_shape_fails_on_small_frame() -> None:
    with pytest.raises(DataValidationError):
        validate_shape(make_valid_frame())


def test_validate_dataset_happy_path_on_real_file() -> None:
    summary = validate_dataset(load_raw_dataset())
    assert summary["rows"] == EXPECTED_ROW_COUNT
    assert summary["columns"] == EXPECTED_COLUMN_COUNT
    assert summary["unique_games"] == EXPECTED_UNIQUE_GAMES
    assert summary["core_complete"] is True
    assert summary["duplicate_ids"] == 0
    assert summary["duplicate_rows"] == 0
    assert summary["plus_minus_violations"] == 0
    assert summary["team_points_total_violations"] == 0
    assert summary["opponent_points_violations"] == 0
    assert summary["date_min"] == "2020-12-23"
    assert summary["date_max"] == "2026-03-05"
    assert summary["season_ids"] == [26, 28, 30, 32, 34, 36]
    assert summary["special_team_rows"] == EXPECTED_SPECIAL_TEAM_ROWS
    assert summary["team_name_count"] == 33
    assert summary["fixture_null_alignment_ok"] is True
    assert summary["three_point_disagreements"] == {
        "3FGA vs three_attempts": 1200,
        "3FGM vs three_makes": 1200,
    }


# --- Phase 3B: missingness, core completeness, tool readiness ----------------

def test_core_completeness_passes_on_real_file() -> None:
    null_counts = validate_core_completeness(load_raw_dataset())
    assert all(v == 0 for v in null_counts.values())


def test_core_completeness_detects_injected_null() -> None:
    broken = make_valid_frame()
    broken.loc[0, "ORTG"] = float("nan")
    with pytest.raises(DataValidationError):
        validate_core_completeness(broken)


def test_missingness_summary_structure_on_real_file() -> None:
    summary = profile_missingness(load_raw_dataset())
    assert summary["core_complete"] is True
    assert summary["core_columns_with_nulls"] == {}
    assert summary["columns_with_nulls"] == 50
    assert summary["null_count_min"] == 8
    assert summary["null_count_max"] == 62
    assert len(summary["advanced_columns_with_nulls"]) == 50
    # Advanced rows carry column/count/pct and are sorted most-null first (stable).
    counts = [row["count"] for row in summary["advanced_columns_with_nulls"]]
    assert counts == sorted(counts, reverse=True)
    assert {"column", "count", "pct"} <= set(summary["advanced_columns_with_nulls"][0])


def test_profiling_does_not_impute_or_modify_frame() -> None:
    df = load_raw_dataset()
    before = int(df.isna().sum().sum())
    profile_missingness(df)
    build_dataset_profile(df)
    after = int(df.isna().sum().sum())
    assert before == after  # nulls untouched: no zero-filling, no imputation
    assert before > 0       # advanced nulls genuinely present and retained


def test_advanced_nullable_columns_are_reported() -> None:
    summary = profile_missingness(load_raw_dataset())
    reported = {row["column"] for row in summary["advanced_columns_with_nulls"]}
    # A known advanced play-type column with nulls is surfaced, not hidden/imputed.
    assert "Post_attempts" in reported


def test_tool_column_readiness_on_real_file() -> None:
    report = summarise_tool_column_readiness(load_raw_dataset())
    assert set(report) == set(TOOL_COLUMN_DEPENDENCIES)
    for tool, info in report.items():
        assert info["all_present"] is True, tool
        assert info["all_complete"] is True, tool
        assert info["null_columns"] == []
