"""Phase 5A–5G tests: tool result contract, shared dataframe helpers, and all six
analytical tools (``team_average_points``, ``average_points_allowed``, ``team_record``,
``top_scoring_teams``, ``head_to_head``, ``team_efficiency_summary``).

Integration tests build the real clean frame through the real pipeline. No network, no LLM.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import src.tools as tools_module
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.tool_results import (
    build_meta,
    error_result,
    no_data_result,
    ok_result,
)
from src.tools import (
    apply_window,
    average_points_allowed,
    date_range_for,
    filter_franchise_games,
    filter_team_games,
    head_to_head,
    team_average_points,
    team_efficiency_summary,
    team_record,
    top_scoring_teams,
)

META_KEYS = {"team", "games_used", "date_range", "window_requested", "season_id"}
TOP_LEVEL_KEYS = {"status", "tool", "result", "meta", "warnings"}

# All six analytical tools are implemented after Phase 5G.
IMPLEMENTED_TOOL_NAMES = (
    "team_average_points",
    "average_points_allowed",
    "team_record",
    "top_scoring_teams",
    "head_to_head",
    "team_efficiency_summary",
)
PENDING_TOOL_NAMES: tuple[str, ...] = ()


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return clean


def make_tie_frame() -> pd.DataFrame:
    """Two franchises with identical mean points_for, to exercise the tie-break rule.

    Alpha Team: 100, 110 -> mean 105.  Zeta Team: 105, 105 -> mean 105.
    The alphabetically-earlier team ("Alpha Team") must rank first on the tie.
    """
    return pd.DataFrame(
        {
            "team_name": ["Zeta Team", "Zeta Team", "Alpha Team", "Alpha Team"],
            "points_for": [105, 105, 100, 110],
            "season_id": [26, 26, 26, 26],
            "game_date": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]
            ),
            "is_exhibition": [False, False, False, False],
            "opponent_is_exhibition": [False, False, False, False],
        }
    )


def make_efficiency_frame() -> pd.DataFrame:
    """One team, two games with DIFFERENT possessions, to distinguish a per-game mean
    from a possession-weighted aggregate.

    ortg [100, 120] -> per-game mean 110.  Possession-weighted (poss 100, 200) would be
    (100*100 + 120*200) / (100+200) = 113.33. The tool must report the per-game mean.
    """
    return pd.DataFrame(
        {
            "team_name": ["Alpha Team", "Alpha Team"],
            "ortg": [100.0, 120.0],
            "drtg": [100.0, 100.0],
            "net_rating": [0.0, 20.0],
            "possessions": [100, 200],
            "game_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "is_exhibition": [False, False],
            "opponent_is_exhibition": [False, False],
        }
    )


def small_df() -> pd.DataFrame:
    """A 10-row frame with ascending dates and a position marker, for window/date tests."""
    return pd.DataFrame(
        {
            "game_date": pd.to_datetime([f"2021-01-{d:02d} 03:00:00" for d in range(1, 11)]),
            "x": list(range(10)),
        }
    )


# --- A. ok_result -----------------------------------------------------------

def test_ok_result_shape_and_serialisable() -> None:
    res = ok_result("demo_tool", {"value": 1})
    assert set(res) == TOP_LEVEL_KEYS
    assert res["status"] == "ok"
    assert res["tool"] == "demo_tool"
    assert res["warnings"] == []
    assert set(res["meta"]) == META_KEYS
    json.dumps(res)  # must not raise


# --- B. no_data_result ------------------------------------------------------

def test_no_data_result() -> None:
    res = no_data_result("demo_tool", warnings=["none found"])
    assert res["status"] == "no_data"
    assert res["warnings"] == ["none found"]
    assert res["result"] == {}
    assert set(res["meta"]) == META_KEYS
    json.dumps(res)


# --- C. error_result --------------------------------------------------------

def test_error_result() -> None:
    res = error_result("demo_tool", "bad input")
    assert res["status"] == "error"
    assert res["result"]["message"] == "bad input"
    assert set(res) == TOP_LEVEL_KEYS
    json.dumps(res)


# --- D. build_meta ----------------------------------------------------------

def test_build_meta_defaults_and_preserves() -> None:
    empty = build_meta()
    assert set(empty) == META_KEYS
    assert all(v is None for v in empty.values())

    full = build_meta(
        team="Golden State Warriors",
        games_used=5,
        date_range=["2020-12-23", "2026-03-05"],
        window_requested=5,
        season_id=26,
    )
    assert full["team"] == "Golden State Warriors"
    assert full["games_used"] == 5
    assert full["date_range"] == ["2020-12-23", "2026-03-05"]
    assert full["window_requested"] == 5
    assert full["season_id"] == 26
    json.dumps(full)


# --- E. filter_franchise_games ---------------------------------------------

def test_filter_franchise_removes_exhibition(clean_df: pd.DataFrame) -> None:
    before = clean_df.copy(deep=True)
    franchise = filter_franchise_games(clean_df)
    assert len(franchise) == len(clean_df) - 8           # the 8 All-Star rows removed
    assert not franchise["is_exhibition"].any()
    assert not franchise["opponent_is_exhibition"].any()
    assert franchise.index.is_monotonic_increasing       # order preserved
    assert clean_df.equals(before)                       # input not mutated


# --- F. filter_team_games ---------------------------------------------------

def test_filter_team_exact_match(clean_df: pd.DataFrame) -> None:
    before = clean_df.copy(deep=True)
    gsw = filter_team_games(clean_df, "Golden State Warriors")
    assert len(gsw) > 0
    assert (gsw["team_name"] == "Golden State Warriors").all()
    assert gsw.index.is_monotonic_increasing
    assert clean_df.equals(before)


def test_filter_team_unknown_returns_empty(clean_df: pd.DataFrame) -> None:
    assert filter_team_games(clean_df, "Nonexistent Team").empty


def test_filter_team_no_alias(clean_df: pd.DataFrame) -> None:
    # "Warriors" is an alias, not a canonical team_name → no match at the tool layer.
    assert filter_team_games(clean_df, "Warriors").empty


# --- G. apply_window --------------------------------------------------------

def test_apply_window_none_returns_all() -> None:
    df = small_df()
    out, warnings = apply_window(df, None)
    assert len(out) == len(df)
    assert warnings == []


def test_apply_window_last_n_preserves_order() -> None:
    df = small_df()
    out, warnings = apply_window(df, 5)
    assert out["x"].tolist() == [5, 6, 7, 8, 9]
    assert warnings == []


def test_apply_window_over_large_returns_all_with_warning() -> None:
    df = small_df()
    out, warnings = apply_window(df, 999)
    assert len(out) == len(df)
    assert len(warnings) == 1


def test_apply_window_zero_raises() -> None:
    with pytest.raises(ValueError):
        apply_window(small_df(), 0)


def test_apply_window_negative_raises() -> None:
    with pytest.raises(ValueError):
        apply_window(small_df(), -3)


def test_apply_window_bool_raises() -> None:
    with pytest.raises(ValueError):
        apply_window(small_df(), True)


def test_apply_window_non_int_raises() -> None:
    with pytest.raises(ValueError):
        apply_window(small_df(), "5")


def test_apply_window_does_not_mutate() -> None:
    df = small_df()
    before = df.copy(deep=True)
    apply_window(df, 5)
    assert df.equals(before)


# --- H. date_range_for ------------------------------------------------------

def test_date_range_for_non_empty() -> None:
    assert date_range_for(small_df()) == ["2021-01-01", "2021-01-10"]


def test_date_range_for_empty() -> None:
    empty = pd.DataFrame({"game_date": pd.to_datetime([])})
    assert date_range_for(empty) is None


# --- I. scope / import sanity ----------------------------------------------

def test_pending_analytical_tools_not_implemented_yet() -> None:
    for name in IMPLEMENTED_TOOL_NAMES:
        assert hasattr(tools_module, name), f"{name} should exist"
    for name in PENDING_TOOL_NAMES:
        assert not hasattr(tools_module, name), f"{name} should not exist yet"


# --- Phase 5B: team_average_points -----------------------------------------

@pytest.mark.parametrize(
    "team,expected",
    [
        ("Golden State Warriors", 114.4),
        ("Boston Celtics", 108.6),
        ("Los Angeles Lakers", 117.2),
    ],
)
def test_team_average_points_oracle_last5(clean_df, team, expected) -> None:
    res = team_average_points(clean_df, team, window=5)
    assert res["status"] == "ok"
    assert res["result"]["average_points"] == pytest.approx(expected, abs=1e-2)
    assert res["meta"]["team"] == team
    assert res["meta"]["games_used"] == 5
    assert res["meta"]["window_requested"] == 5
    assert res["meta"]["date_range"] is not None
    json.dumps(res)


def test_team_average_points_window_none_uses_all(clean_df) -> None:
    res = team_average_points(clean_df, "Golden State Warriors", window=None)
    assert res["status"] == "ok"
    assert res["meta"]["games_used"] == 512  # GSW franchise games (matches 289-223 record)
    assert res["meta"]["window_requested"] is None
    assert res["warnings"] == []


def test_team_average_points_over_large_window_warns(clean_df) -> None:
    res = team_average_points(clean_df, "Golden State Warriors", window=10_000)
    assert res["status"] == "ok"
    assert res["meta"]["games_used"] == 512
    assert len(res["warnings"]) == 1


def test_team_average_points_window_zero_errors(clean_df) -> None:
    res = team_average_points(clean_df, "Golden State Warriors", window=0)
    assert res["status"] == "error"
    assert "message" in res["result"]


def test_team_average_points_unknown_team_no_data(clean_df) -> None:
    res = team_average_points(clean_df, "Nonexistent Team", window=5)
    assert res["status"] == "no_data"
    assert res["warnings"]  # a clear "no games found" warning is present
    assert res["meta"]["team"] == "Nonexistent Team"
    json.dumps(res)


def test_team_average_points_invalid_window_errors_even_for_unknown_team(clean_df) -> None:
    # An invalid argument errors regardless of whether the team exists.
    res = team_average_points(clean_df, "Nonexistent Team", window=0)
    assert res["status"] == "error"


def test_team_average_points_does_not_mutate_clean_df(clean_df) -> None:
    before = clean_df.copy(deep=True)
    team_average_points(clean_df, "Golden State Warriors", window=5)
    assert clean_df.equals(before)


# --- Phase 5C: average_points_allowed --------------------------------------

def test_average_points_allowed_oracle_last5(clean_df) -> None:
    res = average_points_allowed(clean_df, "Golden State Warriors", window=5)
    assert res["status"] == "ok"
    assert res["tool"] == "average_points_allowed"
    assert res["result"]["average_points_allowed"] == pytest.approx(117.0, abs=1e-2)
    assert res["result"]["games_used"] == 5
    assert res["result"]["team"] == "Golden State Warriors"
    assert res["meta"]["team"] == "Golden State Warriors"
    assert res["meta"]["games_used"] == 5
    assert res["meta"]["window_requested"] == 5
    json.dumps(res)


def test_average_points_allowed_window_none_uses_all(clean_df) -> None:
    res = average_points_allowed(clean_df, "Golden State Warriors", window=None)
    assert res["status"] == "ok"
    assert res["meta"]["games_used"] == 512
    assert res["meta"]["window_requested"] is None
    assert res["warnings"] == []


def test_average_points_allowed_over_large_window_warns(clean_df) -> None:
    res = average_points_allowed(clean_df, "Golden State Warriors", window=10_000)
    assert res["status"] == "ok"
    assert res["meta"]["games_used"] == 512
    assert len(res["warnings"]) == 1


def test_average_points_allowed_window_zero_errors(clean_df) -> None:
    res = average_points_allowed(clean_df, "Golden State Warriors", window=0)
    assert res["status"] == "error"


def test_average_points_allowed_unknown_team_no_data(clean_df) -> None:
    res = average_points_allowed(clean_df, "Not A Real Team", window=5)
    assert res["status"] == "no_data"
    assert res["result"]["games_used"] == 0
    assert res["meta"]["games_used"] == 0
    assert res["meta"]["date_range"] is None
    assert res["warnings"]
    json.dumps(res)


def test_average_points_allowed_unknown_team_invalid_window_errors(clean_df) -> None:
    res = average_points_allowed(clean_df, "Not A Real Team", window=0)
    assert res["status"] == "error"


def test_average_points_allowed_does_not_mutate_clean_df(clean_df) -> None:
    before = clean_df.copy(deep=True)
    average_points_allowed(clean_df, "Golden State Warriors", window=5)
    assert clean_df.equals(before)


def test_average_points_allowed_bool_window_errors(clean_df) -> None:
    res = average_points_allowed(clean_df, "Golden State Warriors", window=True)
    assert res["status"] == "error"


def test_average_points_allowed_non_int_window_errors(clean_df) -> None:
    res = average_points_allowed(clean_df, "Golden State Warriors", window="5")
    assert res["status"] == "error"


def test_average_points_allowed_no_data_warning_mentions_team(clean_df) -> None:
    res = average_points_allowed(clean_df, "Not A Real Team", window=5)
    assert res["status"] == "no_data"
    assert any("Not A Real Team" in w for w in res["warnings"])


# --- Phase 5D: team_record --------------------------------------------------

@pytest.mark.parametrize(
    "team,wins,losses",
    [
        ("Golden State Warriors", 289, 223),
        ("Boston Celtics", 359, 183),
        ("Los Angeles Lakers", 267, 228),
    ],
)
def test_team_record_oracle_all_games(clean_df, team, wins, losses) -> None:
    res = team_record(clean_df, team, window=None)
    games = wins + losses
    assert res["status"] == "ok"
    assert res["tool"] == "team_record"
    assert res["result"]["wins"] == wins
    assert res["result"]["losses"] == losses
    assert res["result"]["record"] == f"{wins}-{losses}"
    assert res["result"]["games_used"] == games
    assert res["result"]["win_percentage"] == wins / games
    json.dumps(res)


def test_team_record_metadata(clean_df) -> None:
    res = team_record(clean_df, "Golden State Warriors", window=None)
    assert res["meta"]["team"] == "Golden State Warriors"
    assert res["meta"]["games_used"] == 512
    assert res["meta"]["window_requested"] is None
    assert isinstance(res["meta"]["date_range"], list) and len(res["meta"]["date_range"]) == 2
    assert res["meta"]["season_id"] is None


def test_team_record_window_none_no_warning(clean_df) -> None:
    res = team_record(clean_df, "Golden State Warriors", window=None)
    assert res["status"] == "ok"
    assert res["warnings"] == []


def test_team_record_positive_window_internal_consistency(clean_df) -> None:
    res = team_record(clean_df, "Golden State Warriors", window=5)
    r = res["result"]
    assert res["status"] == "ok"
    assert r["games_used"] == 5
    assert r["wins"] + r["losses"] == 5
    assert res["meta"]["window_requested"] == 5
    assert r["win_percentage"] == r["wins"] / 5


def test_team_record_over_large_window_warns(clean_df) -> None:
    res = team_record(clean_df, "Golden State Warriors", window=10_000)
    assert res["status"] == "ok"
    assert res["result"]["games_used"] == 512
    assert len(res["warnings"]) == 1


def test_team_record_window_zero_errors(clean_df) -> None:
    res = team_record(clean_df, "Golden State Warriors", window=0)
    assert res["status"] == "error"


def test_team_record_unknown_team_no_data(clean_df) -> None:
    res = team_record(clean_df, "Not A Real Team", window=5)
    assert res["status"] == "no_data"
    assert res["result"]["games_used"] == 0
    assert res["meta"]["games_used"] == 0
    assert res["meta"]["date_range"] is None
    assert res["warnings"]
    json.dumps(res)


def test_team_record_unknown_team_invalid_window_errors(clean_df) -> None:
    res = team_record(clean_df, "Not A Real Team", window=0)
    assert res["status"] == "error"


def test_team_record_does_not_mutate_clean_df(clean_df) -> None:
    before = clean_df.copy(deep=True)
    team_record(clean_df, "Golden State Warriors", window=5)
    assert clean_df.equals(before)


def test_team_record_bool_window_errors(clean_df) -> None:
    assert team_record(clean_df, "Golden State Warriors", window=True)["status"] == "error"


def test_team_record_non_int_window_errors(clean_df) -> None:
    assert team_record(clean_df, "Golden State Warriors", window="5")["status"] == "error"


def test_team_record_no_data_warning_mentions_team(clean_df) -> None:
    res = team_record(clean_df, "Not A Real Team", window=5)
    assert res["status"] == "no_data"
    assert any("Not A Real Team" in w for w in res["warnings"])


# --- Phase 5E: top_scoring_teams -------------------------------------------

ALL_TIME_TOP5 = [
    ("Atlanta Hawks", 116.13),
    ("Indiana Pacers", 115.94),
    ("Milwaukee Bucks", 115.84),
    ("Denver Nuggets", 115.67),
    ("Utah Jazz", 115.08),
]


def test_top_scoring_teams_oracle_all_time(clean_df) -> None:
    res = top_scoring_teams(clean_df, n=5)
    assert res["status"] == "ok"
    assert res["tool"] == "top_scoring_teams"
    assert res["result"]["teams_returned"] == 5
    assert res["result"]["n_requested"] == 5
    teams = res["result"]["teams"]
    assert [t["team"] for t in teams] == [name for name, _ in ALL_TIME_TOP5]
    for item, (_, expected_avg) in zip(teams, ALL_TIME_TOP5):
        assert {"rank", "team", "average_points", "games_used"} <= set(item)
        assert isinstance(item["average_points"], float)
        assert round(item["average_points"], 2) == expected_avg  # rounded display only
    assert [t["rank"] for t in teams] == [1, 2, 3, 4, 5]
    json.dumps(res)


def test_top_scoring_teams_metadata(clean_df) -> None:
    res = top_scoring_teams(clean_df, n=5)
    assert res["meta"]["team"] is None
    assert res["meta"]["games_used"] == len(filter_franchise_games(clean_df))
    assert isinstance(res["meta"]["date_range"], list) and len(res["meta"]["date_range"]) == 2
    assert res["meta"]["window_requested"] is None
    assert res["meta"]["season_id"] is None


def test_top_scoring_teams_season_filter(clean_df) -> None:
    season = 34
    res = top_scoring_teams(clean_df, n=5, season_id=season)
    assert res["status"] == "ok"
    assert res["meta"]["season_id"] == season
    # Independent recomputation from the clean frame (no hard-coded season ranking).
    sub = filter_franchise_games(clean_df)
    sub = sub[sub["season_id"] == season]
    assert res["meta"]["games_used"] == len(sub)
    expected = (
        sub.groupby("team_name")["points_for"].mean().reset_index()
        .sort_values(["points_for", "team_name"], ascending=[False, True], kind="mergesort")
        .head(5)["team_name"].tolist()
    )
    assert [t["team"] for t in res["result"]["teams"]] == expected


def test_top_scoring_teams_season_no_rows(clean_df) -> None:
    res = top_scoring_teams(clean_df, n=5, season_id=999)
    assert res["status"] == "no_data"
    assert res["warnings"]
    assert res["meta"]["season_id"] == 999
    assert res["meta"]["games_used"] == 0
    assert res["meta"]["date_range"] is None


@pytest.mark.parametrize("bad_season", ["36", True])
def test_top_scoring_teams_invalid_season_errors(clean_df, bad_season) -> None:
    res = top_scoring_teams(clean_df, n=5, season_id=bad_season)
    assert res["status"] == "error"
    assert "message" in res["result"]


@pytest.mark.parametrize("bad_n", [0, -1, True, "5"])
def test_top_scoring_teams_invalid_n_errors(clean_df, bad_n) -> None:
    res = top_scoring_teams(clean_df, n=bad_n)
    assert res["status"] == "error"
    assert "message" in res["result"]


def test_top_scoring_teams_over_large_n(clean_df) -> None:
    total_teams = filter_franchise_games(clean_df)["team_name"].nunique()
    res = top_scoring_teams(clean_df, n=10_000)
    assert res["status"] == "ok"
    assert res["result"]["teams_returned"] == total_teams
    assert len(res["warnings"]) == 1
    assert "10000" in res["warnings"][0] and str(total_teams) in res["warnings"][0]
    avgs = [t["average_points"] for t in res["result"]["teams"]]
    assert avgs == sorted(avgs, reverse=True)


def test_top_scoring_teams_tie_breaks_by_team_name() -> None:
    res = top_scoring_teams(make_tie_frame(), n=2)
    assert res["status"] == "ok"
    teams = res["result"]["teams"]
    # Both teams have mean 105; alphabetical order wins the tie.
    assert teams[0]["team"] == "Alpha Team"
    assert teams[1]["team"] == "Zeta Team"
    assert teams[0]["average_points"] == teams[1]["average_points"] == pytest.approx(105.0)


def test_top_scoring_teams_does_not_mutate_clean_df(clean_df) -> None:
    before = clean_df.copy(deep=True)
    top_scoring_teams(clean_df, n=5)
    assert clean_df.equals(before)


# --- Phase 5F: head_to_head -------------------------------------------------

def test_head_to_head_oracle_celtics_heat(clean_df) -> None:
    res = head_to_head(clean_df, "Boston Celtics", "Miami Heat", window=None)
    assert res["status"] == "ok"
    assert res["tool"] == "head_to_head"
    r = res["result"]
    assert r["team_a"] == "Boston Celtics"
    assert r["team_b"] == "Miami Heat"
    assert r["meetings"] == 39
    assert r["team_a_wins"] == 25
    assert r["team_b_wins"] == 14
    assert r["record"] == "25-14"
    assert r["average_point_differential"] == pytest.approx(
        r["average_points_for"] - r["average_points_against"]
    )
    json.dumps(res)


def test_head_to_head_reverse_direction(clean_df) -> None:
    res = head_to_head(clean_df, "Miami Heat", "Boston Celtics", window=None)
    r = res["result"]
    assert r["meetings"] == 39
    assert r["team_a_wins"] == 14
    assert r["team_b_wins"] == 25
    assert r["record"] == "14-25"


def test_head_to_head_symmetry(clean_df) -> None:
    ab = head_to_head(clean_df, "Boston Celtics", "Miami Heat")["result"]
    ba = head_to_head(clean_df, "Miami Heat", "Boston Celtics")["result"]
    assert ab["meetings"] == ba["meetings"]
    assert ab["team_a_wins"] == ba["team_b_wins"]
    assert ab["team_b_wins"] == ba["team_a_wins"]


def test_head_to_head_metadata(clean_df) -> None:
    res = head_to_head(clean_df, "Boston Celtics", "Miami Heat", window=None)
    assert res["meta"]["team"] == "Boston Celtics"
    assert res["meta"]["games_used"] == 39
    assert res["meta"]["window_requested"] is None
    assert isinstance(res["meta"]["date_range"], list) and len(res["meta"]["date_range"]) == 2
    assert res["meta"]["season_id"] is None


def test_head_to_head_positive_window(clean_df) -> None:
    res = head_to_head(clean_df, "Boston Celtics", "Miami Heat", window=5)
    r = res["result"]
    assert res["status"] == "ok"
    assert r["meetings"] == 5
    assert r["team_a_wins"] + r["team_b_wins"] == 5
    assert res["meta"]["games_used"] == 5
    assert res["meta"]["window_requested"] == 5


def test_head_to_head_over_large_window_warns(clean_df) -> None:
    res = head_to_head(clean_df, "Boston Celtics", "Miami Heat", window=10_000)
    assert res["status"] == "ok"
    assert res["result"]["meetings"] == 39
    assert res["meta"]["games_used"] == 39
    assert len(res["warnings"]) == 1


def test_head_to_head_window_zero_errors(clean_df) -> None:
    res = head_to_head(clean_df, "Boston Celtics", "Miami Heat", window=0)
    assert res["status"] == "error"
    assert "message" in res["result"]


def test_head_to_head_same_team_errors(clean_df) -> None:
    res = head_to_head(clean_df, "Boston Celtics", "Boston Celtics")
    assert res["status"] == "error"
    assert "message" in res["result"]


def test_head_to_head_unknown_team_no_data(clean_df) -> None:
    res = head_to_head(clean_df, "Not A Real Team", "Miami Heat", window=5)
    assert res["status"] == "no_data"
    assert res["meta"]["games_used"] == 0
    assert res["meta"]["date_range"] is None
    assert res["warnings"]
    json.dumps(res)


def test_head_to_head_unknown_team_invalid_window_errors(clean_df) -> None:
    res = head_to_head(clean_df, "Not A Real Team", "Miami Heat", window=0)
    assert res["status"] == "error"


def test_head_to_head_does_not_mutate_clean_df(clean_df) -> None:
    before = clean_df.copy(deep=True)
    head_to_head(clean_df, "Boston Celtics", "Miami Heat", window=5)
    assert clean_df.equals(before)


def test_head_to_head_bool_window_errors(clean_df) -> None:
    assert head_to_head(clean_df, "Boston Celtics", "Miami Heat", window=True)["status"] == "error"


def test_head_to_head_non_int_window_errors(clean_df) -> None:
    assert head_to_head(clean_df, "Boston Celtics", "Miami Heat", window="5")["status"] == "error"


def test_head_to_head_no_data_warning_mentions_both_teams(clean_df) -> None:
    res = head_to_head(clean_df, "Not A Real Team", "Miami Heat", window=5)
    assert res["status"] == "no_data"
    warning = " ".join(res["warnings"])
    assert "Not A Real Team" in warning and "Miami Heat" in warning


# --- Phase 5G: team_efficiency_summary -------------------------------------

def test_team_efficiency_summary_oracle_celtics_last10(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Boston Celtics", window=10)
    assert res["status"] == "ok"
    assert res["tool"] == "team_efficiency_summary"
    r = res["result"]
    assert r["team"] == "Boston Celtics"
    assert r["games_used"] == 10
    assert round(r["average_ortg"], 2) == 106.98  # rounded display only
    assert round(r["average_drtg"], 2) == 101.93
    assert "average_net_rating" in r
    json.dumps(res)


def test_team_efficiency_summary_oracle_gsw_last10(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Golden State Warriors", window=10)
    assert res["status"] == "ok"
    assert res["result"]["games_used"] == 10
    assert round(res["result"]["average_ortg"], 2) == 105.57
    assert round(res["result"]["average_drtg"], 2) == 109.17
    json.dumps(res)


def test_team_efficiency_summary_metadata(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Boston Celtics", window=10)
    assert res["meta"]["team"] == "Boston Celtics"
    assert res["meta"]["games_used"] == 10
    assert res["meta"]["window_requested"] == 10
    assert isinstance(res["meta"]["date_range"], list) and len(res["meta"]["date_range"]) == 2
    assert res["meta"]["season_id"] is None


def test_team_efficiency_summary_window_none_uses_all(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Boston Celtics", window=None)
    expected_games = len(filter_team_games(clean_df, "Boston Celtics"))
    assert res["status"] == "ok"
    assert res["result"]["games_used"] == expected_games
    assert res["meta"]["window_requested"] is None
    assert res["warnings"] == []


def test_team_efficiency_summary_positive_window(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Boston Celtics", window=10)
    assert res["result"]["games_used"] == 10
    assert res["meta"]["window_requested"] == 10


def test_team_efficiency_summary_over_large_window_warns(clean_df) -> None:
    expected_games = len(filter_team_games(clean_df, "Boston Celtics"))
    res = team_efficiency_summary(clean_df, "Boston Celtics", window=10_000)
    assert res["status"] == "ok"
    assert res["result"]["games_used"] == expected_games
    assert len(res["warnings"]) == 1


def test_team_efficiency_summary_window_zero_errors(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Boston Celtics", window=0)
    assert res["status"] == "error"


def test_team_efficiency_summary_unknown_team_no_data(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Not A Real Team", window=10)
    assert res["status"] == "no_data"
    assert res["result"]["games_used"] == 0
    assert res["meta"]["games_used"] == 0
    assert res["meta"]["date_range"] is None
    assert res["warnings"]
    json.dumps(res)


def test_team_efficiency_summary_unknown_team_invalid_window_errors(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Not A Real Team", window=0)
    assert res["status"] == "error"


def test_team_efficiency_summary_does_not_mutate_clean_df(clean_df) -> None:
    before = clean_df.copy(deep=True)
    team_efficiency_summary(clean_df, "Boston Celtics", window=10)
    assert clean_df.equals(before)


def test_team_efficiency_summary_net_rating_consistency(clean_df) -> None:
    res = team_efficiency_summary(clean_df, "Boston Celtics", window=10)
    r = res["result"]
    # net_rating column mean equals mean(ortg) - mean(drtg) up to float error.
    assert r["average_net_rating"] == pytest.approx(r["average_ortg"] - r["average_drtg"])


def test_team_efficiency_summary_bool_window_errors(clean_df) -> None:
    assert team_efficiency_summary(clean_df, "Boston Celtics", window=True)["status"] == "error"


def test_team_efficiency_summary_non_int_window_errors(clean_df) -> None:
    assert team_efficiency_summary(clean_df, "Boston Celtics", window="5")["status"] == "error"


def test_team_efficiency_summary_uses_per_game_means_not_weighted() -> None:
    res = team_efficiency_summary(make_efficiency_frame(), "Alpha Team", window=None)
    r = res["result"]
    assert res["status"] == "ok"
    # Per-game mean of ortg = 110.0; a possession-weighted aggregate would be 113.33.
    assert r["average_ortg"] == pytest.approx(110.0)
    assert r["average_ortg"] != pytest.approx((100 * 100 + 120 * 200) / 300)
    assert r["average_possessions"] == pytest.approx(150.0)


def test_tools_import_needs_no_registry_parser_llm_formatter() -> None:
    # Importing the helpers must not require any later-phase module.
    import importlib

    assert importlib.import_module("src.tools") is tools_module
