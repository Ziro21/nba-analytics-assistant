"""Phase 5A tests: tool result contract and shared dataframe helpers only.

No analytical tool outputs are tested here (those arrive in Phases 5B–5G). Integration
tests build the real clean frame through the real pipeline. No network, no LLM.
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
    date_range_for,
    filter_franchise_games,
    filter_team_games,
)

META_KEYS = {"team", "games_used", "date_range", "window_requested", "season_id"}
TOP_LEVEL_KEYS = {"status", "tool", "result", "meta", "warnings"}

ANALYTICAL_TOOL_NAMES = (
    "team_average_points",
    "average_points_allowed",
    "team_record",
    "top_scoring_teams",
    "head_to_head",
    "team_efficiency_summary",
)


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return clean


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

def test_tools_module_exposes_no_analytical_tools_yet() -> None:
    for name in ANALYTICAL_TOOL_NAMES:
        assert not hasattr(tools_module, name), f"{name} should not exist yet"


def test_tools_import_needs_no_registry_parser_llm_formatter() -> None:
    # Importing the helpers must not require any later-phase module.
    import importlib

    assert importlib.import_module("src.tools") is tools_module
