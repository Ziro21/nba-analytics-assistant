"""Phase 5H integration review: all six analytical tools verified together.

This is a consolidation / quality-gate suite, not new tool development. It checks the
shared result contract, consolidated oracles, status semantics, metadata, immutability,
exhibition exclusion, and scope (no premature next-layer systems). No network, no LLM.
"""

from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest

from src.config import SPECIAL_TEAMS
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.tools import (
    average_points_allowed,
    head_to_head,
    team_average_points,
    team_efficiency_summary,
    team_record,
    top_scoring_teams,
)

META_KEYS = {"team", "games_used", "date_range", "window_requested", "season_id"}
TOP_LEVEL_KEYS = {"status", "tool", "result", "meta", "warnings"}
VALID_STATUSES = {"ok", "no_data", "error"}

# Team-level tools share the (clean, team, window) signature.
TEAM_LEVEL_TOOLS = (
    team_average_points,
    average_points_allowed,
    team_record,
    team_efficiency_summary,
)

# Representative successful calls: (expected_tool_name, callable(clean) -> result).
SUCCESS_CALLS = (
    ("team_average_points", lambda c: team_average_points(c, "Golden State Warriors", window=5)),
    ("average_points_allowed", lambda c: average_points_allowed(c, "Golden State Warriors", window=5)),
    ("team_record", lambda c: team_record(c, "Golden State Warriors", window=None)),
    ("top_scoring_teams", lambda c: top_scoring_teams(c, n=5)),
    ("head_to_head", lambda c: head_to_head(c, "Boston Celtics", "Miami Heat")),
    ("team_efficiency_summary", lambda c: team_efficiency_summary(c, "Boston Celtics", window=10)),
)

NEXT_LAYER_MODULES = (
    "src.tool_registry",
    "src.query_parser",
    "src.llm_query_parser",
    "src.intent_validator",
    "src.response_formatter",
    "src.assistant",
)


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return clean


# --- 1 & 9. existence / scope -----------------------------------------------

def test_all_six_tools_callable() -> None:
    for _, call in SUCCESS_CALLS:
        assert callable(call)


def test_no_premature_next_layer_systems() -> None:
    for module in NEXT_LAYER_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


# --- 2. contract consistency ------------------------------------------------

@pytest.mark.parametrize("name,call", SUCCESS_CALLS, ids=[n for n, _ in SUCCESS_CALLS])
def test_result_contract_shape(clean_df, name, call) -> None:
    res = call(clean_df)
    assert isinstance(res, dict)
    assert set(res) == TOP_LEVEL_KEYS
    assert res["status"] in VALID_STATUSES
    assert res["status"] == "ok"
    assert res["tool"] == name
    assert isinstance(res["result"], dict)
    assert set(res["meta"]) == META_KEYS
    assert isinstance(res["warnings"], list)
    json.dumps(res)  # JSON-serialisable


# --- 3. consolidated oracles ------------------------------------------------

def test_all_oracles(clean_df) -> None:
    # team_average_points
    for team, expected in [("Golden State Warriors", 114.4), ("Boston Celtics", 108.6),
                           ("Los Angeles Lakers", 117.2)]:
        r = team_average_points(clean_df, team, window=5)["result"]
        assert round(r["average_points"], 2) == expected

    # average_points_allowed
    r = average_points_allowed(clean_df, "Golden State Warriors", window=5)["result"]
    assert round(r["average_points_allowed"], 2) == 117.0

    # team_record
    for team, record in [("Golden State Warriors", "289-223"), ("Boston Celtics", "359-183"),
                         ("Los Angeles Lakers", "267-228")]:
        assert team_record(clean_df, team, window=None)["result"]["record"] == record

    # top_scoring_teams
    top = top_scoring_teams(clean_df, n=5)["result"]["teams"]
    assert [(t["team"], round(t["average_points"], 2)) for t in top] == [
        ("Atlanta Hawks", 116.13), ("Indiana Pacers", 115.94), ("Milwaukee Bucks", 115.84),
        ("Denver Nuggets", 115.67), ("Utah Jazz", 115.08),
    ]

    # head_to_head (and reverse)
    ab = head_to_head(clean_df, "Boston Celtics", "Miami Heat")["result"]
    ba = head_to_head(clean_df, "Miami Heat", "Boston Celtics")["result"]
    assert (ab["meetings"], ab["record"]) == (39, "25-14")
    assert (ba["meetings"], ba["record"]) == (39, "14-25")

    # team_efficiency_summary
    bos = team_efficiency_summary(clean_df, "Boston Celtics", window=10)["result"]
    gsw = team_efficiency_summary(clean_df, "Golden State Warriors", window=10)["result"]
    assert (round(bos["average_ortg"], 2), round(bos["average_drtg"], 2)) == (106.98, 101.93)
    assert (round(gsw["average_ortg"], 2), round(gsw["average_drtg"], 2)) == (105.57, 109.17)


# --- 4. status semantics ----------------------------------------------------

@pytest.mark.parametrize("tool", TEAM_LEVEL_TOOLS, ids=lambda t: t.__name__)
def test_team_level_status_semantics(clean_df, tool) -> None:
    assert tool(clean_df, "Not A Real Team", 5)["status"] == "no_data"
    assert tool(clean_df, "Not A Real Team", 0)["status"] == "error"
    assert tool(clean_df, "Golden State Warriors", 0)["status"] == "error"


def test_head_to_head_status_semantics(clean_df) -> None:
    assert head_to_head(clean_df, "Not A Real Team", "Miami Heat", 5)["status"] == "no_data"
    assert head_to_head(clean_df, "Not A Real Team", "Miami Heat", 0)["status"] == "error"
    assert head_to_head(clean_df, "Boston Celtics", "Boston Celtics")["status"] == "error"


def test_top_scoring_teams_status_semantics(clean_df) -> None:
    assert top_scoring_teams(clean_df, n=0)["status"] == "error"
    assert top_scoring_teams(clean_df, n=5, season_id="36")["status"] == "error"
    assert top_scoring_teams(clean_df, n=5, season_id=999)["status"] == "no_data"
    big = top_scoring_teams(clean_df, n=10_000)
    assert big["status"] == "ok" and len(big["warnings"]) == 1


# --- 5. warning consistency -------------------------------------------------

@pytest.mark.parametrize("tool", TEAM_LEVEL_TOOLS, ids=lambda t: t.__name__)
def test_team_level_over_large_window_warns(clean_df, tool) -> None:
    res = tool(clean_df, "Golden State Warriors", 10_000)
    assert res["status"] == "ok"
    assert len(res["warnings"]) == 1


# --- 6. metadata ------------------------------------------------------------

def test_metadata_team_field_conventions(clean_df) -> None:
    # team-level tools set meta.team to the team
    assert team_average_points(clean_df, "Golden State Warriors", 5)["meta"]["team"] == "Golden State Warriors"
    # head_to_head sets meta.team to team_a
    assert head_to_head(clean_df, "Boston Celtics", "Miami Heat")["meta"]["team"] == "Boston Celtics"
    # top_scoring_teams has no single team
    assert top_scoring_teams(clean_df, n=5)["meta"]["team"] is None


def test_windowed_date_range_reflects_window_not_full_history(clean_df) -> None:
    full = team_average_points(clean_df, "Golden State Warriors", window=None)["meta"]["date_range"]
    windowed = team_average_points(clean_df, "Golden State Warriors", window=5)["meta"]["date_range"]
    assert windowed != full
    assert windowed[0] >= full[0]  # the windowed start is no earlier than full history


def test_season_id_meta_only_relevant_for_top_scoring(clean_df) -> None:
    assert top_scoring_teams(clean_df, n=5, season_id=34)["meta"]["season_id"] == 34
    assert team_average_points(clean_df, "Golden State Warriors", 5)["meta"]["season_id"] is None


# --- 7. immutability --------------------------------------------------------

@pytest.mark.parametrize("name,call", SUCCESS_CALLS, ids=[n for n, _ in SUCCESS_CALLS])
def test_no_tool_mutates_clean_df(clean_df, name, call) -> None:
    before = clean_df.copy(deep=True)
    call(clean_df)
    assert clean_df.equals(before)


@pytest.mark.parametrize("name,call", SUCCESS_CALLS, ids=[n for n, _ in SUCCESS_CALLS])
def test_successful_standard_calls_have_no_warnings(clean_df, name, call) -> None:
    # Normal (non-over-large) successful calls should not emit warnings.
    res = call(clean_df)
    assert res["status"] == "ok"
    assert res["warnings"] == []


# --- 8. exhibition exclusion ------------------------------------------------

def test_top_scoring_excludes_exhibition_teams(clean_df) -> None:
    teams = {t["team"] for t in top_scoring_teams(clean_df, n=10_000)["result"]["teams"]}
    assert teams.isdisjoint(set(SPECIAL_TEAMS))


@pytest.mark.parametrize("tool", TEAM_LEVEL_TOOLS, ids=lambda t: t.__name__)
@pytest.mark.parametrize("special", list(SPECIAL_TEAMS))
def test_team_level_tools_treat_exhibition_team_as_no_data(clean_df, special, tool) -> None:
    assert tool(clean_df, special, window=5)["status"] == "no_data"
