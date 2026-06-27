"""Pre-UI core analytics: the team_advanced_profile tool, end to end.

Oracles are computed from the dataset (recompute-and-compare), not hardcoded from any prompt.
The tool composes existing clean-view metrics; it adds no new statistical primitive.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assistant import answer_query
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.response_formatter import format_tool_result
from src.tool_registry import DEFAULT_REGISTRY
from src.tools import filter_team_games, team_advanced_profile
from src.validation_context import build_validation_context

WARRIORS = "Golden State Warriors"
CELTICS = "Boston Celtics"


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return clean


@pytest.fixture(scope="module")
def context(clean_df):
    return build_validation_context(clean_df, registry=DEFAULT_REGISTRY)


def _ask(query, clean_df, context):
    return answer_query(query, clean_df=clean_df, validation_context=context, registry=DEFAULT_REGISTRY)


# --- Tool: values match an independent pandas recomputation -----------------

@pytest.mark.parametrize("team,window", [(WARRIORS, 5), (CELTICS, 10), (WARRIORS, None)])
def test_profile_values_match_independent_pandas(clean_df, team, window) -> None:
    res = team_advanced_profile(clean_df, team, window)
    assert res["status"] == "ok"
    games = filter_team_games(clean_df, team)
    expected = games if window is None else games.tail(window)
    n = len(expected)
    wins = int(expected["win_flag"].sum())
    r = res["result"]
    assert r["games_used"] == n
    assert r["wins"] == wins and r["losses"] == n - wins and r["record"] == f"{wins}-{n - wins}"
    assert r["win_percentage"] == pytest.approx(wins / n)
    assert r["average_points_for"] == pytest.approx(expected["points_for"].mean())
    assert r["average_points_against"] == pytest.approx(expected["points_against"].mean())
    assert r["average_plus_minus"] == pytest.approx(expected["plus_minus"].mean())
    assert r["average_ortg"] == pytest.approx(expected["ortg"].mean())
    assert r["average_drtg"] == pytest.approx(expected["drtg"].mean())
    assert r["average_net_rating"] == pytest.approx(expected["net_rating"].mean())


def test_profile_oracle_warriors_last_5(clean_df) -> None:
    r = team_advanced_profile(clean_df, WARRIORS, 5)["result"]
    assert r["games_used"] == 5
    assert r["wins"] == 2 and r["losses"] == 3 and r["record"] == "2-3"
    assert r["average_points_for"] == pytest.approx(114.4)
    assert r["average_points_against"] == pytest.approx(117.0)


def test_profile_all_games_matches_record_oracle(clean_df) -> None:
    res = team_advanced_profile(clean_df, WARRIORS, None)["result"]
    assert res["games_used"] == len(filter_team_games(clean_df, WARRIORS))
    assert res["record"] == "289-223"  # consistent with the existing team_record oracle


def test_profile_over_large_window_uses_all_with_warning(clean_df) -> None:
    games = filter_team_games(clean_df, WARRIORS)
    res = team_advanced_profile(clean_df, WARRIORS, len(games) + 50)
    assert res["status"] == "ok"
    assert res["result"]["games_used"] == len(games)
    assert res["warnings"]


@pytest.mark.parametrize("bad", [0, -3, True, "5"])
def test_profile_invalid_window_is_error(clean_df, bad) -> None:
    assert team_advanced_profile(clean_df, WARRIORS, bad)["status"] == "error"


def test_profile_unknown_team_is_no_data(clean_df) -> None:
    res = team_advanced_profile(clean_df, "Nonexistent Team", None)
    assert res["status"] == "no_data"
    assert res["result"]["games_used"] == 0


def test_profile_excludes_exhibition_team(clean_df) -> None:
    # exhibition rows are franchise-filtered out, so a special team has no franchise games.
    assert team_advanced_profile(clean_df, "Team World", None)["status"] == "no_data"


def test_profile_does_not_mutate_input(clean_df) -> None:
    before = clean_df.copy()
    team_advanced_profile(clean_df, WARRIORS, 5)
    pd.testing.assert_frame_equal(clean_df, before)


# --- Formatter message (concise; no redundant metadata) ---------------------

def test_profile_message_windowed(clean_df) -> None:
    out = format_tool_result(team_advanced_profile(clean_df, WARRIORS, 5), query="x")
    assert out.status == "answer"
    msg = out.message
    assert msg.startswith("Golden State Warriors over the last 5 games:")
    assert "2-3 record" in msg
    assert "114.4 points scored per game" in msg
    assert "117.0 points allowed" in msg
    assert "ORTG" in msg and "DRTG" in msg and "net rating" in msg
    # the normal answer never shows tool metadata or parsed parameters
    for forbidden in ("Tool used", "team_advanced_profile", "Window:", "Games included"):
        assert forbidden not in msg


def test_profile_message_all_games(clean_df) -> None:
    out = format_tool_result(team_advanced_profile(clean_df, WARRIORS, None), query="x")
    assert "across all available games:" in out.message
    assert "289-223 record" in out.message


def test_profile_no_data_maps_to_clarification(clean_df) -> None:
    out = format_tool_result(team_advanced_profile(clean_df, "Nonexistent Team", None), query="x")
    assert out.status == "clarification_needed"
    assert out.errors and out.errors[0].code == "no_data"


# --- Validation (schema-driven) via the assistant ---------------------------

def test_profile_resolves_nickname(clean_df, context) -> None:
    r = _ask("How are the Warriors performing over the last 5 games?", clean_df, context)
    assert r.status == "answer" and r.tool_name == "team_advanced_profile"


def test_profile_ambiguous_team_clarifies(clean_df, context) -> None:
    r = _ask("How is LA performing over the last 5 games?", clean_df, context)
    assert r.status == "clarification_needed"
    assert any(i.code == "ambiguous_team" for i in r.errors)
    assert r.data is None


def test_profile_unknown_typo_does_not_execute(clean_df, context) -> None:
    r = _ask("How are the Celics performing over the last 5 games?", clean_df, context)
    assert r.status == "clarification_needed"
    assert any(i.code == "unknown_team" for i in r.errors)
    assert "Boston Celtics" in tuple(s for i in r.errors for s in (i.suggestions or ()))
    assert r.data is None


def test_profile_special_team_rejected(clean_df, context) -> None:
    r = _ask("How is Team World performing?", clean_df, context)
    assert r.status == "clarification_needed"
    assert any(i.code == "invalid_special_team" for i in r.errors)
    assert r.data is None


# --- End to end / regression guards -----------------------------------------

def test_profile_full_chain_answer(clean_df, context) -> None:
    r = _ask("Give me the Celtics advanced profile over the last 10 games.", clean_df, context)
    assert r.status == "answer" and r.tool_name == "team_advanced_profile"
    assert r.data is not None and r.meta is not None
    assert "Boston Celtics over the last 10 games:" in r.message
    json.dumps(r.to_dict())  # JSON-safe


def test_simple_queries_are_not_hijacked_by_profile(clean_df, context) -> None:
    assert _ask("How many points do the Warriors average over the last 5 games?",
                clean_df, context).tool_name == "team_average_points"
    assert _ask("How many points do the Warriors allow over the last 5 games?",
                clean_df, context).tool_name == "average_points_allowed"
    assert _ask("What is the Warriors record?", clean_df, context).tool_name == "team_record"
    assert _ask("Warriors efficiency last 10 games", clean_df, context).tool_name == "team_efficiency_summary"


def test_compare_offense_defense_routes_to_profile(clean_df, context) -> None:
    r = _ask("Compare the Warriors offense and defense over the last 5 games.", clean_df, context)
    assert r.status == "answer" and r.tool_name == "team_advanced_profile"


def test_location_profile_is_supported(clean_df, context) -> None:
    # home/away splits are now supported for single-team tools (see test_home_away_splits.py).
    r = _ask("Warriors advanced profile at home", clean_df, context)
    assert r.status == "answer" and r.tool_name == "team_advanced_profile"
    assert r.meta.get("location") == "home"
    assert "home games" in r.message


def test_subjective_and_betting_queries_remain_unsupported(clean_df, context) -> None:
    for query in ("Who is better?", "Should I bet on the Warriors?", "Will the Warriors win?"):
        r = _ask(query, clean_df, context)
        assert r.status == "unsupported"
        assert r.data is None
