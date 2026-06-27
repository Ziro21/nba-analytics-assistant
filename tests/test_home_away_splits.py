"""Home/away contextual splits across the five single-team tools, end to end.

Oracles are recomputed from the dataset (compare-against-pandas), never hardcoded. Location filters
to ``is_home`` (1=home, 0=away) BEFORE any window, so "last N home games" means the last N home games.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from src.assistant import answer_query
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.intent_types import ParsedIntent
from src.intent_validator import validate_intent
from src.response_formatter import format_tool_result
from src.tool_registry import DEFAULT_REGISTRY
from src.tools import (
    average_points_allowed,
    filter_team_games,
    team_advanced_profile,
    team_average_points,
    team_efficiency_summary,
    team_record,
)
from src.validation_context import build_validation_context

WARRIORS = "Golden State Warriors"
CELTICS = "Boston Celtics"
LAKERS = "Los Angeles Lakers"
SINGLE_TEAM_TOOLS = ("team_average_points", "average_points_allowed", "team_record",
                     "team_efficiency_summary", "team_advanced_profile")


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


def _expected(clean_df, team, location, window):
    games = filter_team_games(clean_df, team)
    if location == "home":
        games = games[games["is_home"] == 1]
    elif location == "away":
        games = games[games["is_home"] == 0]
    return games if window is None else games.tail(window)


# --- Tool layer: values match an independent pandas recomputation -----------

@pytest.mark.parametrize("location", ["home", "away"])
@pytest.mark.parametrize("window", [None, 10])
def test_average_points_location(clean_df, location, window) -> None:
    res = team_average_points(clean_df, WARRIORS, window=window, location=location)
    exp = _expected(clean_df, WARRIORS, location, window)
    assert res["status"] == "ok"
    assert res["meta"]["location"] == location
    assert res["result"]["games_used"] == len(exp)
    assert res["result"]["average_points"] == pytest.approx(exp["points_for"].mean())


@pytest.mark.parametrize("location", ["home", "away"])
def test_points_allowed_location(clean_df, location) -> None:
    res = average_points_allowed(clean_df, LAKERS, window=10, location=location)
    exp = _expected(clean_df, LAKERS, location, 10)
    assert res["status"] == "ok" and res["meta"]["location"] == location
    assert res["result"]["average_points_allowed"] == pytest.approx(exp["points_against"].mean())


@pytest.mark.parametrize("location", ["home", "away"])
def test_record_location(clean_df, location) -> None:
    res = team_record(clean_df, WARRIORS, location=location)
    exp = _expected(clean_df, WARRIORS, location, None)
    wins = int(exp["win_flag"].sum())
    assert res["result"]["record"] == f"{wins}-{len(exp) - wins}"
    assert res["meta"]["location"] == location


@pytest.mark.parametrize("location", ["home", "away"])
def test_efficiency_location(clean_df, location) -> None:
    res = team_efficiency_summary(clean_df, CELTICS, window=10, location=location)
    exp = _expected(clean_df, CELTICS, location, 10)
    assert res["result"]["average_ortg"] == pytest.approx(exp["ortg"].mean())
    assert res["result"]["average_drtg"] == pytest.approx(exp["drtg"].mean())
    assert res["meta"]["location"] == location


@pytest.mark.parametrize("location", ["home", "away"])
def test_advanced_profile_location(clean_df, location) -> None:
    res = team_advanced_profile(clean_df, WARRIORS, window=5, location=location)
    exp = _expected(clean_df, WARRIORS, location, 5)  # last 5 home/away games
    assert res["result"]["games_used"] == len(exp) == 5
    assert res["result"]["average_net_rating"] == pytest.approx(exp["net_rating"].mean())
    assert res["meta"]["location"] == location


def test_home_plus_away_partition_all_games(clean_df) -> None:
    # home and away record must partition the all-games record exactly.
    home = team_record(clean_df, WARRIORS, location="home")["result"]
    away = team_record(clean_df, WARRIORS, location="away")["result"]
    allg = team_record(clean_df, WARRIORS)["result"]
    assert home["games_used"] + away["games_used"] == allg["games_used"]
    assert home["wins"] + away["wins"] == allg["wins"]


def test_no_location_behaviour_unchanged(clean_df) -> None:
    # location=None is identical to omitting it, and meta location is None (no oracle drift).
    assert team_record(clean_df, WARRIORS) == team_record(clean_df, WARRIORS, location=None)
    assert team_record(clean_df, WARRIORS)["meta"]["location"] is None
    assert team_average_points(clean_df, WARRIORS, window=5)["result"]["average_points"] == pytest.approx(114.4)


@pytest.mark.parametrize("tool", [team_average_points, team_record, team_efficiency_summary,
                                  team_advanced_profile])
def test_invalid_location_is_tool_error(clean_df, tool) -> None:
    assert tool(clean_df, WARRIORS, location="sideways")["status"] == "error"


def test_location_does_not_mutate_input(clean_df) -> None:
    before = clean_df.copy()
    for loc in ("home", "away"):
        team_advanced_profile(clean_df, WARRIORS, window=5, location=loc)
    pd.testing.assert_frame_equal(clean_df, before)


# --- Registry schemas -------------------------------------------------------

@pytest.mark.parametrize("name", SINGLE_TEAM_TOOLS)
def test_single_team_schemas_include_location(name) -> None:
    params = {p["name"]: p for p in DEFAULT_REGISTRY.schema(name)["parameters"]}
    assert "location" in params
    assert params["location"]["type"] == "str|null" and params["location"]["required"] is False


@pytest.mark.parametrize("name", ["top_scoring_teams", "head_to_head"])
def test_unsupported_tool_schemas_exclude_location(name) -> None:
    params = {p["name"] for p in DEFAULT_REGISTRY.schema(name)["parameters"]}
    assert "location" not in params


@pytest.mark.parametrize("location", ["home", "away"])
def test_registry_execution_with_location(clean_df, location) -> None:
    res = DEFAULT_REGISTRY.execute("team_record", {"team": WARRIORS, "location": location},
                                   clean_df=clean_df)
    assert res["status"] == "ok" and res["meta"]["location"] == location


# --- Validator --------------------------------------------------------------

def _parsed(tool, arguments):
    return ParsedIntent(tool_name=tool, arguments=arguments, parser_mode="rule", raw_query="x")


@pytest.mark.parametrize("location", ["home", "away", None])
def test_validator_accepts_valid_location(context, location) -> None:
    args = {"team": "Warriors"} | ({"location": location} if location is not None else {})
    assert validate_intent(_parsed("team_record", args), context=context).is_valid


def test_validator_rejects_invalid_location_string(context) -> None:
    v = validate_intent(_parsed("team_record", {"team": "Warriors", "location": "sideways"}),
                        context=context)
    assert not v.is_valid and any(e.code == "invalid_location" for e in v.errors)


@pytest.mark.parametrize("bad", [True, 1, ["home"], {"x": 1}])
def test_validator_rejects_non_string_location(context, bad) -> None:
    v = validate_intent(_parsed("team_record", {"team": "Warriors", "location": bad}),
                        context=context)
    assert not v.is_valid and any(e.code == "invalid_argument_type" for e in v.errors)


@pytest.mark.parametrize("tool,args", [
    ("top_scoring_teams", {"n": 5, "location": "home"}),
    ("head_to_head", {"team_a": "Boston Celtics", "team_b": "Miami Heat", "location": "home"}),
])
def test_validator_rejects_location_for_unsupported_tools(context, tool, args) -> None:
    v = validate_intent(_parsed(tool, args), context=context)
    assert not v.is_valid and any(e.code == "unexpected_argument" and e.field == "location"
                                  for e in v.errors)


# --- Parser routing + assistant integration ---------------------------------

@pytest.mark.parametrize("query,tool,loc", [
    ("What is the Warriors home record?", "team_record", "home"),
    ("What is the Warriors away record?", "team_record", "away"),
    ("How many points do the Warriors average at home?", "team_average_points", "home"),
    ("How many points do the Warriors average away?", "team_average_points", "away"),
    ("How many points do the Celtics score on the road over the last 5 games?",
     "team_average_points", "away"),
    ("How many points do the Lakers allow at home over the last 10 games?",
     "average_points_allowed", "home"),
    ("Warriors efficiency at home last 10 games", "team_efficiency_summary", "home"),
    ("How are the Warriors performing away over the last 5 games?", "team_advanced_profile", "away"),
    ("Give me the Celtics home advanced profile over the last 10 games.",
     "team_advanced_profile", "home"),
])
def test_location_queries_answer(clean_df, context, query, tool, loc) -> None:
    r = _ask(query, clean_df, context)
    assert r.status == "answer" and r.tool_name == tool
    assert r.meta.get("location") == loc
    assert f"{loc} games" in r.message            # location named naturally in the answer
    json.dumps(r.to_dict())


def test_simple_queries_unchanged(clean_df, context) -> None:
    avg = _ask("How many points do the Warriors average over the last 5 games?", clean_df, context)
    assert avg.tool_name == "team_average_points" and avg.meta.get("location") is None
    assert avg.message == "Golden State Warriors averaged 114.4 points over the last 5 games."


@pytest.mark.parametrize("query", [
    "Are the Warriors good at home?",
    "Should I bet on the Warriors at home?",
    "Will the Warriors win at home?",
])
def test_subjective_and_betting_with_location_unsupported(clean_df, context, query) -> None:
    r = _ask(query, clean_df, context)
    assert r.status == "unsupported" and r.data is None


@pytest.mark.parametrize("query", ["Top 5 scoring teams at home", "Celtics vs Heat at home"])
def test_location_on_unsupported_tool_fails_safely(clean_df, context, query) -> None:
    r = _ask(query, clean_df, context)
    assert r.status == "clarification_needed"
    assert r.data is None
    assert "single-team" in r.message              # explains location is single-team only


def test_ambiguous_team_with_location_clarifies(clean_df, context) -> None:
    r = _ask("How many points do LA average at home?", clean_df, context)
    assert r.status == "clarification_needed"
    assert any(i.code == "ambiguous_team" for i in r.errors) and r.data is None


def test_typo_team_with_location_does_not_execute(clean_df, context) -> None:
    r = _ask("How many points do Celics average at home?", clean_df, context)
    assert r.status == "clarification_needed"
    assert any(i.code == "unknown_team" for i in r.errors) and r.data is None


# --- Formatter --------------------------------------------------------------

def test_formatter_home_record_wording(clean_df) -> None:
    out = format_tool_result(team_record(clean_df, WARRIORS, location="home"), query="x")
    assert out.status == "answer"
    assert "home games" in out.message and "Tool used" not in out.message


def test_formatter_away_window_wording(clean_df) -> None:
    out = format_tool_result(team_average_points(clean_df, WARRIORS, window=5, location="away"),
                             query="x")
    assert "over the last 5 away games" in out.message


def test_formatter_all_games_location_wording(clean_df) -> None:
    out = format_tool_result(team_average_points(clean_df, WARRIORS, location="home"), query="x")
    assert "across all available home games" in out.message


def test_formatter_json_preserves_location(clean_df) -> None:
    out = format_tool_result(team_advanced_profile(clean_df, WARRIORS, window=5, location="home"),
                             query="x")
    payload = out.to_dict()
    assert payload["meta"]["location"] == "home"
    json.dumps(payload)


# --- unsupported venue modifiers must NOT be silently ignored ---------------

@pytest.mark.parametrize("query", [
    "Warriors record neutral site",
    "Warriors record on neutral court",
    "How many points do the Warriors average at a neutral site?",
])
def test_unsupported_venue_modifier_is_not_ignored(clean_df, context, query) -> None:
    # the assistant must NOT quietly answer the all-games question for a venue it cannot honour.
    r = _ask(query, clean_df, context)
    assert r.status == "clarification_needed"
    assert r.data is None                                   # no execution
    assert "home" in r.message.lower() and "away" in r.message.lower()
    assert "289-223" not in r.message                       # not the all-games record


def test_neutral_modifier_with_typo_team_does_not_execute(clean_df, context) -> None:
    r = _ask("Celics record neutral site", clean_df, context)
    assert r.status == "clarification_needed" and r.data is None


def test_parser_location_policy_only_accepts_home_away_road() -> None:
    from src.rule_query_normalisation import normalise_query_text as n
    from src.rule_slot_extractor import _extract_location
    assert _extract_location(n("warriors home record")) == "home"
    assert _extract_location(n("warriors record at home")) == "home"
    assert _extract_location(n("warriors away record")) == "away"
    assert _extract_location(n("warriors record on the road")) == "away"
    assert _extract_location(n("warriors record")) is None
    # an unsupported venue is surfaced as a raw, invalid location so the validator rejects it.
    assert _extract_location(n("warriors record neutral site")) == "neutral"
    assert _extract_location(n("warriors record on neutral court")) == "neutral"


# --- real CLI regression (subprocess) ---------------------------------------

def _cli(*args):
    return subprocess.run([sys.executable, "-m", "src.cli", *args],
                          capture_output=True, text=True, cwd=str(REPO_ROOT))


def test_cli_neutral_site_does_not_answer_all_games() -> None:
    res = _cli("Warriors record neutral site")
    assert res.returncode == 1                          # clarification, not an answer
    assert "289-223" not in res.stdout                  # NOT the all-games record
    assert "home" in res.stdout.lower() and "away" in res.stdout.lower()
    assert "Traceback" not in res.stderr


def test_cli_home_record_answers_with_location() -> None:
    res = _cli("Warriors home record")
    assert res.returncode == 0 and "home games" in res.stdout
