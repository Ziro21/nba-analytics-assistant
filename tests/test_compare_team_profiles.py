"""Team-to-Team Performance Comparison (compare_team_profiles).

Tool logic, two-team validation, parser routing + an explicit routing-regression guard, formatter
wording, and end-to-end coverage. Expected values are computed from the dataset (never hardcoded);
the comparison is descriptive only — never predictive, never betting.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.assistant import answer_query
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.intent_types import ParsedIntent, SAME_TEAM_COMPARISON
from src.intent_validator import validate_intent
from src.response_formatter import format_tool_result
from src.rule_intent_router import route_intent
from src.rule_parser import parse_rule_query
from src.tool_registry import DEFAULT_REGISTRY
from src.tools import (
    COMPARE_NEAR_TIE_THRESHOLD,
    _build_comparison,
    compare_team_profiles,
    filter_franchise_games,
    team_advanced_profile,
)
from src.validation_context import build_validation_context

GSW = "Golden State Warriors"
BOS = "Boston Celtics"
LAL = "Los Angeles Lakers"
NYK = "New York Knicks"
MIA = "Miami Heat"


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


# --- tool: structure, reuse, windows, location ------------------------------

def test_compare_all_games_reuses_single_team_profile(clean_df) -> None:
    result = compare_team_profiles(clean_df, GSW, BOS)
    assert result["status"] == "ok" and result["tool"] == "compare_team_profiles"
    res = result["result"]
    assert res["team_a"] == GSW and res["team_b"] == BOS
    assert res["window"] is None and res["location"] is None
    # each side equals that team's standalone advanced profile (same numbers, by construction).
    profile = team_advanced_profile(clean_df, GSW)["result"]
    assert res["team_a_profile"]["average_net_rating"] == profile["average_net_rating"]
    assert res["team_a_profile"]["games"] == profile["games_used"]
    assert res["team_a_profile"]["record"] == profile["record"]


def test_compare_with_window(clean_df) -> None:
    res = compare_team_profiles(clean_df, GSW, BOS, window=10)["result"]
    assert res["team_a_profile"]["games"] == 10 and res["team_b_profile"]["games"] == 10


def test_compare_home_and_away(clean_df) -> None:
    home = compare_team_profiles(clean_df, LAL, NYK, location="home")["result"]
    assert home["location"] == "home"
    away = compare_team_profiles(clean_df, BOS, MIA, location="away", window=5)["result"]
    assert away["location"] == "away"
    assert away["team_a_profile"]["games"] <= 5 and away["team_b_profile"]["games"] <= 5


def test_location_applied_before_window(clean_df) -> None:
    # away first, then last-5 per team — identical to the single-team profile with the same args.
    res = compare_team_profiles(clean_df, BOS, MIA, location="away", window=5)["result"]
    expected = team_advanced_profile(clean_df, BOS, 5, "away")["result"]
    assert res["team_a_profile"]["average_net_rating"] == expected["average_net_rating"]
    assert res["team_a_profile"]["games"] == expected["games_used"]


def test_values_match_independent_pandas(clean_df) -> None:
    franchise = filter_franchise_games(clean_df)
    last10 = franchise[franchise["team_name"] == GSW].tail(10)
    res = compare_team_profiles(clean_df, GSW, BOS, window=10)["result"]
    assert res["team_a_profile"]["average_net_rating"] == float(last10["net_rating"].mean())
    assert res["team_a_profile"]["average_points_for"] == float(last10["points_for"].mean())


# --- tool: comparison block, near-tie, ordering -----------------------------

def test_comparison_block_has_all_fields(clean_df) -> None:
    comp = compare_team_profiles(clean_df, GSW, BOS, window=10)["result"]["comparison"]
    for key in ("higher_net_rating_team", "higher_points_for_team", "lower_points_against_team",
                "better_record_team", "net_rating_difference", "points_for_difference",
                "points_against_difference", "is_near_tie", "near_tie_threshold",
                "stronger_profile_team", "profile_strength_summary"):
        assert key in comp
    assert comp["net_rating_difference"] >= 0 and comp["near_tie_threshold"] == COMPARE_NEAR_TIE_THRESHOLD


def _profile(team, net, pf, pa, wp):
    return {"team": team, "average_net_rating": net, "average_points_for": pf,
            "average_points_against": pa, "win_pct": wp}


def test_clear_winner_and_near_tie_logic() -> None:
    clear = _build_comparison(_profile("A", 5.0, 110.0, 100.0, 0.6),
                              _profile("B", 1.0, 108.0, 105.0, 0.5))
    assert clear["is_near_tie"] is False and clear["stronger_profile_team"] == "A"
    assert "stronger profile" in clear["profile_strength_summary"]

    near = _build_comparison(_profile("A", 2.4, 110.0, 100.0, 0.9),
                             _profile("B", 2.1, 108.0, 105.0, 0.1))
    assert near["is_near_tie"] is True and near["stronger_profile_team"] is None
    assert near["higher_net_rating_team"] == "A"            # fact still recorded
    assert near["better_record_team"] == "A"                # but record is NOT a tiebreaker
    assert "similar" in near["profile_strength_summary"]


def test_order_stability(clean_df) -> None:
    ab = compare_team_profiles(clean_df, GSW, BOS, window=10)["result"]
    ba = compare_team_profiles(clean_df, BOS, GSW, window=10)["result"]
    assert ab["comparison"] == ba["comparison"]            # verdict identical regardless of order
    assert ab["team_a_profile"] == ba["team_b_profile"]
    assert ab["team_b_profile"] == ba["team_a_profile"]


def test_asymmetric_sample_sizes_reported_per_team(clean_df) -> None:
    res = compare_team_profiles(clean_df, LAL, NYK, location="home")["result"]
    expected_a = team_advanced_profile(clean_df, LAL, None, "home")["result"]["games_used"]
    expected_b = team_advanced_profile(clean_df, NYK, None, "home")["result"]["games_used"]
    assert res["team_a_profile"]["games"] == expected_a
    assert res["team_b_profile"]["games"] == expected_b


# --- tool: safety (over-large window, no data, no mutation, same team) -------

def test_over_large_window_uses_all_with_warning(clean_df) -> None:
    result = compare_team_profiles(clean_df, GSW, BOS, window=100_000)
    assert result["status"] == "ok"
    assert any("using all" in w for w in result["warnings"])


def test_no_data_when_a_team_has_no_qualifying_games(clean_df) -> None:
    # defensive path: a name with no rows yields no_data (never a 0-0 comparison against an empty side).
    result = compare_team_profiles(clean_df, GSW, "Nonexistent Team")
    assert result["status"] == "no_data"
    assert "Nonexistent Team" in result["result"]["teams_without_data"]
    assert "team_a_profile" not in result["result"]


def test_same_resolved_team_is_an_error(clean_df) -> None:
    assert compare_team_profiles(clean_df, BOS, BOS)["status"] == "error"


def test_does_not_mutate_input(clean_df) -> None:
    before = clean_df.copy()
    compare_team_profiles(clean_df, GSW, BOS, window=5, location="home")
    pd.testing.assert_frame_equal(clean_df, before)


# --- validation (two-team) --------------------------------------------------

def _intent(args):
    return ParsedIntent(tool_name="compare_team_profiles", arguments=args, parser_mode="rule")


def test_validation_two_valid_teams(context) -> None:
    res = validate_intent(_intent({"team_a": "Warriors", "team_b": "Celtics"}), context=context)
    assert res.is_valid
    assert res.validated_intent.arguments["team_a"] == GSW
    assert res.validated_intent.arguments["team_b"] == BOS


def test_validation_same_team_after_resolution_rejected(context) -> None:
    # different surfaces, same franchise -> rejected with the comparison-specific code.
    res = validate_intent(_intent({"team_a": "GSW", "team_b": "Warriors"}), context=context)
    assert not res.is_valid
    assert any(e.code == SAME_TEAM_COMPARISON for e in res.errors)


@pytest.mark.parametrize("args,code", [
    ({"team_a": "LA", "team_b": "Celtics"}, "ambiguous_team"),
    ({"team_a": "Celics", "team_b": "Warriors"}, "unknown_team"),
    ({"team_a": "Team World", "team_b": "Warriors"}, "invalid_special_team"),
    ({"team_a": "Warriors", "team_b": "Celtics", "window": -1}, "invalid_window"),
    ({"team_a": "Warriors", "team_b": "Celtics", "window": True}, "invalid_argument_type"),
    ({"team_a": "Warriors", "team_b": "Celtics", "location": "neutral"}, "invalid_location"),
    ({"team_a": "Warriors", "team_b": "Celtics", "location": 1}, "invalid_argument_type"),
    ({"team_a": "Warriors"}, "missing_required_argument"),
    ({"team_a": "Warriors", "team_b": "Celtics", "extra": 1}, "unexpected_argument"),
])
def test_validation_failures(args, code, context) -> None:
    res = validate_intent(_intent(args), context=context)
    assert not res.is_valid and any(e.code == code for e in res.errors)


# --- parser routing + explicit ROUTING-REGRESSION guard ---------------------

@pytest.mark.parametrize("query,args", [
    ("Compare Warriors and Celtics over the last 10 games.", {"team_a": "Warriors", "team_b": "Celtics", "window": 10}),
    ("How do Warriors and Celtics compare over the last 10 games?", {"team_a": "Warriors", "team_b": "Celtics", "window": 10}),
    ("Give me a comparison between Lakers and Bucks.", {"team_a": "Lakers", "team_b": "Bucks"}),
    ("Give me a profile comparison between Lakers and Bucks.", {"team_a": "Lakers", "team_b": "Bucks"}),
    ("Compare Lakers and Knicks at home.", {"team_a": "Lakers", "team_b": "Knicks", "location": "home"}),
    ("Compare the Celtics and Heat away over the last 5 games.", {"team_a": "Celtics", "team_b": "Heat", "location": "away", "window": 5}),
    ("Compare Warriors with Celtics last 10.", {"team_a": "Warriors", "team_b": "Celtics", "window": 10}),
])
def test_supported_comparison_queries_parse(query, args) -> None:
    res = parse_rule_query(query)
    assert res.status == "parsed" and res.parsed_intent.tool_name == "compare_team_profiles"
    assert dict(res.parsed_intent.arguments) == args


# The single intentional routing change in this phase: previously ambiguous, now a comparison.
def test_intentional_change_compare_a_and_b_now_parses() -> None:
    res = parse_rule_query("Compare Lakers and Celtics")
    assert res.status == "parsed" and res.parsed_intent.tool_name == "compare_team_profiles"


@pytest.mark.parametrize("query,tool", [
    ("Compare Warriors and Celtics last 10", "compare_team_profiles"),
    ("Compare Warriors with Celtics last 10.", "compare_team_profiles"),
    ("Lakers record last 10", "team_record"),                # the fix is uniform across window tools
])
def test_bare_last_n_reads_as_window_not_all_games(query, tool) -> None:
    # a numbered "last N" (no 'games') is a 10-game window, never silently dropped to all-games.
    res = parse_rule_query(query)
    assert res.status == "parsed" and res.parsed_intent.tool_name == tool
    assert dict(res.parsed_intent.arguments).get("window") == 10


@pytest.mark.parametrize("query,tool", [
    ("Warriors vs Celtics", "head_to_head"),                 # bare vs preserved
    ("Warriors vs Celtics record", "head_to_head"),
    ("Celtics vs Heat head to head", "head_to_head"),
    ("Warriors vs Celtics head to head", "head_to_head"),
    ("Compare Warriors vs Celtics", "head_to_head"),         # 'compare ... vs' not forced to compare
    ("Compare Lakers vs Knicks at home", "head_to_head"),
    ("What is the Warriors record?", "team_record"),
    ("Compare the Warriors record", "team_record"),          # single-team metric preserved
    ("How are the Warriors performing last 10?", "team_advanced_profile"),
    ("Compare the Warriors offense and defense", "team_advanced_profile"),
    ("Top 5 scoring teams", "top_scoring_teams"),
])
def test_routing_regression_preserved_behaviours(query, tool) -> None:
    res = route_intent(query)
    assert res.status == "routed" and res.tool_name == tool


@pytest.mark.parametrize("query", [
    "Who will win, Warriors or Celtics?",
    "Should I bet on Warriors or Celtics?",
    "Who is better, Warriors or Celtics?",
    "Are Warriors better than Celtics?",
])
def test_predictive_and_betting_stay_unsupported(query) -> None:
    assert route_intent(query).status == "no_route"


# --- formatter wording ------------------------------------------------------

def _message(clean_df, team_a, team_b, **kw) -> str:
    result = compare_team_profiles(clean_df, team_a, team_b, **kw)
    return format_tool_result(result, query="q").message


def test_formatter_standard_wording(clean_df) -> None:
    msg = _message(clean_df, GSW, BOS, window=10)
    assert "Over the last 10 games" in msg
    assert GSW in msg and BOS in msg
    assert "net rating" in msg and "selected sample" in msg
    assert "based on net rating" in msg
    for banned in ("will win", "bet", "should", "Tool used:", "team_a:"):
        assert banned not in msg


def test_formatter_home_and_away_and_all_games(clean_df) -> None:
    assert "At home over" in _message(clean_df, LAL, NYK, location="home", window=10) or \
           "home games" in _message(clean_df, LAL, NYK, location="home", window=10)
    assert "Away over" in _message(clean_df, BOS, MIA, location="away", window=5)
    assert "Across all available games" in _message(clean_df, GSW, BOS)


def test_formatter_near_tie_neutral_wording() -> None:
    near = {"team_a": GSW, "team_b": BOS, "window": None, "location": None,
            "team_a_profile": _profile(GSW, 2.2, 112.0, 110.0, 0.55) | {"games": 10, "record": "6-4"},
            "team_b_profile": _profile(BOS, 2.0, 111.0, 109.0, 0.5) | {"games": 10, "record": "5-5"},
            "comparison": _build_comparison(
                _profile(GSW, 2.2, 112.0, 110.0, 0.55), _profile(BOS, 2.0, 111.0, 109.0, 0.5))}
    from src.tool_results import ok_result, build_meta
    result = ok_result("compare_team_profiles", near, meta=build_meta(games_used=20))
    msg = format_tool_result(result, query="q").message
    assert "similar overall profiles" in msg and "stronger profile" not in msg


# --- end to end (runtime + assistant) ---------------------------------------

def _answer(query, clean_df, context):
    return answer_query(query, clean_df=clean_df, validation_context=context,
                        registry=DEFAULT_REGISTRY)


def test_e2e_comparison_answers(clean_df, context) -> None:
    res = _answer("Compare Warriors and Celtics over the last 10 games.", clean_df, context)
    assert res.status == "answer" and res.tool_name == "compare_team_profiles"
    assert res.data["comparison"]["stronger_profile_team"] in (GSW, BOS, None)
    json.dumps(res.to_dict())


def test_e2e_home_comparison_answers(clean_df, context) -> None:
    res = _answer("Compare Lakers and Knicks at home.", clean_df, context)
    assert res.status == "answer" and res.data["location"] == "home"


def test_location_rejection_message_mentions_comparison(clean_df, context) -> None:
    # 'compare X vs Y at home' routes to head_to_head (vs preserved); location is rejected there,
    # but the message now tells the user comparisons DO support home/away.
    res = _answer("Compare Lakers vs Knicks at home", clean_df, context)
    assert res.status == "clarification_needed" and "comparison" in res.message.lower()


def test_release_and_submission_docs_list_comparison() -> None:
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for doc in ("README.md", "SUBMISSION.md", "RELEASE_NOTES.md", "docs/usage_examples.md"):
        text = (root / doc).read_text().lower()
        assert "comparison" in text or "compare_team_profiles" in text, doc


@pytest.mark.parametrize("query,status", [
    ("Compare Celtics and Celtics.", "clarification_needed"),    # same team
    ("Compare GSW and Warriors.", "clarification_needed"),       # same resolved team
    ("Compare LA and Celtics.", "clarification_needed"),         # ambiguous team
    ("Compare Celics and Warriors.", "clarification_needed"),    # typo -> suggestion
    ("Compare Team World and Warriors.", "clarification_needed"),  # exhibition team
    ("Warriors vs Celtics.", "answer"),                          # preserved head_to_head
    ("Should I bet on Warriors or Celtics?", "unsupported"),
    ("Who is better, Warriors or Celtics?", "unsupported"),
])
def test_e2e_safety_and_preserved_paths(query, status, clean_df, context) -> None:
    res = _answer(query, clean_df, context)
    assert res.status == status
    if status != "answer":
        assert res.data is None
