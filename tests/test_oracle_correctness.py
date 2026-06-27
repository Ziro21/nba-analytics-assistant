"""Oracle / correctness tests — independent validation that the numbers are RIGHT.

These deliberately do NOT trust ``src.tools``. They recompute answers a second, independent way and
compare:

  * Layer 1 — clean-view *derivations* are checked against the **raw CSV** (opponent pairing, net
    rating, points, win flag, exhibition flagging, date ordering).
  * Layer 2 — *tool outputs* are checked against vanilla-pandas aggregation written here.
  * End to end — the user-facing message reports the oracle numbers.

This catches definitional, derivation, windowing, exclusion, and double-counting errors that tests
built on the same helpers structurally cannot. The dataset is static and self-contained (synthetic
season IDs, built-in exhibition teams), so the dataset itself is the ground truth the assistant must
report faithfully — there is no external real-world source to (or that should) be cross-referenced.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assistant_runtime import build_default_runtime
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view
from src.tool_registry import DEFAULT_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent.parent
EXHIBITION_TEAMS = {"Team Stars", "Team Stripes", "Team World"}


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return build_clean_view(load_raw_dataset())


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "data" / "nba_dataset.csv")


@pytest.fixture(scope="module")
def franchise(clean) -> pd.DataFrame:
    # independent replication of franchise filtering (exclude exhibition team OR exhibition opponent)
    return clean[~(clean["is_exhibition"] | clean["opponent_is_exhibition"])]


@pytest.fixture(scope="module")
def runtime():
    return build_default_runtime()


# --- Layer 1: clean-view derivations vs the raw CSV -------------------------

def test_derivations_match_raw_csv(clean, raw) -> None:
    # raw `possessions` is renamed to avoid a name collision with the clean column of the same name.
    raw_cols = raw[["game_id", "team_name", "team_points", "opponent_points",
                    "ORTG", "DRTG", "possessions"]].rename(columns={"possessions": "raw_possessions"})
    merged = clean.merge(raw_cols, on=["game_id", "team_name"])
    assert len(merged) == len(clean)  # (game_id, team_name) is a unique key — no fan-out
    assert (merged["points_for"] == merged["team_points"]).all()
    assert (merged["points_against"] == merged["opponent_points"]).all()
    assert (merged["win_flag"] == (merged["team_points"] > merged["opponent_points"])).all()
    # direct column pass-throughs (rename / carry-over): clean values equal the raw source values.
    assert (merged["ortg"] == merged["ORTG"]).all()
    assert (merged["drtg"] == merged["DRTG"]).all()
    assert (merged["possessions"] == merged["raw_possessions"]).all()
    assert (merged["net_rating"] - (merged["ORTG"] - merged["DRTG"])).abs().max() < 1e-9
    # data-integrity invariant: plus_minus is exactly points scored minus points allowed
    assert (clean["plus_minus"] == (clean["points_for"] - clean["points_against"])).all()


def test_opponent_team_name_matches_independent_game_pairing(clean, raw) -> None:
    opponents: dict[tuple, str] = {}
    for game_id, rows in raw.groupby("game_id"):
        names = list(rows["team_name"])
        if len(names) == 2:
            opponents[(game_id, names[0])] = names[1]
            opponents[(game_id, names[1])] = names[0]
    independent = [opponents.get((gid, team))
                   for gid, team in zip(clean["game_id"], clean["team_name"])]
    assert independent == list(clean["opponent_team_name"])


def test_exhibition_flagging_and_date_ordering(clean) -> None:
    assert set(clean.loc[clean["is_exhibition"], "team_name"].unique()) == EXHIBITION_TEAMS
    # each team's games are date-ascending, so "last N" really is the N most recent games.
    for team in ("Golden State Warriors", "Boston Celtics", "Los Angeles Lakers"):
        assert clean[clean["team_name"] == team]["game_date"].is_monotonic_increasing


# --- Layer 2: tool outputs vs independent pandas aggregation -----------------
# The registry ``execute(...)`` call is the SUBJECT-UNDER-TEST; the from-scratch pandas computation
# beside it is the independent oracle. They must never share logic from ``src.tools``.

def _oracle_slice(franchise, team, window, location):
    games = franchise[franchise["team_name"] == team]
    if location == "home":
        games = games[games["is_home"] == 1]
    elif location == "away":
        games = games[games["is_home"] == 0]
    return games.tail(window) if window else games


@pytest.mark.parametrize("team", ["Golden State Warriors", "Boston Celtics", "Miami Heat"])
@pytest.mark.parametrize("window", [None, 5, 10])
@pytest.mark.parametrize("location", [None, "home", "away"])
def test_record_averages_and_profile_match_oracle(franchise, clean, team, window, location) -> None:
    games = _oracle_slice(franchise, team, window, location)
    count = len(games)
    wins = int((games["points_for"] > games["points_against"]).sum())
    args = {"team": team}
    if window:
        args["window"] = window
    if location:
        args["location"] = location

    record = DEFAULT_REGISTRY.execute("team_record", args, clean_df=clean)["result"]
    assert (record["wins"], record["losses"], record["games_used"]) == (wins, count - wins, count)
    assert record["win_percentage"] == pytest.approx(wins / count)

    scored = DEFAULT_REGISTRY.execute("team_average_points", args, clean_df=clean)["result"]
    assert scored["average_points"] == pytest.approx(float(games["points_for"].mean()))

    allowed = DEFAULT_REGISTRY.execute("average_points_allowed", args, clean_df=clean)["result"]
    assert allowed["average_points_allowed"] == pytest.approx(float(games["points_against"].mean()))

    profile = DEFAULT_REGISTRY.execute("team_advanced_profile", args, clean_df=clean)["result"]
    assert profile["wins"] == wins
    assert profile["average_net_rating"] == pytest.approx(float(games["net_rating"].mean()))
    assert profile["average_ortg"] == pytest.approx(float(games["ortg"].mean()))


@pytest.mark.parametrize("team", ["Golden State Warriors", "Boston Celtics", "Miami Heat"])
@pytest.mark.parametrize("window", [None, 10])
@pytest.mark.parametrize("location", [None, "home", "away"])
def test_efficiency_summary_matches_oracle(franchise, clean, team, window, location) -> None:
    # team_efficiency_summary is the only tool reporting average_possessions — checked directly here
    # (not just via the advanced profile) so every one of its output fields has an independent oracle.
    games = _oracle_slice(franchise, team, window, location)
    args = {"team": team}
    if window:
        args["window"] = window
    if location:
        args["location"] = location
    eff = DEFAULT_REGISTRY.execute("team_efficiency_summary", args, clean_df=clean)["result"]
    assert eff["games_used"] == len(games)
    assert eff["average_ortg"] == pytest.approx(float(games["ortg"].mean()))
    assert eff["average_drtg"] == pytest.approx(float(games["drtg"].mean()))
    assert eff["average_net_rating"] == pytest.approx(float(games["net_rating"].mean()))
    assert eff["average_possessions"] == pytest.approx(float(games["possessions"].mean()))


@pytest.mark.parametrize("n", [5, 10])
def test_top_scoring_ranking_matches_oracle(franchise, clean, n) -> None:
    # independent ranking with the tool's documented tie-break (mean desc, team_name asc).
    oracle = (franchise.groupby("team_name")["points_for"].mean().reset_index()
              .sort_values(["points_for", "team_name"], ascending=[False, True], kind="mergesort"))
    teams = DEFAULT_REGISTRY.execute("top_scoring_teams", {"n": n}, clean_df=clean)["result"]["teams"]
    assert [(t["rank"], t["team"]) for t in teams] == \
        [(i + 1, name) for i, name in enumerate(oracle["team_name"].head(n))]
    means = dict(zip(oracle["team_name"], oracle["points_for"]))
    for t in teams:
        assert t["average_points"] == pytest.approx(float(means[t["team"]]))


@pytest.mark.parametrize("team_a,team_b", [
    ("Boston Celtics", "Miami Heat"),
    ("Golden State Warriors", "Los Angeles Lakers"),
])
def test_head_to_head_counts_each_meeting_once(franchise, clean, team_a, team_b) -> None:
    games = franchise[(franchise["team_name"] == team_a)
                      & (franchise["opponent_team_name"] == team_b)]
    wins = int((games["points_for"] > games["points_against"]).sum())
    result = DEFAULT_REGISTRY.execute("head_to_head", {"team_a": team_a, "team_b": team_b},
                                      clean_df=clean)["result"]
    assert (result["meetings"], result["team_a_wins"], result["team_b_wins"]) == \
        (len(games), wins, len(games) - wins)
    # the meeting count is symmetric from the opponent's perspective (no double counting)
    mirror = franchise[(franchise["team_name"] == team_b)
                       & (franchise["opponent_team_name"] == team_a)]
    assert len(mirror) == len(games)


@pytest.mark.parametrize("team_a,team_b,window,location", [
    ("Golden State Warriors", "Boston Celtics", 10, None),
    ("Los Angeles Lakers", "New York Knicks", 10, "home"),
])
def test_compare_profiles_match_independent_single_team_profiles(
        franchise, clean, team_a, team_b, window, location) -> None:
    args = {"team_a": team_a, "team_b": team_b, "window": window}
    if location:
        args["location"] = location
    result = DEFAULT_REGISTRY.execute("compare_team_profiles", args, clean_df=clean)["result"]
    for team, profile in ((team_a, result["team_a_profile"]), (team_b, result["team_b_profile"])):
        games = _oracle_slice(franchise, team, window, location)
        assert profile["games"] == len(games)
        assert profile["wins"] == int((games["points_for"] > games["points_against"]).sum())
        assert profile["average_net_rating"] == pytest.approx(float(games["net_rating"].mean()))


# --- End to end: the user-facing message reports the oracle numbers ----------

@pytest.mark.parametrize("query,team,location", [
    ("What is the Warriors record?", "Golden State Warriors", None),
    ("What is the Boston Celtics home record?", "Boston Celtics", "home"),
])
def test_user_facing_record_message_reports_oracle_numbers(franchise, runtime, query, team, location) -> None:
    games = _oracle_slice(franchise, team, None, location)
    wins = int((games["points_for"] > games["points_against"]).sum())
    message = runtime.answer(query).message
    assert f"{wins}-{len(games) - wins}" in message
    assert str(len(games)) in message
