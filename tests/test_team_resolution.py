"""Phase 7B tests: pure team resolver (no pandas, no registry, no data)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.team_resolution import (
    ALIAS_MAP,
    AMBIGUITY_MAP,
    TEAM_AMBIGUOUS,
    TEAM_INVALID_SPECIAL,
    TEAM_RESOLVED,
    TEAM_UNKNOWN,
    TeamResolutionResult,
    normalise_team_text,
    resolve_team_name,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_TEAMS = (
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons",
    "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers",
    "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks",
    "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder",
    "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
    "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards",
)
SPECIAL_TEAMS = ("Team Stars", "Team Stripes", "Team World")

FORBIDDEN_MODULES = (
    "src.query_parser", "src.llm_query_parser", "src.response_formatter", "src.assistant",
)


def _resolve(value: str) -> TeamResolutionResult:
    return resolve_team_name(
        value,
        canonical_teams=CANONICAL_TEAMS,
        special_teams=SPECIAL_TEAMS,
        alias_map=ALIAS_MAP,
        ambiguity_map=AMBIGUITY_MAP,
    )


# --- Normalisation ----------------------------------------------------------

def test_normalisation() -> None:
    assert normalise_team_text("Boston Celtics") == "boston celtics"
    assert normalise_team_text("  Boston   Celtics  ") == "boston celtics"
    assert normalise_team_text("L.A.") == "la"
    assert normalise_team_text("76ers") == "76ers"
    assert normalise_team_text("Philadelphia 76ers") == "philadelphia 76ers"


# --- Exact / normalised resolution ------------------------------------------

@pytest.mark.parametrize("value", ["Boston Celtics", "boston celtics", "Boston  Celtics"])
def test_canonical_resolution(value) -> None:
    res = _resolve(value)
    assert res.status == TEAM_RESOLVED
    assert res.canonical_name == "Boston Celtics"


# --- Alias resolution -------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("gsw", "Golden State Warriors"),
    ("warriors", "Golden State Warriors"),
    ("lal", "Los Angeles Lakers"),
    ("lakers", "Los Angeles Lakers"),
    ("lac", "Los Angeles Clippers"),
    ("sixers", "Philadelphia 76ers"),
    ("76ers", "Philadelphia 76ers"),
    ("blazers", "Portland Trail Blazers"),
    ("wolves", "Minnesota Timberwolves"),
])
def test_alias_resolution(value, expected) -> None:
    res = _resolve(value)
    assert res.status == TEAM_RESOLVED
    assert res.canonical_name == expected


# --- Ambiguity --------------------------------------------------------------

@pytest.mark.parametrize("value,candidates", [
    ("LA", ("Los Angeles Lakers", "Los Angeles Clippers")),
    ("los angeles", ("Los Angeles Lakers", "Los Angeles Clippers")),
    ("NY", ("New York Knicks", "Brooklyn Nets")),
    ("new york", ("New York Knicks", "Brooklyn Nets")),
])
def test_ambiguous_inputs(value, candidates) -> None:
    res = _resolve(value)
    assert res.status == TEAM_AMBIGUOUS
    assert res.canonical_name is None
    assert res.suggestions == candidates


def test_ambiguous_full_names_still_resolve() -> None:
    assert _resolve("New York Knicks").canonical_name == "New York Knicks"
    assert _resolve("Brooklyn Nets").canonical_name == "Brooklyn Nets"


# --- Unknown / fuzzy --------------------------------------------------------

def test_typo_is_unknown_with_suggestion_not_resolved() -> None:
    res = _resolve("Celics")
    assert res.status == TEAM_UNKNOWN
    assert res.canonical_name is None
    assert "Boston Celtics" in res.suggestions


def test_unknown_suggestions_are_deterministic_and_capped() -> None:
    a = _resolve("Celics")
    b = _resolve("Celics")
    assert a.suggestions == b.suggestions
    capped = resolve_team_name(
        "xyz", canonical_teams=CANONICAL_TEAMS, special_teams=SPECIAL_TEAMS,
        alias_map=ALIAS_MAP, ambiguity_map=AMBIGUITY_MAP, max_suggestions=2,
    )
    assert len(capped.suggestions) <= 2


def test_completely_unknown_team() -> None:
    res = _resolve("Definitely Not A Team Zzz")
    assert res.status == TEAM_UNKNOWN


# --- Special teams ----------------------------------------------------------

@pytest.mark.parametrize("value", ["Team Stars", "Team Stripes", "Team World"])
def test_special_team_rejected(value) -> None:
    res = _resolve(value)
    assert res.status == TEAM_INVALID_SPECIAL
    assert res.canonical_name is None


# --- Alias safety -----------------------------------------------------------

def test_alias_targets_are_canonical_and_not_special() -> None:
    canonical_set, special_set = set(CANONICAL_TEAMS), set(SPECIAL_TEAMS)
    for key, target in ALIAS_MAP.items():
        assert target in canonical_set, f"{key} -> {target} not canonical"
        assert target not in special_set


def test_alias_and_ambiguity_keys_disjoint_and_normalised() -> None:
    assert set(ALIAS_MAP).isdisjoint(set(AMBIGUITY_MAP))
    for key in ALIAS_MAP:
        assert normalise_team_text(key) == key
    for key in AMBIGUITY_MAP:
        assert normalise_team_text(key) == key


def test_ambiguity_candidates_are_canonical() -> None:
    canonical_set = set(CANONICAL_TEAMS)
    for candidates in AMBIGUITY_MAP.values():
        for candidate in candidates:
            assert candidate in canonical_set


# --- JSON serialisation -----------------------------------------------------

@pytest.mark.parametrize("value", ["Boston Celtics", "gsw", "LA", "Celics", "Team World"])
def test_result_to_dict_json_serialisable(value) -> None:
    json.dumps(_resolve(value).to_dict())


# --- TeamResolutionResult direct-construction validation --------------------

def test_team_resolution_result_rejects_invalid_status() -> None:
    with pytest.raises((TypeError, ValueError)):
        TeamResolutionResult(status="weird", input_value="x")


def test_team_resolution_result_rejects_non_string_input_value() -> None:
    with pytest.raises((TypeError, ValueError)):
        TeamResolutionResult(status=TEAM_RESOLVED, input_value=123)


def test_team_resolution_result_rejects_non_string_suggestions() -> None:
    with pytest.raises((TypeError, ValueError)):
        TeamResolutionResult(status=TEAM_UNKNOWN, input_value="x", suggestions=("Boston Celtics", 123))


# --- Import safety ----------------------------------------------------------

def test_forbidden_modules_absent() -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


def test_team_resolution_import_is_lightweight() -> None:
    code = (
        "import sys; import src.team_resolution;"
        "assert 'pandas' not in sys.modules, 'pandas imported';"
        "assert 'src.tool_registry' not in sys.modules, 'registry imported';"
        "assert 'src.tools' not in sys.modules, 'tools imported';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
