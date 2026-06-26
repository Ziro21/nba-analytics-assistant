"""Phase 8C tests: the explicit team-surface catalogue + drift protection."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.team_surface_catalogue import (
    AMBIGUOUS_TEAM_SURFACES,
    CANONICAL_TEAM_NAMES,
    SPECIAL_TEAM_SURFACES,
    TEAM_ALIAS_SURFACES,
    TEAM_SURFACE_FORMS,
    get_team_surface_forms,
    get_team_surface_forms_by_length,
    normalise_surface,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SPECIAL_TEAMS = ("Team Stars", "Team Stripes", "Team World")


# --- catalogue shape --------------------------------------------------------

def test_collections_are_tuples() -> None:
    for collection in (CANONICAL_TEAM_NAMES, TEAM_ALIAS_SURFACES, AMBIGUOUS_TEAM_SURFACES,
                       SPECIAL_TEAM_SURFACES, TEAM_SURFACE_FORMS, get_team_surface_forms(),
                       get_team_surface_forms_by_length()):
        assert isinstance(collection, tuple)


def test_no_empty_or_duplicate_surfaces() -> None:
    for collection in (TEAM_ALIAS_SURFACES, AMBIGUOUS_TEAM_SURFACES, SPECIAL_TEAM_SURFACES,
                       TEAM_SURFACE_FORMS):
        assert all(s and isinstance(s, str) for s in collection)
        assert len(collection) == len(set(collection))
    assert len(CANONICAL_TEAM_NAMES) == len(set(CANONICAL_TEAM_NAMES)) == 30


def test_surfaces_are_normalised_and_preserve_digits() -> None:
    for surface in TEAM_SURFACE_FORMS:
        assert normalise_surface(surface) == surface
    assert "76ers" in TEAM_SURFACE_FORMS
    assert "76ers" in TEAM_ALIAS_SURFACES


def test_special_teams_recognised_as_surfaces_but_never_canonical() -> None:
    # Special teams ARE extractable surfaces (so the validator can reject them cleanly with
    # invalid_special_team), but are never valid franchises.
    for special in SPECIAL_TEAMS:
        assert special not in CANONICAL_TEAM_NAMES
        assert normalise_surface(special) in SPECIAL_TEAM_SURFACES
        assert normalise_surface(special) in TEAM_SURFACE_FORMS
    assert not (set(SPECIAL_TEAM_SURFACES) & set(TEAM_ALIAS_SURFACES))
    assert not (set(SPECIAL_TEAM_SURFACES) & {normalise_surface(n) for n in CANONICAL_TEAM_NAMES})


def test_json_serialisable() -> None:
    json.dumps([CANONICAL_TEAM_NAMES, TEAM_ALIAS_SURFACES, AMBIGUOUS_TEAM_SURFACES,
                list(TEAM_SURFACE_FORMS)])


# --- required canonical teams -----------------------------------------------

@pytest.mark.parametrize("name", [
    "Boston Celtics", "Golden State Warriors", "Los Angeles Lakers", "Los Angeles Clippers",
    "Miami Heat", "New York Knicks", "Brooklyn Nets", "Philadelphia 76ers",
])
def test_required_canonical_teams_present(name) -> None:
    assert name in CANONICAL_TEAM_NAMES


def test_all_thirty_franchises_present() -> None:
    assert len(CANONICAL_TEAM_NAMES) == 30


# --- ambiguous surfaces -----------------------------------------------------

@pytest.mark.parametrize("surface", ["la", "los angeles", "ny", "new york"])
def test_ambiguous_surfaces_present(surface) -> None:
    assert surface in AMBIGUOUS_TEAM_SURFACES
    assert surface in TEAM_SURFACE_FORMS


# --- longest-match ordering -------------------------------------------------

def test_longest_match_ordering() -> None:
    ordered = get_team_surface_forms_by_length()
    pos = {s: i for i, s in enumerate(ordered)}
    assert pos["boston celtics"] < pos["celtics"]
    assert pos["golden state warriors"] < pos["warriors"]
    assert pos["los angeles lakers"] < pos["los angeles"]
    assert pos["los angeles clippers"] < pos["los angeles"]
    assert pos["los angeles"] < pos["la"]


def test_ordering_is_deterministic() -> None:
    assert get_team_surface_forms_by_length() == get_team_surface_forms_by_length()


# --- drift protection (tests may use Phase 7 sources) -----------------------

@pytest.fixture(scope="module")
def context():
    from src.data_loader import load_raw_dataset
    from src.data_model import build_clean_view, validate_clean_view
    from src.data_validation import validate_dataset
    from src.tool_registry import DEFAULT_REGISTRY
    from src.validation_context import build_validation_context

    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return build_validation_context(clean, registry=DEFAULT_REGISTRY)


def test_canonical_names_match_dataset(context) -> None:
    assert set(CANONICAL_TEAM_NAMES) == set(context.canonical_teams)


def test_special_team_surfaces_match_dataset(context) -> None:
    assert set(SPECIAL_TEAM_SURFACES) == {normalise_surface(t) for t in context.special_teams}
    for special in context.special_teams:
        assert special not in CANONICAL_TEAM_NAMES  # special teams are never canonical
        assert normalise_surface(special) in TEAM_SURFACE_FORMS  # but are extractable surfaces


def test_alias_surfaces_match_resolver_keys() -> None:
    from src.team_resolution import ALIAS_MAP
    assert set(TEAM_ALIAS_SURFACES) == set(ALIAS_MAP.keys())


def test_ambiguous_surfaces_match_resolver_keys() -> None:
    from src.team_resolution import AMBIGUITY_MAP
    assert set(AMBIGUOUS_TEAM_SURFACES) == set(AMBIGUITY_MAP.keys())


def test_alias_and_ambiguous_surfaces_are_disjoint() -> None:
    assert not (set(TEAM_ALIAS_SURFACES) & set(AMBIGUOUS_TEAM_SURFACES))


# --- import / scope safety --------------------------------------------------

def test_catalogue_import_is_lightweight() -> None:
    code = (
        "import sys; import src.team_surface_catalogue;"
        "forbidden = ['pandas', 'src.data_loader', 'src.tool_registry', 'src.tools',"
        " 'src.validation_context', 'src.intent_validator', 'src.rule_parser'];"
        "assert not any(m in sys.modules for m in forbidden), [m for m in forbidden if m in sys.modules];"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_future_modules_absent() -> None:
    for module in ("src.rule_parser_validation_integration", "src.llm_query_parser"):
        assert importlib.util.find_spec(module) is None
