"""Phase 7B tests: the validation reference context built from the clean frame + registry."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.team_resolution import normalise_team_text
from src.tool_registry import DEFAULT_REGISTRY
from src.validation_context import ValidationContext, build_validation_context

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TOOL_ORDER = (
    "team_average_points", "average_points_allowed", "team_record",
    "top_scoring_teams", "head_to_head", "team_efficiency_summary",
)
DUMMY_NAMES = ("dummy_team_tool", "dummy_noarg_tool", "dummy_raising_tool")
FORBIDDEN_MODULES = (
    "src.query_parser", "src.llm_query_parser",
)


@pytest.fixture(scope="module")
def context() -> ValidationContext:
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return build_validation_context(clean, registry=DEFAULT_REGISTRY)


# --- Construction -----------------------------------------------------------

def test_context_builds_and_is_json_serialisable(context) -> None:
    json.dumps(context.to_dict())


def test_context_does_not_store_dataframe(context) -> None:
    for value in vars(context).values():
        assert not isinstance(value, pd.DataFrame)


# --- Registered tools -------------------------------------------------------

def test_registered_tools(context) -> None:
    assert context.registered_tools == EXPECTED_TOOL_ORDER
    assert len(context.tool_schemas) == 6
    for name in DUMMY_NAMES:
        assert name not in context.registered_tools


def test_tool_schemas_safe_and_serialisable(context) -> None:
    for schema in context.tool_schemas:
        assert {"name", "description", "parameters"} <= set(schema)
        assert "function" not in schema
        assert "clean_df" not in {p["name"] for p in schema["parameters"]}
    json.dumps(context.to_dict()["tool_schemas"])  # plain dicts via to_dict


def test_context_tool_schemas_are_not_mutable(context) -> None:
    with pytest.raises(TypeError):
        context.tool_schemas[0]["name"] = "hacked"


def test_context_tool_schema_by_name_nested_values_are_not_mutable(context) -> None:
    schema = context.tool_schema_by_name["team_average_points"]
    with pytest.raises(TypeError):
        schema["description"] = "hacked"
    with pytest.raises(TypeError):
        schema["parameters"][0]["name"] = "hacked"


# --- Canonical teams --------------------------------------------------------

def test_canonical_teams(context) -> None:
    assert len(context.canonical_teams) == 30
    for team in ("Boston Celtics", "Golden State Warriors", "Los Angeles Lakers",
                 "Los Angeles Clippers", "Miami Heat", "New York Knicks", "Brooklyn Nets"):
        assert team in context.canonical_teams
    for special in ("Team Stars", "Team Stripes", "Team World"):
        assert special not in context.canonical_teams


def test_special_teams(context) -> None:
    assert context.special_teams == ("Team Stars", "Team Stripes", "Team World")


def test_valid_season_ids(context) -> None:
    assert context.valid_season_ids == (26, 28, 30, 32, 34, 36)


# --- Normalised lookup ------------------------------------------------------

def test_normalised_lookup(context) -> None:
    assert context.normalised_team_lookup["boston celtics"] == "Boston Celtics"
    assert context.normalised_team_lookup["golden state warriors"] == "Golden State Warriors"
    for norm, canonical in context.normalised_team_lookup.items():
        assert normalise_team_text(canonical) == norm


# --- Alias / ambiguity maps -------------------------------------------------

def test_alias_and_ambiguity_consistency(context) -> None:
    canonical_set, special_set = set(context.canonical_teams), set(context.special_teams)
    for target in context.alias_map.values():
        assert target in canonical_set
        assert target not in special_set
    for candidates in context.ambiguity_map.values():
        for candidate in candidates:
            assert candidate in canonical_set
    assert set(context.alias_map).isdisjoint(set(context.ambiguity_map))


# --- Immutability / defensive copy ------------------------------------------

def test_to_dict_mutation_does_not_affect_context(context) -> None:
    d = context.to_dict()
    d["canonical_teams"].append("Fake Team")
    d["alias_map"]["zzz"] = "Fake Team"
    assert "Fake Team" not in context.canonical_teams
    assert "zzz" not in context.alias_map


def test_internal_maps_are_read_only(context) -> None:
    with pytest.raises(TypeError):
        context.alias_map["x"] = "y"
    with pytest.raises(TypeError):
        context.tool_schema_by_name["x"] = {}


# --- Import / scope safety --------------------------------------------------

def test_forbidden_modules_absent() -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


def test_validation_context_import_is_lightweight() -> None:
    code = (
        "import sys; import src.validation_context;"
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


def test_build_context_does_not_call_registry_execute(context) -> None:
    raw = load_raw_dataset()
    clean = build_clean_view(raw)

    class FakeRegistry:
        def schemas(self):
            return DEFAULT_REGISTRY.schemas()

        def execute(self, *args, **kwargs):
            raise AssertionError("build_validation_context must not call registry.execute")

    built = build_validation_context(clean, registry=FakeRegistry())
    assert built.registered_tools == EXPECTED_TOOL_ORDER
