"""Phase 6C: registry integration review — consolidation checks.

Confirms the registry dispatches *identically* to direct tool calls, and that registry
construction is data-free and import-safe. The broader registry contract (completeness,
schema safety, schema/signature consistency, error separation, immutability, oracles) is
covered by tests/test_tool_registry.py; this file adds the integration-level guarantees.
No network, no LLM.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

import src.tools as tools
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.tool_registry import build_default_registry, execute, schemas


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return clean


# --- Check 6: registry result == direct tool call (all six) -----------------

DIRECT_VS_REGISTRY = [
    ("team_average_points",
     lambda c: tools.team_average_points(c, "Golden State Warriors", window=5),
     {"team": "Golden State Warriors", "window": 5}),
    ("average_points_allowed",
     lambda c: tools.average_points_allowed(c, "Golden State Warriors", window=5),
     {"team": "Golden State Warriors", "window": 5}),
    ("team_record",
     lambda c: tools.team_record(c, "Golden State Warriors"),
     {"team": "Golden State Warriors"}),
    ("top_scoring_teams",
     lambda c: tools.top_scoring_teams(c, n=5),
     {"n": 5}),
    ("head_to_head",
     lambda c: tools.head_to_head(c, "Boston Celtics", "Miami Heat"),
     {"team_a": "Boston Celtics", "team_b": "Miami Heat"}),
    ("team_efficiency_summary",
     lambda c: tools.team_efficiency_summary(c, "Boston Celtics", window=10),
     {"team": "Boston Celtics", "window": 10}),
]


@pytest.mark.parametrize(
    "name,direct_call,args", DIRECT_VS_REGISTRY, ids=[d[0] for d in DIRECT_VS_REGISTRY]
)
def test_registry_matches_direct_tool_call(clean_df, name, direct_call, args) -> None:
    direct = direct_call(clean_df)
    via_registry = execute(name, args, clean_df=clean_df)
    assert via_registry == direct


# --- Check 12: import / dependency safety -----------------------------------

def test_registry_construction_is_data_free() -> None:
    # Building the registry and exporting schemas requires no dataset.
    assert len(build_default_registry().schemas()) == 6
    assert len(schemas()) == 6  # default registry was built at import, data-free


def test_registry_construction_does_not_read_dataset(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("read_csv must not be called during registry construction.")

    monkeypatch.setattr(pd, "read_csv", boom)
    assert len(build_default_registry().schemas()) == 6


def test_imports_are_acyclic() -> None:
    # tool_registry depends on tools (one-directional); tools must not import the registry.
    mod = importlib.import_module("src.tool_registry")
    assert hasattr(mod, "DEFAULT_REGISTRY")
    assert not hasattr(tools, "build_default_registry")
    assert not hasattr(tools, "tool_registry")


@pytest.mark.parametrize("bad_args", [["bad"], "bad", 123])
def test_default_registry_rejects_non_dict_args(clean_df, bad_args) -> None:
    # The module-level execute() (DEFAULT_REGISTRY) rejects non-dict args structurally.
    res = execute("team_average_points", bad_args, clean_df=clean_df)
    assert res["status"] == "error" and res["tool"] == "tool_registry"
