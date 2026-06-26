"""Phase 6A/6B tests: tool registry foundation and the six registered analytical tools.

Phase 6A tests use isolated dummy tools and fresh ``ToolRegistry()`` instances so the
default registry is never polluted. Phase 6B tests exercise the populated
``DEFAULT_REGISTRY`` (the six real tools) end to end. No network, no LLM.
"""

from __future__ import annotations

import importlib.util
import inspect
import json

import pandas as pd
import pytest

import src.tools as real_tools
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.tool_registry import (
    DEFAULT_REGISTRY,
    ToolParameter,
    ToolRegistry,
    ToolSpec,
    build_default_registry,
    execute,
    is_registered,
    schema,
    schemas,
)

EXPECTED_TOOL_ORDER = [
    "team_average_points",
    "average_points_allowed",
    "team_record",
    "top_scoring_teams",
    "head_to_head",
    "team_efficiency_summary",
]

LATER_LAYER_MODULES = (
    "src.query_parser",
    "src.llm_query_parser",
    "src.assistant",
)


# --- dummy tools ------------------------------------------------------------

def dummy_team_tool(clean_df, team, window=None):
    from src.tool_results import ok_result

    return ok_result(
        "dummy_team_tool",
        {"team": team, "window": window, "rows": len(clean_df)},
    )


def dummy_noarg_tool(clean_df):
    from src.tool_results import ok_result

    return ok_result("dummy_noarg_tool", {"rows": len(clean_df)})


def dummy_raising_tool(clean_df, team):
    raise RuntimeError("boom")


def team_param_pair():
    return (
        ToolParameter(name="team", type="str", required=True, description="Canonical team name."),
        ToolParameter(name="window", type="int|null", required=False,
                      description="Optional recent-game window.", default=None),
    )


def team_spec():
    return ToolSpec(
        name="dummy_team_tool",
        description="Dummy team tool for registry testing.",
        parameters=team_param_pair(),
        function=dummy_team_tool,
    )


def noarg_spec():
    return ToolSpec(
        name="dummy_noarg_tool",
        description="Dummy tool with no required args.",
        parameters=(),
        function=dummy_noarg_tool,
    )


@pytest.fixture()
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3]})


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return clean


# --- Test 1 & 2: schemas ----------------------------------------------------

def test_tool_parameter_schema_is_json_serialisable() -> None:
    param = ToolParameter(name="window", type="int|null", required=False,
                          description="Window.", default=None)
    json.dumps(param.to_schema())


def test_tool_spec_schema_excludes_function_and_clean_df() -> None:
    schema = team_spec().to_schema()
    assert schema["name"] == "dummy_team_tool"
    assert schema["description"]
    assert isinstance(schema["parameters"], list) and len(schema["parameters"]) == 2
    assert "function" not in schema
    assert "clean_df" not in {p["name"] for p in schema["parameters"]}
    json.dumps(schema)


# --- Test 3: empty registry -------------------------------------------------

def test_empty_registry() -> None:
    reg = ToolRegistry()
    assert reg.schemas() == []
    assert reg.get("anything") is None
    assert reg.schema("anything") is None
    assert reg.is_registered("anything") is False


# Note: DEFAULT_REGISTRY is empty until Phase 6B registers the six tools; its populated
# state is asserted by test_default_registry_has_exactly_six_tools_in_order below.


# --- Test 4: register dummy -------------------------------------------------

def test_register_dummy_tool() -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    assert reg.is_registered("dummy_team_tool") is True
    assert isinstance(reg.get("dummy_team_tool"), ToolSpec)
    assert len(reg.schemas()) == 1


def test_schemas_deterministic_order() -> None:
    reg = ToolRegistry()
    reg.register(noarg_spec())
    reg.register(team_spec())
    assert [s["name"] for s in reg.schemas()] == ["dummy_noarg_tool", "dummy_team_tool"]


# --- Test 5: duplicate registration -----------------------------------------

def test_duplicate_registration_raises() -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    with pytest.raises(ValueError):
        reg.register(team_spec())


# --- Test 6: invalid ToolSpec / ToolParameter -------------------------------

def test_invalid_specs_raise() -> None:
    with pytest.raises(ValueError):
        ToolSpec(name="", description="d", parameters=(), function=dummy_noarg_tool)
    with pytest.raises(ValueError):
        ToolSpec(name="n", description="", parameters=(), function=dummy_noarg_tool)
    with pytest.raises(ValueError):
        ToolSpec(name="n", description="d", parameters=(), function="not callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dup = (
            ToolParameter(name="team", type="str", required=True, description="a"),
            ToolParameter(name="team", type="str", required=False, description="b"),
        )
        ToolSpec(name="n", description="d", parameters=dup, function=dummy_team_tool)
    with pytest.raises(ValueError):
        ToolParameter(name="clean_df", type="str", required=True, description="bad")


def test_tool_parameter_rejects_raw_python_type_object() -> None:
    with pytest.raises(ValueError):
        ToolParameter(name="x", type=int, required=True, description="x")  # type: ignore[arg-type]


def test_tool_parameter_rejects_unknown_type_string() -> None:
    with pytest.raises(ValueError):
        ToolParameter(name="x", type="integer", required=True, description="x")


def test_tool_parameter_rejects_non_bool_required() -> None:
    with pytest.raises(ValueError):
        ToolParameter(name="x", type="int", required=1, description="x")  # type: ignore[arg-type]


def test_tool_parameter_rejects_empty_description() -> None:
    with pytest.raises(ValueError):
        ToolParameter(name="x", type="int", required=True, description="")


def test_tool_parameter_rejects_non_json_serialisable_default() -> None:
    with pytest.raises(ValueError):
        ToolParameter(name="x", type="int", required=False, description="x", default={1, 2})


def test_tool_spec_rejects_non_tool_parameter_entries() -> None:
    with pytest.raises(ValueError):
        ToolSpec(
            name="n", description="d",
            parameters=("not_a_parameter",),  # type: ignore[arg-type]
            function=dummy_noarg_tool,
        )


# --- Test 7: successful execute ---------------------------------------------

def test_execute_returns_underlying_result_unchanged(small_df) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    res = reg.execute("dummy_team_tool", {"team": "Boston Celtics", "window": 5}, clean_df=small_df)
    assert res["status"] == "ok"
    assert res["tool"] == "dummy_team_tool"
    assert res["result"] == {"team": "Boston Celtics", "window": 5, "rows": 3}
    json.dumps(res)


# --- Test 8: args=None ------------------------------------------------------

def test_execute_args_none(small_df) -> None:
    reg = ToolRegistry()
    reg.register(noarg_spec())
    reg.register(team_spec())
    ok = reg.execute("dummy_noarg_tool", None, clean_df=small_df)
    assert ok["status"] == "ok"
    err = reg.execute("dummy_team_tool", None, clean_df=small_df)
    assert err["status"] == "error"
    assert err["tool"] == "tool_registry"


# --- Test 9: unknown tool ---------------------------------------------------

def test_unknown_tool_structured_error(small_df) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    res = reg.execute("unknown_tool", {}, clean_df=small_df)
    assert res["status"] == "error"
    assert res["tool"] == "tool_registry"
    assert res["result"]["requested_tool"] == "unknown_tool"
    assert "dummy_team_tool" in res["result"]["available_tools"]
    json.dumps(res)


# --- Test 10/11/12/13: shallow validation errors ----------------------------

def test_missing_required_argument_error(small_df) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    res = reg.execute("dummy_team_tool", {}, clean_df=small_df)
    assert res["status"] == "error" and res["tool"] == "tool_registry"
    assert "team" in res["result"]["missing"]


def test_unexpected_argument_error(small_df) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    res = reg.execute("dummy_team_tool", {"team": "Boston Celtics", "bad_arg": 1}, clean_df=small_df)
    assert res["status"] == "error" and res["tool"] == "tool_registry"
    assert "bad_arg" in res["result"]["unexpected"]


def test_clean_df_in_args_error(small_df) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    res = reg.execute(
        "dummy_team_tool", {"team": "Boston Celtics", "clean_df": small_df}, clean_df=small_df
    )
    assert res["status"] == "error" and res["tool"] == "tool_registry"


@pytest.mark.parametrize("bad_args", [["bad"], "bad", 123])
def test_args_must_be_dict_or_none(small_df, bad_args) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    res = reg.execute("dummy_team_tool", bad_args, clean_df=small_df)
    assert res["status"] == "error" and res["tool"] == "tool_registry"


# --- Test 14: shallow validation only (no deep checks) ----------------------

def test_shallow_validation_does_not_reject_zero_window(small_df) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    res = reg.execute("dummy_team_tool", {"team": "Boston Celtics", "window": 0}, clean_df=small_df)
    assert res["status"] == "ok"  # registry doesn't judge window value
    assert res["result"]["window"] == 0


# --- Test 15/16: no mutation ------------------------------------------------

def test_args_not_mutated(small_df) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    args = {"team": "Boston Celtics", "window": 5}
    reg.execute("dummy_team_tool", args, clean_df=small_df)
    assert args == {"team": "Boston Celtics", "window": 5}


def test_clean_df_not_mutated(small_df) -> None:
    reg = ToolRegistry()
    reg.register(team_spec())
    before = small_df.copy(deep=True)
    reg.execute("dummy_team_tool", {"team": "Boston Celtics", "window": 5}, clean_df=small_df)
    assert small_df.equals(before)


# --- Test 17: underlying exception ------------------------------------------

def test_underlying_exception_structured_error(small_df) -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="dummy_raising_tool",
            description="Dummy tool that raises.",
            parameters=(ToolParameter(name="team", type="str", required=True, description="t"),),
            function=dummy_raising_tool,
        )
    )
    res = reg.execute("dummy_raising_tool", {"team": "x"}, clean_df=small_df)
    assert res["status"] == "error"
    assert res["tool"] == "tool_registry"
    assert res["result"]["requested_tool"] == "dummy_raising_tool"
    assert res["result"]["exception_type"] == "RuntimeError"
    assert res["result"]["exception_message"] == "boom"
    json.dumps(res)


# --- Test 18: scope guard ---------------------------------------------------

def test_no_later_layer_systems_exist() -> None:
    for module in LATER_LAYER_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


# ===========================================================================
# Phase 6B — the six real analytical tools registered in DEFAULT_REGISTRY
# ===========================================================================

DUMMY_NAMES = ("dummy_team_tool", "dummy_noarg_tool", "dummy_raising_tool")

# Expected user-facing parameters per tool: {name: (type, required, default)}.
EXPECTED_PARAMS = {
    "team_average_points": {"team": ("str", True, None), "window": ("int|null", False, None)},
    "average_points_allowed": {"team": ("str", True, None), "window": ("int|null", False, None)},
    "team_record": {"team": ("str", True, None), "window": ("int|null", False, None)},
    "top_scoring_teams": {"n": ("int", False, 5), "season_id": ("int|null", False, None)},
    "head_to_head": {
        "team_a": ("str", True, None), "team_b": ("str", True, None),
        "window": ("int|null", False, None),
    },
    "team_efficiency_summary": {"team": ("str", True, None), "window": ("int|null", False, None)},
}


def test_default_registry_has_exactly_six_tools_in_order() -> None:
    names = [s["name"] for s in DEFAULT_REGISTRY.schemas()]
    assert names == EXPECTED_TOOL_ORDER


def test_no_dummy_tools_in_default_registry() -> None:
    for name in DUMMY_NAMES:
        assert not DEFAULT_REGISTRY.is_registered(name)


def test_module_wrappers_delegate_to_default_registry(clean_df) -> None:
    assert schemas() == DEFAULT_REGISTRY.schemas()
    assert schema("team_record") == DEFAULT_REGISTRY.schema("team_record")
    assert is_registered("head_to_head") is True
    res = execute("team_record", {"team": "Golden State Warriors"}, clean_df=clean_df)
    assert res["status"] == "ok" and res["tool"] == "team_record"


def test_default_schemas_are_json_serialisable() -> None:
    s = schemas()
    assert isinstance(s, list) and len(s) == 6
    for entry in s:
        assert {"name", "description", "parameters"} <= set(entry)
        assert "function" not in entry
        assert "clean_df" not in {p["name"] for p in entry["parameters"]}
    json.dumps(s)


@pytest.mark.parametrize("tool_name", EXPECTED_TOOL_ORDER)
def test_individual_schema_correctness(tool_name) -> None:
    params = {p["name"]: p for p in schema(tool_name)["parameters"]}
    expected = EXPECTED_PARAMS[tool_name]
    assert set(params) == set(expected)
    for name, (ptype, required, default) in expected.items():
        assert params[name]["type"] == ptype
        assert params[name]["required"] is required
        if not required:
            assert params[name]["default"] == default


@pytest.mark.parametrize("tool_name", EXPECTED_TOOL_ORDER)
def test_schema_matches_function_signature(tool_name) -> None:
    func = getattr(real_tools, tool_name)
    sig = inspect.signature(func)
    func_params = list(sig.parameters)
    assert func_params[0] == "clean_df"
    user_params = func_params[1:]
    schema_param_names = [p["name"] for p in schema(tool_name)["parameters"]]
    assert schema_param_names == user_params  # same names, same order, clean_df excluded
    for p in schema(tool_name)["parameters"]:
        fp = sig.parameters[p["name"]]
        if p["required"]:
            assert fp.default is inspect.Parameter.empty
        else:
            assert fp.default == p["default"]


def test_schema_returns_safe_copy() -> None:
    s1 = schema("team_average_points")
    s1["name"] = "HACKED"
    s1["parameters"].append({"injected": True})
    s2 = schema("team_average_points")
    assert s2["name"] == "team_average_points"
    assert len(s2["parameters"]) == 2
    assert "function" not in s2


# --- Test 7: oracle execution through the registry --------------------------

def test_registry_execution_oracles(clean_df) -> None:
    avg = execute("team_average_points", {"team": "Golden State Warriors", "window": 5}, clean_df=clean_df)
    assert avg["status"] == "ok" and avg["tool"] == "team_average_points"
    assert round(avg["result"]["average_points"], 2) == 114.4

    allowed = execute("average_points_allowed", {"team": "Golden State Warriors", "window": 5}, clean_df=clean_df)
    assert round(allowed["result"]["average_points_allowed"], 2) == 117.0

    rec = execute("team_record", {"team": "Golden State Warriors"}, clean_df=clean_df)
    assert rec["result"]["record"] == "289-223"

    top = execute("top_scoring_teams", {"n": 5}, clean_df=clean_df)
    first = top["result"]["teams"][0]
    assert first["team"] == "Atlanta Hawks" and round(first["average_points"], 2) == 116.13

    h2h = execute("head_to_head", {"team_a": "Boston Celtics", "team_b": "Miami Heat"}, clean_df=clean_df)
    assert (h2h["result"]["meetings"], h2h["result"]["record"]) == (39, "25-14")

    eff = execute("team_efficiency_summary", {"team": "Boston Celtics", "window": 10}, clean_df=clean_df)
    assert (round(eff["result"]["average_ortg"], 2), round(eff["result"]["average_drtg"], 2)) == (106.98, 101.93)
    json.dumps([avg, allowed, rec, top, h2h, eff])


# --- Test 8: args=None ------------------------------------------------------

def test_args_none_uses_defaults_or_errors(clean_df) -> None:
    ok = execute("top_scoring_teams", None, clean_df=clean_df)
    assert ok["status"] == "ok"
    assert ok["result"]["teams_returned"] == 5
    err = execute("team_average_points", None, clean_df=clean_df)
    assert err["status"] == "error" and err["tool"] == "tool_registry"


# --- Tests 9-12: registry-level errors --------------------------------------

def test_unknown_tool_via_default_registry(clean_df) -> None:
    res = execute("unknown_tool", {}, clean_df=clean_df)
    assert res["status"] == "error" and res["tool"] == "tool_registry"
    assert res["result"]["requested_tool"] == "unknown_tool"
    assert set(EXPECTED_TOOL_ORDER) <= set(res["result"]["available_tools"])
    json.dumps(res)


@pytest.mark.parametrize("name,args", [
    ("team_average_points", {}),
    ("head_to_head", {"team_a": "Boston Celtics"}),
])
def test_missing_required_args_registry_error(clean_df, name, args) -> None:
    res = execute(name, args, clean_df=clean_df)
    assert res["status"] == "error" and res["tool"] == "tool_registry"
    assert res["result"]["missing"]


@pytest.mark.parametrize("name,args", [
    ("team_record", {"team": "Golden State Warriors", "bad_arg": 1}),
    ("top_scoring_teams", {"team": "Golden State Warriors"}),  # 'team' is not a param here
])
def test_unexpected_args_registry_error(clean_df, name, args) -> None:
    res = execute(name, args, clean_df=clean_df)
    assert res["status"] == "error" and res["tool"] == "tool_registry"
    assert res["result"]["unexpected"]


def test_clean_df_in_args_registry_error(clean_df) -> None:
    res = execute(
        "team_average_points",
        {"team": "Golden State Warriors", "clean_df": clean_df},
        clean_df=clean_df,
    )
    assert res["status"] == "error" and res["tool"] == "tool_registry"


# --- Tests 13-14: tool-level pass-through -----------------------------------

@pytest.mark.parametrize("name,args", [
    ("team_average_points", {"team": "Golden State Warriors", "window": 0}),
    ("top_scoring_teams", {"n": 0}),
])
def test_tool_level_invalid_args_pass_through(clean_df, name, args) -> None:
    res = execute(name, args, clean_df=clean_df)
    assert res["status"] == "error"
    assert res["tool"] == name  # NOT tool_registry


def test_tool_level_no_data_passes_through(clean_df) -> None:
    res = execute("team_average_points", {"team": "Not A Real Team", "window": 5}, clean_df=clean_df)
    assert res["status"] == "no_data"
    assert res["tool"] == "team_average_points"


# --- Tests 15-16: no mutation -----------------------------------------------

def test_registry_execution_does_not_mutate_args(clean_df) -> None:
    args = {"team": "Golden State Warriors", "window": 5}
    execute("team_average_points", args, clean_df=clean_df)
    assert args == {"team": "Golden State Warriors", "window": 5}


@pytest.mark.parametrize("name,args", [
    ("team_average_points", {"team": "Golden State Warriors", "window": 5}),
    ("average_points_allowed", {"team": "Golden State Warriors", "window": 5}),
    ("team_record", {"team": "Golden State Warriors"}),
    ("top_scoring_teams", {"n": 5}),
    ("head_to_head", {"team_a": "Boston Celtics", "team_b": "Miami Heat"}),
    ("team_efficiency_summary", {"team": "Boston Celtics", "window": 10}),
])
def test_registry_execution_does_not_mutate_clean_df(clean_df, name, args) -> None:
    before = clean_df.copy(deep=True)
    execute(name, args, clean_df=clean_df)
    assert clean_df.equals(before)


def test_build_default_registry_returns_fresh_registry() -> None:
    a = build_default_registry()
    b = build_default_registry()
    assert a is not b
    assert a is not DEFAULT_REGISTRY
    assert [s["name"] for s in a.schemas()] == EXPECTED_TOOL_ORDER
    # Registering into a fresh registry must not affect DEFAULT_REGISTRY.
    a.register(noarg_spec())
    assert not DEFAULT_REGISTRY.is_registered("dummy_noarg_tool")


def test_default_registry_has_no_duplicate_schema_names() -> None:
    names = [s["name"] for s in DEFAULT_REGISTRY.schemas()]
    assert len(names) == len(set(names))
