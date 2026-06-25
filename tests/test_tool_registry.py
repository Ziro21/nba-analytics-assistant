"""Phase 6A tests: tool registry foundation, using isolated dummy tools only.

No real Phase 5 analytical tools are registered here. Each test uses a fresh
``ToolRegistry()`` so the default registry is never polluted. No network, no LLM.
"""

from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest

from src.tool_registry import ToolParameter, ToolRegistry, ToolSpec

LATER_LAYER_MODULES = (
    "src.query_parser",
    "src.llm_query_parser",
    "src.intent_validator",
    "src.response_formatter",
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


def test_default_registry_starts_empty() -> None:
    from src.tool_registry import DEFAULT_REGISTRY

    assert DEFAULT_REGISTRY.schemas() == []


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
