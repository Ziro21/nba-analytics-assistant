"""Phase 7C tests: the shared intent validator.

A `ValidationContext` is built once from the real pipeline. The validator executes nothing
(no registry, no tools, no data load). No closed-loop validate->execute here (that's 7D).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.intent_types import ParsedIntent
from src.intent_validator import _type_ok, validate_intent
from src.tool_registry import ALLOWED_PARAM_TYPES, DEFAULT_REGISTRY
from src.validation_context import build_validation_context

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_MODULES = (
    "src.query_parser",
)


@pytest.fixture(scope="module")
def context():
    raw = load_raw_dataset()
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    return build_validation_context(clean, registry=DEFAULT_REGISTRY)


def _codes(result) -> set[str]:
    return {e.code for e in result.errors}


# --- 1. Import / scope safety ----------------------------------------------

def test_forbidden_modules_absent() -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


def test_validator_import_is_lightweight() -> None:
    code = (
        "import sys; import src.intent_validator;"
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


# --- 2. Valid intents -------------------------------------------------------

def test_exact_team_valid(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Boston Celtics", "window": 5}, "rule"), context=context)
    assert res.is_valid
    assert res.validated_intent.arguments == {"team": "Boston Celtics", "window": 5}
    assert res.warnings == ()  # exact -> no canonicalisation warning


def test_normalised_team_canonicalises(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "golden state warriors", "window": 5}, "llm"), context=context)
    assert res.is_valid
    assert res.validated_intent.arguments["team"] == "Golden State Warriors"
    assert any(w.code == "canonicalised_team" for w in res.warnings)


def test_alias_team_canonicalises(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "gsw", "window": 5}, "llm"), context=context)
    assert res.is_valid
    assert res.validated_intent.arguments["team"] == "Golden State Warriors"


def test_top_scoring_defaults_valid(context) -> None:
    res = validate_intent(ParsedIntent("top_scoring_teams", {}, "rule"), context=context)
    assert res.is_valid
    assert res.validated_intent.arguments == {}


def test_top_scoring_with_season_valid(context) -> None:
    res = validate_intent(ParsedIntent("top_scoring_teams", {"n": 5, "season_id": 26}, "llm"), context=context)
    assert res.is_valid
    assert res.validated_intent.arguments == {"n": 5, "season_id": 26}


def test_head_to_head_canonicalises_both(context) -> None:
    res = validate_intent(ParsedIntent("head_to_head", {"team_a": "celtics", "team_b": "heat"}, "rule"), context=context)
    assert res.is_valid
    assert res.validated_intent.arguments == {"team_a": "Boston Celtics", "team_b": "Miami Heat"}


def test_validated_intent_json_serialisable(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "gsw", "window": 5}, "llm"), context=context)
    json.dumps(res.to_dict())


# --- 3. Parser-mode invariance ----------------------------------------------

def _pair(context, tool, args):
    r = validate_intent(ParsedIntent(tool, args, "rule"), context=context)
    l = validate_intent(ParsedIntent(tool, args, "llm"), context=context)
    return r, l


def _assert_same_outcome(r, l) -> None:
    assert r.is_valid == l.is_valid
    assert {e.code for e in r.errors} == {e.code for e in l.errors}
    assert {w.code for w in r.warnings} == {w.code for w in l.warnings}
    if r.is_valid:
        assert r.validated_intent.arguments == l.validated_intent.arguments
        assert (r.validated_intent.parser_mode, l.validated_intent.parser_mode) == ("rule", "llm")


@pytest.mark.parametrize("tool,args", [
    ("team_average_points", {"team": "gsw", "window": 5}),       # valid
    ("team_average_points", {"team": "Celics", "window": 5}),    # unknown team
    ("team_average_points", {"team": "LA", "window": 5}),        # ambiguous
    ("team_average_points", {"team": "Boston Celtics", "window": "5"}),  # invalid type
])
def test_parser_mode_invariance(context, tool, args) -> None:
    r, l = _pair(context, tool, args)
    _assert_same_outcome(r, l)


# --- 4. Tool / schema validation --------------------------------------------

def test_unknown_tool(context) -> None:
    res = validate_intent(ParsedIntent("fake_tool", {}, "rule"), context=context)
    assert not res.is_valid and _codes(res) == {"unknown_tool"}


def test_missing_required_argument(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {}, "rule"), context=context)
    assert "missing_required_argument" in _codes(res)


def test_unexpected_argument(context) -> None:
    res = validate_intent(ParsedIntent("team_record", {"team": "Boston Celtics", "bad_arg": 1}, "rule"), context=context)
    assert "unexpected_argument" in _codes(res)


def test_arguments_not_dict(context) -> None:
    intent = ParsedIntent("team_average_points", {"team": "Boston Celtics"}, "rule")
    object.__setattr__(intent, "arguments", ["not", "a", "dict"])  # bypass the contract
    res = validate_intent(intent, context=context)
    assert _codes(res) == {"arguments_not_dict"}


def test_schema_driven_allowed_args(context) -> None:
    # top_scoring_teams allows n/season_id but not team.
    res = validate_intent(ParsedIntent("top_scoring_teams", {"team": "Boston Celtics"}, "rule"), context=context)
    assert "unexpected_argument" in _codes(res)


# --- 4b. location (home/away contextual splits) -----------------------------

@pytest.mark.parametrize("loc", ["home", "away", None])
def test_location_home_away_none_valid(context, loc) -> None:
    args = {"team": "Boston Celtics"} | ({"location": loc} if loc is not None else {})
    assert validate_intent(ParsedIntent("team_record", args, "rule"), context=context).is_valid


def test_validator_rejects_neutral_location(context) -> None:
    # the unsupported "neutral" venue (emitted as a raw slot) must be rejected, not ignored.
    res = validate_intent(
        ParsedIntent("team_record", {"team": "Boston Celtics", "location": "neutral"}, "rule"),
        context=context)
    assert not res.is_valid and "invalid_location" in _codes(res)


def test_validator_rejects_location_on_top_scoring(context) -> None:
    res = validate_intent(
        ParsedIntent("top_scoring_teams", {"n": 5, "location": "home"}, "rule"), context=context)
    assert "unexpected_argument" in _codes(res)


@pytest.mark.parametrize("bad", [True, 1, ["home"]])
def test_validator_rejects_non_string_location(context, bad) -> None:
    res = validate_intent(
        ParsedIntent("team_record", {"team": "Boston Celtics", "location": bad}, "rule"),
        context=context)
    assert "invalid_argument_type" in _codes(res)


# --- 5. Type validation -----------------------------------------------------

def test_string_window_invalid_type(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Boston Celtics", "window": "5"}, "llm"), context=context)
    assert "invalid_argument_type" in _codes(res)
    assert "invalid_window" not in _codes(res)  # no cascade


def test_bool_window_invalid_type(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Boston Celtics", "window": True}, "llm"), context=context)
    assert "invalid_argument_type" in _codes(res)


def test_none_window_valid(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Boston Celtics", "window": None}, "rule"), context=context)
    assert res.is_valid


def test_none_n_invalid(context) -> None:
    res = validate_intent(ParsedIntent("top_scoring_teams", {"n": None}, "rule"), context=context)
    assert "invalid_argument_type" in _codes(res)


def test_string_season_id_invalid_type(context) -> None:
    res = validate_intent(ParsedIntent("top_scoring_teams", {"n": 5, "season_id": "26"}, "rule"), context=context)
    assert "invalid_argument_type" in _codes(res)


def test_bool_season_id_invalid_type(context) -> None:
    res = validate_intent(ParsedIntent("top_scoring_teams", {"n": 5, "season_id": True}, "rule"), context=context)
    assert "invalid_argument_type" in _codes(res)


# --- 6. Team resolution -----------------------------------------------------

def test_unknown_team(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Celics", "window": 5}, "llm"), context=context)
    assert "unknown_team" in _codes(res)
    err = next(e for e in res.errors if e.code == "unknown_team")
    assert "Boston Celtics" in err.suggestions


def test_ambiguous_team(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "LA", "window": 5}, "rule"), context=context)
    assert "ambiguous_team" in _codes(res)
    err = next(e for e in res.errors if e.code == "ambiguous_team")
    assert "Los Angeles Lakers" in err.suggestions and "Los Angeles Clippers" in err.suggestions


def test_special_team_rejected(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Team World", "window": 5}, "rule"), context=context)
    assert "invalid_special_team" in _codes(res)
    assert res.validated_intent is None


# --- 7. Domain validation ---------------------------------------------------

@pytest.mark.parametrize("window", [0, -1])
def test_invalid_window(context, window) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Boston Celtics", "window": window}, "rule"), context=context)
    assert "invalid_window" in _codes(res)


def test_over_large_window_passes(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Boston Celtics", "window": 10_000}, "rule"), context=context)
    assert res.is_valid


@pytest.mark.parametrize("n", [0, -1])
def test_invalid_n(context, n) -> None:
    res = validate_intent(ParsedIntent("top_scoring_teams", {"n": n}, "rule"), context=context)
    assert "invalid_n" in _codes(res)


def test_over_large_n_passes(context) -> None:
    res = validate_intent(ParsedIntent("top_scoring_teams", {"n": 10_000}, "rule"), context=context)
    assert res.is_valid


def test_invalid_season_id(context) -> None:
    res = validate_intent(ParsedIntent("top_scoring_teams", {"n": 5, "season_id": 999}, "rule"), context=context)
    assert "invalid_season_id" in _codes(res)


def test_valid_season_id(context) -> None:
    assert validate_intent(ParsedIntent("top_scoring_teams", {"n": 5, "season_id": 26}, "rule"), context=context).is_valid


# --- 8. Head-to-head --------------------------------------------------------

def test_head_to_head_valid(context) -> None:
    assert validate_intent(ParsedIntent("head_to_head", {"team_a": "Boston Celtics", "team_b": "Miami Heat"}, "rule"), context=context).is_valid


@pytest.mark.parametrize("team_a,team_b", [
    ("Boston Celtics", "boston celtics"),
    ("celtics", "bos"),
])
def test_head_to_head_same_team(context, team_a, team_b) -> None:
    res = validate_intent(ParsedIntent("head_to_head", {"team_a": team_a, "team_b": team_b}, "rule"), context=context)
    assert "same_team_head_to_head" in _codes(res)


def test_head_to_head_same_team_check_skipped_when_one_unknown(context) -> None:
    res = validate_intent(ParsedIntent("head_to_head", {"team_a": "Celics", "team_b": "Miami Heat"}, "rule"), context=context)
    assert "unknown_team" in _codes(res)
    assert "same_team_head_to_head" not in _codes(res)


# --- 9. Error accumulation --------------------------------------------------

def test_multiple_independent_errors_accumulate(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Celics", "window": 0}, "rule"), context=context)
    assert {"unknown_team", "invalid_window"} <= _codes(res)


def test_unknown_tool_stops_validation(context) -> None:
    res = validate_intent(ParsedIntent("fake_tool", {"bad": 1, "team": "Celics"}, "rule"), context=context)
    assert _codes(res) == {"unknown_tool"}


# --- 10. Warnings -----------------------------------------------------------

def test_canonicalisation_warning_in_both_places(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "gsw", "window": 5}, "llm"), context=context)
    assert res.warnings and res.warnings[0].severity == "warning"
    assert [w.code for w in res.warnings] == [w.code for w in res.validated_intent.warnings]
    json.dumps([w.to_dict() for w in res.warnings])


def test_exact_team_no_warning(context) -> None:
    res = validate_intent(ParsedIntent("team_average_points", {"team": "Boston Celtics", "window": 5}, "rule"), context=context)
    assert res.warnings == ()


# --- 11. Immutability -------------------------------------------------------

def test_validator_does_not_mutate_intent_arguments(context) -> None:
    intent = ParsedIntent("team_average_points", {"team": "gsw", "window": 5}, "llm")
    validate_intent(intent, context=context)
    assert dict(intent.arguments) == {"team": "gsw", "window": 5}  # original unchanged


def test_validated_arguments_are_new_canonical_dict(context) -> None:
    intent = ParsedIntent("team_average_points", {"team": "gsw", "window": 5}, "llm")
    res = validate_intent(intent, context=context)
    assert dict(res.validated_intent.arguments) == {"team": "Golden State Warriors", "window": 5}
    # mutating the to_dict output does not affect the result object
    d = res.to_dict()
    d["validated_intent"]["arguments"]["team"] = "Hacked"
    assert res.validated_intent.arguments["team"] == "Golden State Warriors"


# --- 12. No execution / type guards -----------------------------------------

def test_non_parsedintent_raises(context) -> None:
    with pytest.raises(TypeError):
        validate_intent({"tool_name": "team_average_points"}, context=context)


def test_non_context_raises() -> None:
    with pytest.raises(TypeError):
        validate_intent(ParsedIntent("team_average_points", {"team": "Boston Celtics"}, "rule"), context=object())


def test_validation_does_not_need_a_registry(context) -> None:
    # The validator works from the context alone — it never calls registry.execute.
    res = validate_intent(ParsedIntent("team_record", {"team": "lakers"}, "rule"), context=context)
    assert res.is_valid and res.validated_intent.arguments == {"team": "Los Angeles Lakers"}


def test_validator_has_no_registry_dependency() -> None:
    import src.intent_validator as iv

    assert not hasattr(iv, "DEFAULT_REGISTRY")
    assert not hasattr(iv, "registry")
    assert not hasattr(iv, "execute")


_TYPE_SAMPLES = {
    "str": "x", "str|null": "x", "int": 5, "int|null": 5, "float": 1.5, "float|null": 1.5,
    "bool": True, "bool|null": True,
}


def test_type_checker_covers_all_allowed_param_types() -> None:
    # The validator's type vocabulary must match the registry's allowlist (no drift).
    for type_str in ALLOWED_PARAM_TYPES:
        assert _type_ok(_TYPE_SAMPLES[type_str], type_str) is True, type_str
        if type_str.endswith("|null"):
            assert _type_ok(None, type_str) is True


def test_unsupported_schema_type_fails_type_check() -> None:
    assert _type_ok("anything", "decimal") is False
