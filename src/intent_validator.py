"""Shared intent validator (Phase 7C) — the safety boundary before tool execution.

`validate_intent(intent, *, context)` takes a `ParsedIntent` (from either parser mode) and a
pre-built `ValidationContext`, and returns a `ValidationResult`: a canonicalised
`ValidatedIntent` ready for the registry, or structured `ValidationError`s.

It is schema-driven (required/allowed args and types come from the registry schemas in the
context), parser-mode invariant (mode is metadata only), and executes nothing — no registry
call, no tool, no data load, no statistics, no LLM/parser/formatter imports.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.intent_types import (
    AMBIGUOUS_TEAM,
    ARGUMENTS_NOT_DICT,
    INVALID_ARGUMENT_TYPE,
    INVALID_LOCATION,
    INVALID_N,
    INVALID_PARSER_MODE,
    INVALID_SEASON_ID,
    INVALID_SPECIAL_TEAM,
    INVALID_WINDOW,
    MISSING_REQUIRED_ARGUMENT,
    PARSER_MODES,
    SAME_TEAM_COMPARISON,
    SAME_TEAM_HEAD_TO_HEAD,
    SEVERITY_WARNING,
    UNEXPECTED_ARGUMENT,
    UNKNOWN_TEAM,
    UNKNOWN_TOOL,
    ParsedIntent,
    ValidatedIntent,
    ValidationError,
    ValidationResult,
)
from src.team_resolution import (
    TEAM_AMBIGUOUS,
    TEAM_INVALID_SPECIAL,
    TEAM_RESOLVED,
    TEAM_UNKNOWN,
    resolve_team_name,
)
from src.validation_context import ValidationContext

TEAM_ARG_NAMES = frozenset({"team", "team_a", "team_b"})
CANONICALISED_TEAM = "canonicalised_team"  # non-blocking transparency warning code

# Two-team tools that require team_a and team_b to resolve to DIFFERENT franchises, each with its
# own error code so the failure reads correctly for that tool.
SAME_TEAM_CODE_BY_TOOL = {
    "head_to_head": SAME_TEAM_HEAD_TO_HEAD,
    "compare_team_profiles": SAME_TEAM_COMPARISON,
}


def _type_ok(value: object, type_str: str) -> bool:
    """Check a value against a JSON-safe schema type string. No coercion; bool only for bool."""
    nullable = type_str.endswith("|null")
    base = type_str[:-len("|null")] if nullable else type_str
    if value is None:
        return nullable
    if base == "str":
        return isinstance(value, str)
    if base == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if base == "float":
        return isinstance(value, float) and not isinstance(value, bool)
    if base == "bool":
        return isinstance(value, bool)
    return False  # unknown/unsupported type string


def validate_intent(intent: ParsedIntent, *, context: ValidationContext) -> ValidationResult:
    """Validate and canonicalise a ParsedIntent against the reference context."""
    if not isinstance(intent, ParsedIntent):
        raise TypeError("intent must be a ParsedIntent.")
    if not isinstance(context, ValidationContext):
        raise TypeError("context must be a ValidationContext.")

    # Parser mode (defensive — ParsedIntent already constrains it; metadata only).
    if intent.parser_mode not in PARSER_MODES:
        return ValidationResult.invalid(errors=(ValidationError(
            code=INVALID_PARSER_MODE,
            message=f"Unknown parser mode {intent.parser_mode!r}.",
            field="parser_mode", value=intent.parser_mode,
        ),))

    # Tool must be registered (stop here — no schema otherwise).
    tool_name = intent.tool_name
    if tool_name not in context.registered_tools:
        return ValidationResult.invalid(errors=(ValidationError(
            code=UNKNOWN_TOOL,
            message=f"Unknown tool {tool_name!r}.",
            field="tool_name", value=tool_name,
            suggestions=tuple(context.registered_tools),
        ),))

    schema = context.tool_schema_by_name[tool_name]
    parameters = schema["parameters"]
    allowed = {p["name"] for p in parameters}
    required = {p["name"] for p in parameters if p["required"]}
    type_by_name = {p["name"]: p["type"] for p in parameters}

    # Arguments must be a mapping (stop here — inspection unsafe otherwise).
    args = intent.arguments
    if not isinstance(args, Mapping):
        return ValidationResult.invalid(errors=(ValidationError(
            code=ARGUMENTS_NOT_DICT,
            message="Arguments must be a dictionary.",
            field="arguments", value=args,
        ),))

    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    provided = set(args.keys())

    # Unexpected / missing arguments (accumulate).
    for name in sorted(provided - allowed):
        errors.append(ValidationError(
            code=UNEXPECTED_ARGUMENT, message=f"Unexpected argument {name!r}.",
            field=name, value=args[name],
        ))
    for name in sorted(required - provided):
        errors.append(ValidationError(
            code=MISSING_REQUIRED_ARGUMENT, message=f"Missing required argument {name!r}.",
            field=name,
        ))

    # Type validation for known provided args (accumulate); record which passed.
    type_ok: dict[str, bool] = {}
    for name in sorted(provided & allowed):
        ok = _type_ok(args[name], type_by_name[name])
        type_ok[name] = ok
        if not ok:
            errors.append(ValidationError(
                code=INVALID_ARGUMENT_TYPE,
                message=f"Argument {name!r} expected type {type_by_name[name]!r}.",
                field=name, value=args[name],
            ))

    canonical_args = dict(args)
    resolved_teams: dict[str, str] = {}

    # Team canonicalisation (only for type-valid team args; no cascade on type errors).
    for name in sorted(TEAM_ARG_NAMES & provided & allowed):
        if not type_ok.get(name, False):
            continue
        raw_value = args[name]
        result = resolve_team_name(
            raw_value,
            canonical_teams=context.canonical_teams,
            special_teams=context.special_teams,
            alias_map=context.alias_map,
            ambiguity_map=context.ambiguity_map,
        )
        if result.status == TEAM_RESOLVED:
            canonical_args[name] = result.canonical_name
            resolved_teams[name] = result.canonical_name
            if result.canonical_name != raw_value:
                warnings.append(ValidationError(
                    code=CANONICALISED_TEAM,
                    message=f"Interpreted {raw_value!r} as {result.canonical_name!r}.",
                    field=name, value=raw_value,
                    suggestions=(result.canonical_name,), severity=SEVERITY_WARNING,
                ))
        elif result.status == TEAM_UNKNOWN:
            errors.append(ValidationError(
                code=UNKNOWN_TEAM, message=result.message or f"Unknown team {raw_value!r}.",
                field=name, value=raw_value, suggestions=result.suggestions,
            ))
        elif result.status == TEAM_AMBIGUOUS:
            errors.append(ValidationError(
                code=AMBIGUOUS_TEAM, message=result.message or f"Ambiguous team {raw_value!r}.",
                field=name, value=raw_value, suggestions=result.suggestions,
            ))
        elif result.status == TEAM_INVALID_SPECIAL:
            errors.append(ValidationError(
                code=INVALID_SPECIAL_TEAM,
                message=result.message or f"{raw_value!r} is not a supported franchise team.",
                field=name, value=raw_value,
            ))

    # Domain checks (only for type-valid args; no cascade on type errors).
    if "window" in provided and "window" in allowed and type_ok.get("window", False):
        window = args["window"]
        if window is not None and window <= 0:
            errors.append(ValidationError(
                code=INVALID_WINDOW, message=f"window must be a positive integer, got {window!r}.",
                field="window", value=window,
            ))
    if "n" in provided and "n" in allowed and type_ok.get("n", False):
        n_value = args["n"]
        if n_value <= 0:
            errors.append(ValidationError(
                code=INVALID_N, message=f"n must be a positive integer, got {n_value!r}.",
                field="n", value=n_value,
            ))
    if "season_id" in provided and "season_id" in allowed and type_ok.get("season_id", False):
        season_id = args["season_id"]
        if season_id is not None and season_id not in context.valid_season_ids:
            errors.append(ValidationError(
                code=INVALID_SEASON_ID,
                message=f"season_id {season_id!r} is not a known season identifier.",
                field="season_id", value=season_id,
                suggestions=tuple(str(s) for s in context.valid_season_ids),
            ))
    if "location" in provided and "location" in allowed and type_ok.get("location", False):
        location = args["location"]
        if location is not None and location not in ("home", "away"):
            errors.append(ValidationError(
                code=INVALID_LOCATION,
                message=f"location must be 'home' or 'away', got {location!r}.",
                field="location", value=location, suggestions=("home", "away"),
            ))

    # Two-team same-team check (head_to_head / compare_team_profiles) — only when both teams
    # resolved (no cascade), using the per-tool error code.
    if tool_name in SAME_TEAM_CODE_BY_TOOL and "team_a" in resolved_teams and "team_b" in resolved_teams:
        if resolved_teams["team_a"] == resolved_teams["team_b"]:
            errors.append(ValidationError(
                code=SAME_TEAM_CODE_BY_TOOL[tool_name],
                message="team_a and team_b must be two different teams.",
                field="team_b", value=args["team_b"],
            ))

    if errors:
        return ValidationResult.invalid(errors=tuple(errors), warnings=tuple(warnings))

    validated = ValidatedIntent(
        tool_name=tool_name,
        arguments=canonical_args,
        parser_mode=intent.parser_mode,
        raw_query=intent.raw_query,
        warnings=tuple(warnings),
    )
    return ValidationResult.valid(validated, warnings=tuple(warnings))
