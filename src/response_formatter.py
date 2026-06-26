"""Deterministic response formatting (Phase 9B).

Pure formatting only. This module converts already-produced parser, validator, and tool
outcomes into ``AssistantResult`` objects. It never calls the parser, validator, resolver,
registry, tools, data loaders, LLMs, or external services.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from src.assistant_types import (
    AMBIGUOUS_INTENT,
    AMBIGUOUS_TEAM,
    ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    ASSISTANT_STATUS_ERROR,
    EXECUTION_FAILED,
    INTERNAL_ERROR,
    INVALID_SPECIAL_TEAM,
    MISSING_INFORMATION,
    NO_DATA,
    PARSE_FAILED,
    SAME_TEAM_HEAD_TO_HEAD,
    UNKNOWN_TEAM,
    UNSUPPORTED_QUERY,
    VALIDATION_FAILED,
    AssistantIssue,
    AssistantResult,
)
from src.intent_types import (
    AMBIGUOUS_TEAM as VALIDATION_AMBIGUOUS_TEAM,
    INVALID_N,
    INVALID_SEASON_ID,
    INVALID_SPECIAL_TEAM as VALIDATION_INVALID_SPECIAL_TEAM,
    INVALID_WINDOW,
    MISSING_REQUIRED_ARGUMENT,
    SAME_TEAM_HEAD_TO_HEAD as VALIDATION_SAME_TEAM_HEAD_TO_HEAD,
    UNKNOWN_TEAM as VALIDATION_UNKNOWN_TEAM,
    ValidationError,
    ValidationResult,
)
from src.rule_parser_types import (
    AMBIGUOUS_INTENT as PARSE_AMBIGUOUS_INTENT,
    AMBIGUOUS_TEAM_MENTION,
    EMPTY_QUERY,
    MISSING_NUMBER,
    MISSING_OPPONENT,
    MISSING_TEAM,
    PARSE_STATUS_AMBIGUOUS,
    PARSE_STATUS_INCOMPLETE,
    PARSE_STATUS_NO_PARSE,
    PARSE_STATUS_PARSED,
    UNSUPPORTED_QUERY as PARSE_UNSUPPORTED_QUERY,
    UNSUPPORTED_TIME_EXPRESSION,
    ParseError,
    ParseWarning,
    RuleParseResult,
)

TOOL_STATUS_OK = "ok"
TOOL_STATUS_NO_DATA = "no_data"
TOOL_STATUS_ERROR = "error"
TOOL_STATUSES = frozenset({TOOL_STATUS_OK, TOOL_STATUS_NO_DATA, TOOL_STATUS_ERROR})

SUPPORTED_TOOL_NAMES = frozenset({
    "team_average_points",
    "average_points_allowed",
    "team_record",
    "top_scoring_teams",
    "head_to_head",
    "team_efficiency_summary",
})


def _format_number(value: object, *, max_decimals: int = 2, min_decimal: bool = False) -> str:
    """Stable, locale-free numeric display for user-facing messages."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
        if min_decimal and "." not in text:
            text = f"{text}.0"
        return text
    return str(value)


def _plural(value: object, singular: str, plural: str) -> str:
    return singular if value == 1 else plural


def _window_phrase(meta: Mapping[str, object], *, noun: str = "game") -> str:
    window = meta.get("window_requested")
    if window is not None:
        return f"over the last {window} {_plural(window, noun, noun + 's')}"
    return "across all available games"


def _safe_query(query: str) -> str:
    return query if isinstance(query, str) else ""


def _mapping_copy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return dict(value)


def _warnings_from_strings(items: object, *, code: str = VALIDATION_FAILED) -> tuple[AssistantIssue, ...]:
    if not isinstance(items, (list, tuple)):
        raise TypeError("warnings must be a list or tuple of strings")
    warnings: list[AssistantIssue] = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError("warnings must contain only strings")
        message = item
        warnings.append(AssistantIssue(code=code, message=message or "Formatter warning."))
    return tuple(warnings)


def _internal_error(message: str, *, query: str = "", tool_name: Optional[str] = None,
                    value: object = None) -> AssistantResult:
    return AssistantResult.error(
        message,
        (AssistantIssue(code=INTERNAL_ERROR, message=message, value=value),),
        query=_safe_query(query),
        tool_name=tool_name,
    )


def _execution_error(tool_name: str, message: str, *, query: str,
                     data: dict[str, object], meta: dict[str, object],
                     warnings: tuple[AssistantIssue, ...]) -> AssistantResult:
    safe_message = message or "Tool execution failed."
    return AssistantResult(
        ASSISTANT_STATUS_ERROR,
        "I could not prepare an answer because the tool execution failed.",
        query=_safe_query(query),
        tool_name=tool_name,
        data=data,
        errors=(AssistantIssue(code=EXECUTION_FAILED, message=safe_message),),
        warnings=warnings,
        meta=meta,
    )


def _tool_message(tool_name: str, result: Mapping[str, object],
                  meta: Mapping[str, object]) -> str:
    if tool_name == "team_average_points":
        team = result["team"]
        average = _format_number(result["average_points"], min_decimal=True)
        return f"{team} averaged {average} points {_window_phrase(meta)}."

    if tool_name == "average_points_allowed":
        team = result["team"]
        average = _format_number(result["average_points_allowed"], min_decimal=True)
        return f"{team} allowed {average} points per game {_window_phrase(meta)}."

    if tool_name == "team_record":
        team = result["team"]
        record = result["record"]
        games = result.get("games_used", meta.get("games_used"))
        window = meta.get("window_requested")
        if window is not None:
            return f"{team} are {record} over the last {window} {_plural(window, 'game', 'games')}."
        return f"{team} are {record} across {games} {_plural(games, 'game', 'games')}."

    if tool_name == "top_scoring_teams":
        teams = result["teams"]
        if not isinstance(teams, Sequence):
            raise TypeError("top_scoring_teams result must contain a teams sequence")
        parts = []
        for item in teams:
            if not isinstance(item, Mapping):
                raise TypeError("top_scoring_teams teams items must be mappings")
            rank = item["rank"]
            team = item["team"]
            average = _format_number(item["average_points"], min_decimal=True)
            parts.append(f"{rank}. {team} - {average} points per game")
        season_id = meta.get("season_id")
        prefix = "Top scoring teams"
        if season_id is not None:
            prefix = f"{prefix} in season ID {season_id}"
        return f"{prefix}: {'; '.join(parts)}."

    if tool_name == "head_to_head":
        team_a = result["team_a"]
        team_b = result["team_b"]
        record = result["record"]
        meetings = result["meetings"]
        window = meta.get("window_requested")
        if window is not None:
            return (
                f"{team_a} are {record} against {team_b} over the last {window} "
                f"{_plural(window, 'meeting', 'meetings')}."
            )
        return f"{team_a} are {record} against {team_b} across {meetings} {_plural(meetings, 'meeting', 'meetings')}."

    if tool_name == "team_efficiency_summary":
        team = result["team"]
        ortg = _format_number(result["average_ortg"])
        drtg = _format_number(result["average_drtg"])
        net = _format_number(result["average_net_rating"])
        return f"{team} {_window_phrase(meta)}: ORTG {ortg}, DRTG {drtg}, net rating {net}."

    raise KeyError(f"unsupported tool {tool_name!r}")


def format_tool_result(
    tool_result: Mapping[str, object],
    *,
    query: str = "",
) -> AssistantResult:
    """Format an already-produced tool result into an AssistantResult."""
    if not isinstance(tool_result, Mapping):
        return _internal_error("Formatter received a malformed tool result.", query=query,
                               value=tool_result)

    missing = [key for key in ("status", "tool", "result", "meta", "warnings") if key not in tool_result]
    if missing:
        return _internal_error("Formatter received a tool result missing required fields.",
                               query=query, value={"missing": missing})

    status = tool_result["status"]
    tool_name = tool_result["tool"]
    if not isinstance(status, str) or status not in TOOL_STATUSES:
        return _internal_error("Formatter received an unknown tool result status.", query=query,
                               value=status)
    if not isinstance(tool_name, str) or tool_name not in SUPPORTED_TOOL_NAMES:
        return _internal_error("Formatter received an unsupported tool result.", query=query,
                               value=tool_name)

    try:
        result = _mapping_copy(tool_result["result"])
        meta = _mapping_copy(tool_result["meta"])
    except TypeError as exc:
        return _internal_error("Formatter received a malformed tool result payload.",
                               query=query, tool_name=tool_name, value=str(exc))

    if status == TOOL_STATUS_NO_DATA:
        warning_code = NO_DATA
    elif status == TOOL_STATUS_ERROR:
        warning_code = EXECUTION_FAILED
    else:
        warning_code = VALIDATION_FAILED
    try:
        warnings = _warnings_from_strings(tool_result["warnings"], code=warning_code)
    except TypeError as exc:
        return _internal_error("Formatter received malformed tool warnings.",
                               query=query, tool_name=tool_name, value=str(exc))

    if status == TOOL_STATUS_NO_DATA:
        return AssistantResult(
            ASSISTANT_STATUS_CLARIFICATION_NEEDED,
            "I could not find matching NBA data for that request.",
            query=_safe_query(query),
            tool_name=tool_name,
            data=result,
            errors=(AssistantIssue(code=NO_DATA, message="No matching NBA data was found."),),
            warnings=warnings,
            meta=meta,
        )

    if status == TOOL_STATUS_ERROR:
        message = str(result.get("message", "Tool execution failed."))
        return _execution_error(
            tool_name, message, query=_safe_query(query), data=result, meta=meta,
            warnings=warnings,
        )

    try:
        message = _tool_message(tool_name, result, meta)
    except (KeyError, TypeError, ValueError) as exc:
        return _internal_error("Formatter could not format the tool result.",
                               query=query, tool_name=tool_name, value=str(exc))

    return AssistantResult.answer(
        message,
        query=_safe_query(query),
        tool_name=tool_name,
        data=result,
        warnings=warnings,
        meta=meta,
    )


def _assistant_issue_from_parse_error(error: ParseError) -> AssistantIssue:
    code_by_parse_code = {
        PARSE_UNSUPPORTED_QUERY: UNSUPPORTED_QUERY,
        EMPTY_QUERY: UNSUPPORTED_QUERY,
        PARSE_AMBIGUOUS_INTENT: AMBIGUOUS_INTENT,
        MISSING_TEAM: MISSING_INFORMATION,
        MISSING_OPPONENT: MISSING_INFORMATION,
        MISSING_NUMBER: MISSING_INFORMATION,
        AMBIGUOUS_TEAM_MENTION: AMBIGUOUS_INTENT,
        UNSUPPORTED_TIME_EXPRESSION: MISSING_INFORMATION,
    }
    return AssistantIssue(
        code=code_by_parse_code.get(error.code, PARSE_FAILED),
        message=error.message,
        field=error.field,
        value=error.value,
        suggestions=error.suggestions,
    )


def _assistant_warning_from_parse_warning(warning: ParseWarning) -> AssistantIssue:
    return AssistantIssue(
        code=PARSE_FAILED,
        message=warning.message,
        field=warning.field,
        value=warning.value,
        suggestions=warning.suggestions,
    )


# --- clarification-message composition (a human-readable headline from structured issues) ---
# These build the user-facing AssistantResult.message; the structured errors/suggestions are
# unchanged. Pure, deterministic, standard-library only; any unhandled case returns None so the
# caller falls back to the existing generic safe message.

def _format_suggestions(suggestions: Sequence[object]) -> str:
    """Join suggestion strings naturally: '' / 'A' / 'A or B' / 'A, B, or C'. Resolver order kept."""
    items = [s for s in suggestions if isinstance(s, str) and s]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def _select_primary_error(errors: Sequence, priority: Sequence[str]):
    """The highest-priority error by code (for the headline), else the first error, else None."""
    for code in priority:
        for error in errors:
            if error.code == code:
                return error
    return errors[0] if errors else None


_VALIDATION_PRIORITY = (
    VALIDATION_AMBIGUOUS_TEAM, VALIDATION_UNKNOWN_TEAM, VALIDATION_INVALID_SPECIAL_TEAM,
    VALIDATION_SAME_TEAM_HEAD_TO_HEAD, MISSING_REQUIRED_ARGUMENT, INVALID_WINDOW,
    INVALID_N, INVALID_SEASON_ID,
)
_PARSE_INCOMPLETE_PRIORITY = (
    MISSING_TEAM, MISSING_OPPONENT, UNSUPPORTED_TIME_EXPRESSION, MISSING_NUMBER,
)


def _message_for_validation_error(error: ValidationError) -> Optional[str]:
    """A specific user-facing headline for a validation error, or None to use the generic fallback."""
    code = error.code
    value = error.value if isinstance(error.value, str) and error.value else None
    items = [s for s in error.suggestions if isinstance(s, str) and s]
    joined = _format_suggestions(items)

    if code == VALIDATION_AMBIGUOUS_TEAM:
        if value is None:
            return None
        if len(items) == 2:
            return f'"{value}" is ambiguous. Do you mean {joined}?'
        if len(items) == 1:
            return f'"{value}" is ambiguous. Did you mean {joined}?'
        if items:
            return f'"{value}" is ambiguous. Did you mean one of: {joined}?'
        return f'"{value}" is ambiguous. Please use the full team name.'
    if code == VALIDATION_UNKNOWN_TEAM:
        if value is None:
            return None
        if len(items) >= 3:
            return f'I could not find "{value}". Did you mean one of: {joined}?'
        if items:
            return f'I could not find "{value}". Did you mean {joined}?'
        return f'I could not find "{value}". Please use a supported NBA team name.'
    if code == VALIDATION_INVALID_SPECIAL_TEAM:
        if value is not None:
            return f'"{value}" is an exhibition team, not a supported NBA franchise.'
        return "That team is an exhibition team, not a supported NBA franchise."
    if code == VALIDATION_SAME_TEAM_HEAD_TO_HEAD:
        return "A head-to-head query needs two different teams."
    if code == MISSING_REQUIRED_ARGUMENT:
        if error.field == "team_b":
            return "Please name the second team for the head-to-head."
        if error.field in ("team", "team_a"):
            return "Please tell me which team you mean."
        return None
    if code == INVALID_WINDOW:
        return 'Please use a positive whole number of games, such as "last 5 games".'
    if code == INVALID_N:
        return "Please use a positive whole number for the ranking size."
    if code == INVALID_SEASON_ID:
        return "Please use one of the supported season identifiers."
    return None


def _message_for_parse_error(error: ParseError) -> Optional[str]:
    """A specific user-facing headline for an incomplete-parse error, or None for the fallback."""
    code = error.code
    if code == MISSING_TEAM:
        return "Please tell me which team you mean."
    if code == MISSING_OPPONENT:
        return "Please name the second team for the head-to-head."
    if code == UNSUPPORTED_TIME_EXPRESSION:
        return 'Please use a specific number of games, such as "last 5 games".'
    if code == MISSING_NUMBER:
        return "Please include a number for that request."
    return None


def format_parse_failure(
    parse_result: RuleParseResult,
    *,
    query: str = "",
) -> AssistantResult:
    """Format a non-parsed RuleParseResult into an AssistantResult."""
    if not isinstance(parse_result, RuleParseResult):
        return _internal_error("Formatter received a malformed parse result.", query=query,
                               value=parse_result)
    result_query = _safe_query(query) or parse_result.raw_query
    warnings = tuple(_assistant_warning_from_parse_warning(w) for w in parse_result.warnings)

    if parse_result.status == PARSE_STATUS_PARSED:
        return _internal_error("Formatter expected a parse failure but received a parsed result.",
                               query=result_query)

    errors = tuple(_assistant_issue_from_parse_error(error) for error in parse_result.errors)
    if parse_result.status == PARSE_STATUS_NO_PARSE:
        return AssistantResult.unsupported(
            "I can only answer supported NBA analytics questions.",
            errors,
            query=result_query,
            warnings=warnings,
        )

    if parse_result.status == PARSE_STATUS_AMBIGUOUS:
        return AssistantResult.clarification_needed(
            "I need a little more detail to answer that request.",
            errors,
            query=result_query,
            warnings=warnings,
        )

    if parse_result.status == PARSE_STATUS_INCOMPLETE:
        primary = _select_primary_error(parse_result.errors, _PARSE_INCOMPLETE_PRIORITY)
        headline = None
        if primary is not None:
            try:
                headline = _message_for_parse_error(primary)
            except Exception:  # noqa: BLE001 - never let message composition break the formatter
                headline = None
        return AssistantResult.clarification_needed(
            headline or "I need more information to answer that request.",
            errors,
            query=result_query,
            warnings=warnings,
        )

    return _internal_error("Formatter received an unknown parse result status.",
                           query=result_query, value=parse_result.status)


def _assistant_issue_from_validation_error(error: ValidationError) -> AssistantIssue:
    code_by_validation_code = {
        VALIDATION_AMBIGUOUS_TEAM: AMBIGUOUS_TEAM,
        VALIDATION_UNKNOWN_TEAM: UNKNOWN_TEAM,
        VALIDATION_INVALID_SPECIAL_TEAM: INVALID_SPECIAL_TEAM,
        VALIDATION_SAME_TEAM_HEAD_TO_HEAD: SAME_TEAM_HEAD_TO_HEAD,
        MISSING_REQUIRED_ARGUMENT: MISSING_INFORMATION,
        INVALID_WINDOW: MISSING_INFORMATION,
        INVALID_N: MISSING_INFORMATION,
        INVALID_SEASON_ID: MISSING_INFORMATION,
    }
    return AssistantIssue(
        code=code_by_validation_code.get(error.code, VALIDATION_FAILED),
        message=error.message,
        field=error.field,
        value=error.value,
        suggestions=error.suggestions,
    )


def format_validation_failure(
    validation_result: ValidationResult,
    *,
    query: str = "",
    tool_name: Optional[str] = None,
) -> AssistantResult:
    """Format an invalid ValidationResult into an AssistantResult."""
    if not isinstance(validation_result, ValidationResult):
        return _internal_error("Formatter received a malformed validation result.", query=query,
                               tool_name=tool_name, value=validation_result)
    if validation_result.is_valid:
        return _internal_error("Formatter expected a validation failure but received a valid result.",
                               query=query, tool_name=tool_name)

    errors = tuple(_assistant_issue_from_validation_error(error) for error in validation_result.errors)
    warnings = tuple(
        AssistantIssue(
            code=VALIDATION_FAILED,
            message=warning.message,
            field=warning.field,
            value=warning.value,
            suggestions=warning.suggestions,
        )
        for warning in validation_result.warnings
    )
    primary = _select_primary_error(validation_result.errors, _VALIDATION_PRIORITY)
    headline = None
    if primary is not None:
        try:
            headline = _message_for_validation_error(primary)
        except Exception:  # noqa: BLE001 - never let message composition break the formatter
            headline = None
    return AssistantResult.clarification_needed(
        headline or "I need you to clarify or correct part of that request.",
        errors,
        query=_safe_query(query),
        tool_name=tool_name,
        warnings=warnings,
    )
