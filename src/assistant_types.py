"""Assistant response contracts (Phase 9A).

Contracts only: no parsing, validation, registry dispatch, tool execution, data loading,
formatting, LLM calls, or statistics. These objects define the JSON-safe shape that the
future formatter/orchestrator will return to a user-facing caller.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

# --- Assistant statuses -----------------------------------------------------

ASSISTANT_STATUS_ANSWER = "answer"
ASSISTANT_STATUS_CLARIFICATION_NEEDED = "clarification_needed"
ASSISTANT_STATUS_UNSUPPORTED = "unsupported"
ASSISTANT_STATUS_ERROR = "error"

ASSISTANT_STATUSES = (
    ASSISTANT_STATUS_ANSWER,
    ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    ASSISTANT_STATUS_UNSUPPORTED,
    ASSISTANT_STATUS_ERROR,
)

# --- Assistant issue codes --------------------------------------------------

PARSE_FAILED = "parse_failed"
UNSUPPORTED_QUERY = "unsupported_query"
AMBIGUOUS_INTENT = "ambiguous_intent"
MISSING_INFORMATION = "missing_information"
VALIDATION_FAILED = "validation_failed"
AMBIGUOUS_TEAM = "ambiguous_team"
UNKNOWN_TEAM = "unknown_team"
INVALID_SPECIAL_TEAM = "invalid_special_team"
SAME_TEAM_HEAD_TO_HEAD = "same_team_head_to_head"
SAME_TEAM_COMPARISON = "same_team_comparison"
NO_DATA = "no_data"
EXECUTION_FAILED = "execution_failed"
INTERNAL_ERROR = "internal_error"

ASSISTANT_ISSUE_CODES = (
    PARSE_FAILED,
    UNSUPPORTED_QUERY,
    AMBIGUOUS_INTENT,
    MISSING_INFORMATION,
    VALIDATION_FAILED,
    AMBIGUOUS_TEAM,
    UNKNOWN_TEAM,
    INVALID_SPECIAL_TEAM,
    SAME_TEAM_HEAD_TO_HEAD,
    SAME_TEAM_COMPARISON,
    NO_DATA,
    EXECUTION_FAILED,
    INTERNAL_ERROR,
)


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-safe copy of ``value``; non-serialisable leaves become ``repr``."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, MappingProxyType):
        return _json_safe_mapping(value)
    if isinstance(value, dict):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]

    try:
        json.dumps(value)
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return repr(value)


def _json_safe_mapping(value: Any) -> dict[str, Any]:
    """Stringify mapping keys for JSON, rejecting collisions after stringification."""
    safe: dict[str, Any] = {}
    for key, item in value.items():
        safe_key = str(key)
        if safe_key in safe:
            raise ValueError(f"mapping contains duplicate JSON key after stringification: {safe_key!r}")
        safe[safe_key] = _json_safe_value(item)
    return safe


def _deep_freeze_json_safe_dict(value: Optional[dict[str, object]], field_name: str):
    """Validate a dict-or-None field and store an immutable JSON-safe deep copy."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict or None.")
    return _deep_freeze(_json_safe_value(value))


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze dict/list containers after JSON-safety normalisation."""
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _to_plain(value: Any) -> Any:
    """Convert frozen containers back to plain JSON-safe dict/list structures."""
    if isinstance(value, MappingProxyType):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    return copy.deepcopy(value)


def _coerce_suggestions(suggestions: Any) -> tuple[str, ...]:
    if isinstance(suggestions, str):
        raise TypeError("suggestions must be a sequence of strings, not a string.")
    coerced = tuple(suggestions)
    for item in coerced:
        if not isinstance(item, str):
            raise TypeError("suggestions must contain only strings.")
    return coerced


def _coerce_issues(items: Any, *, field_name: str) -> tuple["AssistantIssue", ...]:
    if isinstance(items, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of AssistantIssue objects.")
    coerced = tuple(items)
    for item in coerced:
        if not isinstance(item, AssistantIssue):
            raise TypeError(f"{field_name} must contain only AssistantIssue objects.")
    return coerced


@dataclass(frozen=True)
class AssistantIssue:
    """One user-facing assistant issue or warning. Never exposes stack traces."""

    code: str
    message: str
    field: Optional[str] = None
    value: Any = None
    suggestions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("code must be a non-empty string.")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string.")
        if self.field is not None and not isinstance(self.field, str):
            raise TypeError("field must be None or a string.")
        object.__setattr__(self, "suggestions", _coerce_suggestions(self.suggestions))
        object.__setattr__(self, "value", _deep_freeze(_json_safe_value(self.value)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "value": _to_plain(self.value),
            "suggestions": list(self.suggestions),
        }


@dataclass(frozen=True)
class AssistantResult:
    """Structured assistant-layer response, before any transport/UI concerns."""

    status: str
    message: str
    query: str = ""
    tool_name: Optional[str] = None
    data: Optional[dict[str, object]] = None
    errors: tuple[AssistantIssue, ...] = ()
    warnings: tuple[AssistantIssue, ...] = ()
    meta: Optional[dict[str, object]] = None

    def __post_init__(self) -> None:
        if self.status not in ASSISTANT_STATUSES:
            raise ValueError(f"status must be one of {ASSISTANT_STATUSES}, got {self.status!r}.")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string.")
        if not isinstance(self.query, str):
            raise TypeError("query must be a string.")
        if self.tool_name is not None and not isinstance(self.tool_name, str):
            raise TypeError("tool_name must be None or a string.")

        errors = _coerce_issues(self.errors, field_name="errors")
        warnings = _coerce_issues(self.warnings, field_name="warnings")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "data", _deep_freeze_json_safe_dict(self.data, "data"))
        object.__setattr__(self, "meta", _deep_freeze_json_safe_dict(self.meta, "meta"))

        if self.status == ASSISTANT_STATUS_ANSWER:
            if errors:
                raise ValueError("an answer result must not contain errors.")
        elif not errors:
            raise ValueError(f"a {self.status!r} result must contain at least one error.")

    @classmethod
    def answer(
        cls,
        message: str,
        *,
        query: str = "",
        tool_name: Optional[str] = None,
        data: Optional[dict[str, object]] = None,
        warnings: tuple[AssistantIssue, ...] = (),
        meta: Optional[dict[str, object]] = None,
    ) -> "AssistantResult":
        return cls(
            ASSISTANT_STATUS_ANSWER, message, query=query, tool_name=tool_name,
            data=data, errors=(), warnings=warnings, meta=meta,
        )

    @classmethod
    def clarification_needed(
        cls,
        message: str,
        errors: tuple[AssistantIssue, ...],
        *,
        query: str = "",
        tool_name: Optional[str] = None,
        warnings: tuple[AssistantIssue, ...] = (),
        meta: Optional[dict[str, object]] = None,
    ) -> "AssistantResult":
        return cls(
            ASSISTANT_STATUS_CLARIFICATION_NEEDED, message, query=query, tool_name=tool_name,
            errors=errors, warnings=warnings, meta=meta,
        )

    @classmethod
    def unsupported(
        cls,
        message: str,
        errors: tuple[AssistantIssue, ...],
        *,
        query: str = "",
        warnings: tuple[AssistantIssue, ...] = (),
        meta: Optional[dict[str, object]] = None,
    ) -> "AssistantResult":
        return cls(
            ASSISTANT_STATUS_UNSUPPORTED, message, query=query, tool_name=None,
            errors=errors, warnings=warnings, meta=meta,
        )

    @classmethod
    def error(
        cls,
        message: str,
        errors: tuple[AssistantIssue, ...],
        *,
        query: str = "",
        tool_name: Optional[str] = None,
        warnings: tuple[AssistantIssue, ...] = (),
        meta: Optional[dict[str, object]] = None,
    ) -> "AssistantResult":
        return cls(
            ASSISTANT_STATUS_ERROR, message, query=query, tool_name=tool_name,
            errors=errors, warnings=warnings, meta=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "query": self.query,
            "tool_name": self.tool_name,
            "data": _to_plain(self.data) if self.data is not None else None,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "meta": _to_plain(self.meta) if self.meta is not None else None,
        }
