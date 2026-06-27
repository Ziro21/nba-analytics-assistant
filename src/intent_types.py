"""Shared intent and validation contract objects (Phase 7A).

These are the JSON-friendly data structures that future parsers (rule and LLM) and the
shared validator (Phase 7C) will use. This module contains contracts ONLY — no validation
logic, no team resolution, no registry access, no pandas, no data loading.

Immutability note: frozen dataclasses do not deep-freeze contained dicts/lists, so each
contract defensively copies mutable inputs on construction and exposes JSON-serialisable
copies via ``to_dict()``. User-provided dictionaries/lists are never mutated.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

# --- Parser modes -----------------------------------------------------------
PARSER_MODE_RULE = "rule"
PARSER_MODE_LLM = "llm"
PARSER_MODES = (PARSER_MODE_RULE, PARSER_MODE_LLM)

# --- Severities -------------------------------------------------------------
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITIES = (SEVERITY_ERROR, SEVERITY_WARNING)

# --- Validation error codes -------------------------------------------------
# Tool / argument-shape
UNKNOWN_TOOL = "unknown_tool"
ARGUMENTS_NOT_DICT = "arguments_not_dict"
MISSING_REQUIRED_ARGUMENT = "missing_required_argument"
UNEXPECTED_ARGUMENT = "unexpected_argument"
INVALID_ARGUMENT_TYPE = "invalid_argument_type"
INVALID_PARSER_MODE = "invalid_parser_mode"
# Team
UNKNOWN_TEAM = "unknown_team"
AMBIGUOUS_TEAM = "ambiguous_team"
INVALID_SPECIAL_TEAM = "invalid_special_team"
# Domain arguments
INVALID_WINDOW = "invalid_window"
INVALID_N = "invalid_n"
INVALID_SEASON_ID = "invalid_season_id"
INVALID_LOCATION = "invalid_location"
SAME_TEAM_HEAD_TO_HEAD = "same_team_head_to_head"

# Canonical catalogue of the validation error codes defined above.
ERROR_CODES = (
    UNKNOWN_TOOL, ARGUMENTS_NOT_DICT, MISSING_REQUIRED_ARGUMENT, UNEXPECTED_ARGUMENT,
    INVALID_ARGUMENT_TYPE, INVALID_PARSER_MODE, UNKNOWN_TEAM, AMBIGUOUS_TEAM,
    INVALID_SPECIAL_TEAM, INVALID_WINDOW, INVALID_N, INVALID_SEASON_ID, INVALID_LOCATION,
    SAME_TEAM_HEAD_TO_HEAD,
)


# --- Helpers ----------------------------------------------------------------

def _validate_confidence(value: Any) -> None:
    """Confidence must be None, or a non-bool number in [0, 1]."""
    if value is None:
        return
    if isinstance(value, bool):
        raise TypeError("confidence must not be a bool.")
    if not isinstance(value, (int, float)):
        raise TypeError("confidence must be None or a number.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("confidence must be between 0 and 1 inclusive.")


def _prepare_arguments(arguments: Any) -> MappingProxyType:
    """Validate and freeze tool arguments.

    Arguments must be a JSON-serialisable dict (a non-JSON value is a parser bug and is
    rejected fail-fast). Returns a deep-copied, read-only mapping so the contract cannot be
    mutated after construction and never aliases the caller's dict.
    """
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be a dict.")
    try:
        json.dumps(arguments)
    except (TypeError, ValueError) as exc:
        raise ValueError("arguments must be JSON-serialisable.") from exc
    return MappingProxyType(copy.deepcopy(arguments))


def _json_safe_value(value: Any) -> Any:
    """Return ``value`` if JSON-serialisable, else a safe string representation."""
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _coerce_suggestions(suggestions: Any) -> tuple[str, ...]:
    if isinstance(suggestions, str):
        raise TypeError("suggestions must be a sequence of strings, not a string.")
    coerced = tuple(suggestions)
    for item in coerced:
        if not isinstance(item, str):
            raise TypeError("suggestions must contain only strings.")
    return coerced


def _coerce_validation_errors(
    items: Any, *, expected_severity: str, field_name: str
) -> tuple["ValidationError", ...]:
    if isinstance(items, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of ValidationError objects.")
    coerced = tuple(items)
    for item in coerced:
        if not isinstance(item, ValidationError):
            raise TypeError(f"{field_name} must contain only ValidationError objects.")
        if item.severity != expected_severity:
            raise ValueError(
                f"{field_name} must contain only {expected_severity!r}-severity items."
            )
    return coerced


# --- Contract objects -------------------------------------------------------

@dataclass(frozen=True)
class ParsedIntent:
    """Structured parser output, before validation.

    ``parser_mode`` and ``confidence`` are metadata only and must never make validation
    stronger or weaker.
    """

    tool_name: str
    arguments: dict
    parser_mode: str
    raw_query: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("tool_name must be a non-empty string.")
        if self.parser_mode not in PARSER_MODES:
            raise ValueError(f"parser_mode must be one of {PARSER_MODES}, got {self.parser_mode!r}.")
        if self.raw_query is not None and not isinstance(self.raw_query, str):
            raise TypeError("raw_query must be None or a string.")
        _validate_confidence(self.confidence)
        object.__setattr__(self, "arguments", _prepare_arguments(self.arguments))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "parser_mode": self.parser_mode,
            "raw_query": self.raw_query,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ValidationError:
    """One structured validation error or warning. Never carries a stack trace."""

    code: str
    message: str
    field: Optional[str] = None
    value: Any = None
    suggestions: tuple[str, ...] = ()
    severity: str = SEVERITY_ERROR

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("code must be a non-empty string.")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string.")
        if self.field is not None and not isinstance(self.field, str):
            raise TypeError("field must be None or a string.")
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}.")
        object.__setattr__(self, "suggestions", _coerce_suggestions(self.suggestions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "value": _json_safe_value(self.value),
            "suggestions": list(self.suggestions),
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ValidatedIntent:
    """A canonicalised, registry-ready intent after successful validation."""

    tool_name: str
    arguments: dict
    parser_mode: str
    raw_query: Optional[str] = None
    warnings: tuple[ValidationError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("tool_name must be a non-empty string.")
        if self.parser_mode not in PARSER_MODES:
            raise ValueError(f"parser_mode must be one of {PARSER_MODES}, got {self.parser_mode!r}.")
        if self.raw_query is not None and not isinstance(self.raw_query, str):
            raise TypeError("raw_query must be None or a string.")
        object.__setattr__(self, "arguments", _prepare_arguments(self.arguments))
        object.__setattr__(
            self,
            "warnings",
            _coerce_validation_errors(
                self.warnings, expected_severity=SEVERITY_WARNING, field_name="warnings"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "parser_mode": self.parser_mode,
            "raw_query": self.raw_query,
            "warnings": [w.to_dict() for w in self.warnings],
        }


@dataclass(frozen=True)
class ValidationResult:
    """Either a valid result with a ``ValidatedIntent``, or structured errors."""

    is_valid: bool
    validated_intent: Optional[ValidatedIntent]
    errors: tuple[ValidationError, ...] = ()
    warnings: tuple[ValidationError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.is_valid, bool):
            raise TypeError("is_valid must be a bool.")
        if self.validated_intent is not None and not isinstance(
            self.validated_intent, ValidatedIntent
        ):
            raise TypeError("validated_intent must be a ValidatedIntent or None.")
        object.__setattr__(
            self,
            "errors",
            _coerce_validation_errors(
                self.errors, expected_severity=SEVERITY_ERROR, field_name="errors"
            ),
        )
        object.__setattr__(
            self,
            "warnings",
            _coerce_validation_errors(
                self.warnings, expected_severity=SEVERITY_WARNING, field_name="warnings"
            ),
        )
        if self.is_valid:
            if self.validated_intent is None:
                raise ValueError("a valid result must carry a validated_intent.")
            if self.errors:
                raise ValueError("a valid result must not contain errors.")
        else:
            if self.validated_intent is not None:
                raise ValueError("an invalid result must not carry a validated_intent.")
            if not self.errors:
                raise ValueError("an invalid result must contain at least one error.")

    @classmethod
    def valid(
        cls, validated_intent: ValidatedIntent, warnings: tuple[ValidationError, ...] = ()
    ) -> "ValidationResult":
        return cls(is_valid=True, validated_intent=validated_intent, errors=(), warnings=warnings)

    @classmethod
    def invalid(
        cls,
        errors: tuple[ValidationError, ...],
        warnings: tuple[ValidationError, ...] = (),
    ) -> "ValidationResult":
        return cls(is_valid=False, validated_intent=None, errors=errors, warnings=warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "validated_intent": (
                self.validated_intent.to_dict() if self.validated_intent is not None else None
            ),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }
