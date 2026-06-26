"""Rule-parser contract objects (Phase 8A).

The shape of the deterministic rule parser's output. A query either parses into a
``ParsedIntent`` (wrapped in a ``RuleParseResult``) or produces a structured parse failure
(``no_parse`` / ``ambiguous`` / ``incomplete``). This module is contracts ONLY — no
normalisation, routing, slot extraction, validation, resolution, execution, or LLM.

The parser emits raw candidate structure; the Phase 7 validator canonicalises and protects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from src.intent_types import ParsedIntent

# --- Parse statuses ---------------------------------------------------------
PARSE_STATUS_PARSED = "parsed"
PARSE_STATUS_NO_PARSE = "no_parse"
PARSE_STATUS_AMBIGUOUS = "ambiguous"
PARSE_STATUS_INCOMPLETE = "incomplete"

PARSE_STATUSES = (
    PARSE_STATUS_PARSED,
    PARSE_STATUS_NO_PARSE,
    PARSE_STATUS_AMBIGUOUS,
    PARSE_STATUS_INCOMPLETE,
)

# --- Parse error codes ------------------------------------------------------
EMPTY_QUERY = "empty_query"
UNSUPPORTED_QUERY = "unsupported_query"
AMBIGUOUS_INTENT = "ambiguous_intent"
MISSING_TEAM = "missing_team"
MISSING_OPPONENT = "missing_opponent"
MISSING_NUMBER = "missing_number"
AMBIGUOUS_TEAM_MENTION = "ambiguous_team_mention"
UNSUPPORTED_TIME_EXPRESSION = "unsupported_time_expression"

PARSE_ERROR_CODES = (
    EMPTY_QUERY,
    UNSUPPORTED_QUERY,
    AMBIGUOUS_INTENT,
    MISSING_TEAM,
    MISSING_OPPONENT,
    MISSING_NUMBER,
    AMBIGUOUS_TEAM_MENTION,
    UNSUPPORTED_TIME_EXPRESSION,
)

# No parse-warning codes are defined yet; the slot for them exists for later phases.
PARSE_WARNING_CODES: tuple[str, ...] = ()


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


@dataclass(frozen=True)
class ParseError:
    """One structured parse error. Never carries a stack trace."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "value": _json_safe_value(self.value),
            "suggestions": list(self.suggestions),
        }


@dataclass(frozen=True)
class ParseWarning:
    """One non-blocking parse warning. Same discipline as ParseError."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "value": _json_safe_value(self.value),
            "suggestions": list(self.suggestions),
        }


@dataclass(frozen=True)
class RuleParseResult:
    """Either a parsed intent or a structured parse failure."""

    status: str
    parsed_intent: Optional[ParsedIntent] = None
    errors: tuple[ParseError, ...] = ()
    warnings: tuple[ParseWarning, ...] = ()
    raw_query: str = ""

    def __post_init__(self) -> None:
        if self.status not in PARSE_STATUSES:
            raise ValueError(f"status must be one of {PARSE_STATUSES}, got {self.status!r}.")
        if not isinstance(self.raw_query, str):
            raise TypeError("raw_query must be a string.")
        if self.parsed_intent is not None and not isinstance(self.parsed_intent, ParsedIntent):
            raise TypeError("parsed_intent must be a ParsedIntent or None.")
        errors = tuple(self.errors)
        warnings = tuple(self.warnings)
        for error in errors:
            if not isinstance(error, ParseError):
                raise TypeError("errors must contain only ParseError objects.")
        for warning in warnings:
            if not isinstance(warning, ParseWarning):
                raise TypeError("warnings must contain only ParseWarning objects.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)

        if self.status == PARSE_STATUS_PARSED:
            if self.parsed_intent is None:
                raise ValueError("a parsed result must carry a parsed_intent.")
            if errors:
                raise ValueError("a parsed result must not contain errors.")
        else:
            if self.parsed_intent is not None:
                raise ValueError("a non-parsed result must not carry a parsed_intent.")
            if not errors:
                raise ValueError("a non-parsed result must contain at least one error.")

    # --- convenience constructors ------------------------------------------

    @classmethod
    def parsed(
        cls, parsed_intent: ParsedIntent, *, raw_query: str = "",
        warnings: tuple[ParseWarning, ...] = (),
    ) -> "RuleParseResult":
        return cls(PARSE_STATUS_PARSED, parsed_intent, (), warnings, raw_query)

    @classmethod
    def no_parse(
        cls, errors: tuple[ParseError, ...], *, raw_query: str = "",
        warnings: tuple[ParseWarning, ...] = (),
    ) -> "RuleParseResult":
        return cls(PARSE_STATUS_NO_PARSE, None, errors, warnings, raw_query)

    @classmethod
    def ambiguous(
        cls, errors: tuple[ParseError, ...], *, raw_query: str = "",
        warnings: tuple[ParseWarning, ...] = (),
    ) -> "RuleParseResult":
        return cls(PARSE_STATUS_AMBIGUOUS, None, errors, warnings, raw_query)

    @classmethod
    def incomplete(
        cls, errors: tuple[ParseError, ...], *, raw_query: str = "",
        warnings: tuple[ParseWarning, ...] = (),
    ) -> "RuleParseResult":
        return cls(PARSE_STATUS_INCOMPLETE, None, errors, warnings, raw_query)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "parsed_intent": self.parsed_intent.to_dict() if self.parsed_intent is not None else None,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "raw_query": self.raw_query,
        }
