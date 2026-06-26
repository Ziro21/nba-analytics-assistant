"""Production assistant orchestration (Phase 9C).

Thin coordination only: parse, validate, execute through an injected registry, and delegate
all response mapping to the deterministic formatter. This module does not load data, build
validation context, use pandas, call analytical tools directly, or compute statistics.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.assistant_types import INTERNAL_ERROR, AssistantIssue, AssistantResult
from src.intent_validator import validate_intent
from src.response_formatter import (
    format_parse_failure,
    format_tool_result,
    format_validation_failure,
)
from src.rule_parser import parse_rule_query
from src.rule_parser_types import PARSE_STATUS_PARSED

CONFIGURATION_ERROR_MESSAGE = (
    "The assistant could not process the query because of an internal configuration issue."
)
INTERNAL_ERROR_MESSAGE = "The assistant could not process the query because of an internal error."


def _safe_query(query: object) -> str:
    return query if isinstance(query, str) else ""


def _internal_error(message: str, *, query: object = "") -> AssistantResult:
    return AssistantResult.error(
        message,
        (AssistantIssue(code=INTERNAL_ERROR, message=message),),
        query=_safe_query(query),
    )


def _dependencies_are_usable(
    *,
    clean_df: object,
    validation_context: object,
    registry: object,
) -> bool:
    if clean_df is None or validation_context is None or registry is None:
        return False
    execute = getattr(registry, "execute", None)
    return callable(execute)


def answer_query(
    query: str,
    *,
    clean_df: object,
    validation_context: object,
    registry: object,
) -> AssistantResult:
    """Answer one natural-language query using injected production dependencies.

    The caller owns data loading and context construction. This function coordinates the
    existing safe components and returns an ``AssistantResult`` for every normal and
    unexpected path.
    """
    if not isinstance(query, str):
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)
    if not _dependencies_are_usable(
        clean_df=clean_df,
        validation_context=validation_context,
        registry=registry,
    ):
        return _internal_error(CONFIGURATION_ERROR_MESSAGE, query=query)

    try:
        parse_result = parse_rule_query(query)
    except Exception:  # noqa: BLE001 - assistant boundary; fail closed for callers
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)

    try:
        if parse_result.status != PARSE_STATUS_PARSED:
            return format_parse_failure(parse_result, query=query)
    except Exception:  # noqa: BLE001 - formatter boundary; fail closed
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)

    parsed_intent = parse_result.parsed_intent
    if parsed_intent is None:
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)

    try:
        validation_result = validate_intent(parsed_intent, context=validation_context)
    except Exception:  # noqa: BLE001 - validator boundary; fail closed
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)

    try:
        if not validation_result.is_valid:
            return format_validation_failure(
                validation_result,
                query=query,
                tool_name=parsed_intent.tool_name,
            )
    except Exception:  # noqa: BLE001 - formatter boundary; fail closed
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)

    validated_intent = validation_result.validated_intent
    if validated_intent is None:
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)

    try:
        tool_result = registry.execute(
            validated_intent.tool_name,
            dict(validated_intent.arguments)
            if isinstance(validated_intent.arguments, Mapping)
            else validated_intent.arguments,
            clean_df=clean_df,
        )
    except Exception:  # noqa: BLE001 - registry boundary; fail closed
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)

    try:
        return format_tool_result(tool_result, query=query)
    except Exception:  # noqa: BLE001 - formatter boundary; fail closed
        return _internal_error(INTERNAL_ERROR_MESSAGE, query=query)
