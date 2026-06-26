"""Deterministic rule parser assembly (Phase 8D).

One public function, ``parse_rule_query``, composing the Phase 8B router and the Phase 8C slot
extractor into a ``RuleParseResult``. It builds a ``ParsedIntent(parser_mode="rule")`` ONLY when
routing and extraction both succeed. It performs no validation, no canonicalisation, no resolver
call, no registry execution, no data loading, no statistics — it just assembles candidate
structure for the Phase 7 validator (Phase 8E) to canonicalise and protect.

Status mapping (route -> slot -> parse):
  route no_route                  -> RuleParseResult.no_parse   (e.g. empty_query / unsupported_query)
  route ambiguous                 -> RuleParseResult.ambiguous  (ambiguous_intent)
  routed + slot extracted         -> RuleParseResult.parsed
  routed + slot incomplete        -> RuleParseResult.incomplete (missing_team / missing_opponent /
                                                                  unsupported_time_expression)
  routed + slot unsupported       -> RuleParseResult.ambiguous  (ambiguous_team_mention: two teams
                                     for a single-team tool — consistent with generic-compare
                                     ambiguity; the unknown-tool case cannot occur because routing
                                     only ever emits a registered tool)
"""

from __future__ import annotations

from src.intent_types import ParsedIntent
from src.rule_intent_router import (
    ROUTE_STATUS_AMBIGUOUS,
    ROUTE_STATUS_NO_ROUTE,
    ROUTE_STATUS_ROUTED,
    route_intent,
)
from src.rule_parser_types import UNSUPPORTED_QUERY, ParseError, RuleParseResult
from src.rule_slot_extractor import (
    SLOT_STATUS_EXTRACTED,
    SLOT_STATUS_INCOMPLETE,
    SLOT_STATUS_UNSUPPORTED,
    extract_slots,
)


def parse_rule_query(query: str) -> RuleParseResult:
    """Parse a natural-language query into a ``RuleParseResult`` deterministically.

    Pure and offline: the same input always yields the same result. Raises ``TypeError`` for
    non-string input. Never validates, resolves, or executes.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string.")

    route = route_intent(query)
    if route.status == ROUTE_STATUS_NO_ROUTE:
        return RuleParseResult.no_parse(route.errors, raw_query=query, warnings=route.warnings)
    if route.status == ROUTE_STATUS_AMBIGUOUS:
        return RuleParseResult.ambiguous(route.errors, raw_query=query, warnings=route.warnings)
    if route.status != ROUTE_STATUS_ROUTED:  # unreachable (router status is validated); fail safe
        return RuleParseResult.no_parse(
            (ParseError(UNSUPPORTED_QUERY, f"Unexpected route status {route.status!r}."),),
            raw_query=query,
        )

    slots = extract_slots(query, tool_name=route.tool_name)
    if slots.status == SLOT_STATUS_INCOMPLETE:
        return RuleParseResult.incomplete(slots.errors, raw_query=query, warnings=slots.warnings)
    if slots.status == SLOT_STATUS_UNSUPPORTED:
        # Reachable case: two team mentions for a single-team tool (ambiguous_team_mention).
        return RuleParseResult.ambiguous(slots.errors, raw_query=query, warnings=slots.warnings)
    if slots.status != SLOT_STATUS_EXTRACTED:  # unreachable (slot status is validated); fail safe
        return RuleParseResult.no_parse(
            (ParseError(UNSUPPORTED_QUERY, f"Unexpected slot status {slots.status!r}."),),
            raw_query=query,
        )

    parsed_intent = ParsedIntent(
        tool_name=route.tool_name,
        arguments=dict(slots.arguments),  # plain copy; slot arguments are not mutated
        parser_mode="rule",
        raw_query=query,
        confidence=None,
    )
    return RuleParseResult.parsed(
        parsed_intent, raw_query=query,
        warnings=tuple(route.warnings) + tuple(slots.warnings),  # never drop component warnings
    )
