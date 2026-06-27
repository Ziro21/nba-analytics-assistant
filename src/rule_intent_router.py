"""Deterministic intent routing for the rule parser (Phase 8B).

Maps a query to ONE candidate tool name, or returns a structured routing failure. Routing is
intent detection ONLY — it does not extract slots (team/team_a/team_b/window/n/season_id), build
ParsedIntent, validate, resolve, or execute. Missing operands (e.g. "Celtics vs") are a slot
concern for Phase 8C, so such a query still ROUTES here (clear h2h intent) and fails later.

Priority (first match wins): head_to_head > average_points_allowed > team_efficiency_summary >
team_record > top_scoring_teams > team_average_points. h2h beats record (an explicit "against"/
"vs" outranks generic "record"); points-allowed beats generic average points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.rule_parser_types import (
    AMBIGUOUS_INTENT,
    EMPTY_QUERY,
    UNSUPPORTED_QUERY,
    ParseError,
    ParseWarning,
)
from src.rule_query_normalisation import normalise_query_text

# The routable tools (must mirror the registry / Phase 8A SUPPORTED_TOOL_NAMES).
ROUTABLE_TOOL_NAMES = (
    "team_average_points",
    "average_points_allowed",
    "team_record",
    "top_scoring_teams",
    "head_to_head",
    "team_efficiency_summary",
    "team_advanced_profile",
)

ROUTE_STATUS_ROUTED = "routed"
ROUTE_STATUS_NO_ROUTE = "no_route"
ROUTE_STATUS_AMBIGUOUS = "ambiguous"
ROUTE_STATUSES = (ROUTE_STATUS_ROUTED, ROUTE_STATUS_NO_ROUTE, ROUTE_STATUS_AMBIGUOUS)

# --- Signal vocabularies (matched as whole space-delimited phrases) ---------
# Strong head-to-head signals: unambiguous "two teams playing each other" language.
STRONG_H2H_SIGNALS = ("vs", "versus", "h2h", "head to head", "matchup", "record against")
# Bare "against" is also h2h, EXCEPT in a defensive "points against" phrasing (= points allowed).
DEFENSIVE_AGAINST_SIGNALS = ("points against",)

ALLOWED_SIGNALS = (
    "points allowed", "points against", "allowed", "allow", "allowing",
    "concede", "conceded", "concedes", "conceding", "give up", "giving up", "given up",
)
EFFICIENCY_SIGNALS = (
    "efficiency", "offensive rating", "defensive rating", "net rating", "ortg", "drtg",
)
RECORD_SIGNALS = ("record", "win loss", "wins", "losses")
TOP_SCORING_SIGNALS = (
    "scoring teams", "highest scoring", "top scoring",
    "best offences by points", "best offenses by points",
    "offences by points", "offenses by points",
    "top teams by points", "highest points teams",
)
AVERAGE_POINTS_SIGNALS = (
    "average points", "points average", "averaging", "how many points", "points scored",
)

# Broad performance/profile language: a holistic "how are they doing" question, not a single metric.
PROFILE_SIGNALS = (
    "advanced profile", "performance profile", "profile",
    "performing", "performance", "doing",
    "summarise", "summarize",
    "offense and defense", "offence and defence",
)

# Generic comparison keyword: ambiguous unless a metric/h2h/profile signal is also present.
COMPARE_SIGNAL = "compare"


@dataclass(frozen=True)
class IntentRouteResult:
    """The outcome of routing: a candidate tool, or a structured routing failure."""

    status: str
    tool_name: Optional[str] = None
    errors: tuple[ParseError, ...] = ()
    warnings: tuple[ParseWarning, ...] = ()
    raw_query: str = ""
    normalised_query: str = ""
    matched_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ROUTE_STATUSES:
            raise ValueError(f"status must be one of {ROUTE_STATUSES}, got {self.status!r}.")
        if not isinstance(self.raw_query, str):
            raise TypeError("raw_query must be a string.")
        if not isinstance(self.normalised_query, str):
            raise TypeError("normalised_query must be a string.")
        errors = tuple(self.errors)
        warnings = tuple(self.warnings)
        signals = tuple(self.matched_signals)
        for error in errors:
            if not isinstance(error, ParseError):
                raise TypeError("errors must contain only ParseError objects.")
        for warning in warnings:
            if not isinstance(warning, ParseWarning):
                raise TypeError("warnings must contain only ParseWarning objects.")
        for signal in signals:
            if not isinstance(signal, str):
                raise TypeError("matched_signals must contain only strings.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "matched_signals", signals)

        if self.status == ROUTE_STATUS_ROUTED:
            if self.tool_name not in ROUTABLE_TOOL_NAMES:
                raise ValueError("a routed result must name a routable tool.")
            if errors:
                raise ValueError("a routed result must not contain errors.")
        else:
            if self.tool_name is not None:
                raise ValueError("a non-routed result must not name a tool.")
            if not errors:
                raise ValueError("a non-routed result must contain at least one error.")

    @classmethod
    def routed(cls, tool_name, *, raw_query="", normalised_query="", matched_signals=(),
               warnings=()) -> "IntentRouteResult":
        return cls(ROUTE_STATUS_ROUTED, tool_name, (), warnings, raw_query, normalised_query,
                   matched_signals)

    @classmethod
    def no_route(cls, errors, *, raw_query="", normalised_query="", warnings=()) -> "IntentRouteResult":
        return cls(ROUTE_STATUS_NO_ROUTE, None, errors, warnings, raw_query, normalised_query, ())

    @classmethod
    def ambiguous(cls, errors, *, raw_query="", normalised_query="", warnings=()) -> "IntentRouteResult":
        return cls(ROUTE_STATUS_AMBIGUOUS, None, errors, warnings, raw_query, normalised_query, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "tool_name": self.tool_name,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "raw_query": self.raw_query,
            "normalised_query": self.normalised_query,
            "matched_signals": list(self.matched_signals),
        }


def _present(padded: str, signal: str) -> bool:
    """Whole-phrase containment against a space-padded normalised query."""
    return f" {signal} " in padded


def _matches(padded: str, signals: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(s for s in signals if _present(padded, s))


def route_intent(query: str) -> IntentRouteResult:
    """Route a raw query to one candidate tool name, or a structured routing failure."""
    normalised = normalise_query_text(query)  # raises TypeError on non-str input
    if not normalised:
        return IntentRouteResult.no_route(
            (ParseError(EMPTY_QUERY, "The query is empty."),),
            raw_query=query, normalised_query=normalised,
        )

    padded = f" {normalised} "

    def _routed(tool: str, signals: tuple[str, ...]) -> IntentRouteResult:
        return IntentRouteResult.routed(
            tool, raw_query=query, normalised_query=normalised, matched_signals=signals,
        )

    # 1. head_to_head — strong signals, or a bare "against" that is not "points against".
    strong = _matches(padded, STRONG_H2H_SIGNALS)
    if strong:
        return _routed("head_to_head", strong)
    if _present(padded, "against") and not _matches(padded, DEFENSIVE_AGAINST_SIGNALS):
        return _routed("head_to_head", ("against",))

    # 2. average_points_allowed — points conceded language beats generic average points.
    allowed = _matches(padded, ALLOWED_SIGNALS)
    if allowed:
        return _routed("average_points_allowed", allowed)

    # 3. team_efficiency_summary — efficiency / rating language.
    efficiency = _matches(padded, EFFICIENCY_SIGNALS)
    if efficiency:
        return _routed("team_efficiency_summary", efficiency)

    # 4. team_record — record / win-loss language (h2h already took priority above).
    record = _matches(padded, RECORD_SIGNALS)
    if record:
        return _routed("team_record", record)

    # 5. top_scoring_teams — ranking-style scoring language ("Top teams" alone is too vague).
    top_scoring = _matches(padded, TOP_SCORING_SIGNALS)
    if top_scoring:
        return _routed("top_scoring_teams", top_scoring)

    # 6. team_average_points — generic team scoring when nothing stronger applied.
    average = _matches(padded, AVERAGE_POINTS_SIGNALS)
    if average:
        return _routed("team_average_points", average)

    # 7. team_advanced_profile — broad performance/profile language. Checked AFTER every single-metric
    #    route (so a simple metric query is never hijacked) and BEFORE generic-compare ambiguity (so
    #    "compare ... offense and defense" becomes a profile rather than an ambiguous comparison).
    profile = _matches(padded, PROFILE_SIGNALS)
    if profile:
        return _routed("team_advanced_profile", profile)

    # Generic comparison with no metric/h2h signal is ambiguous, not a guessed head_to_head.
    # Note: "compare A and B <metric>" (e.g. "...record") routes by the metric above — 8B cannot
    # see the two teams; Phase 8C must reject two teams supplied to a single-team tool.
    if _present(padded, COMPARE_SIGNAL):
        return IntentRouteResult.ambiguous(
            (ParseError(
                AMBIGUOUS_INTENT,
                "Comparison is ambiguous; specify a metric (points/record/efficiency) "
                "or a head-to-head signal.",
            ),),
            raw_query=query, normalised_query=normalised,
        )

    return IntentRouteResult.no_route(
        (ParseError(UNSUPPORTED_QUERY, "The query is outside the supported tool catalogue."),),
        raw_query=query, normalised_query=normalised,
    )
