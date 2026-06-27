"""Executable rule-parser query catalogue (Phase 8A).

Static test DATA — not parsing logic. Declares the supported natural-language queries
(with their expected tool + raw candidate arguments) and the unsupported/ambiguous/
incomplete queries (with their expected parse error codes). Later parser phases (8B–8E)
are tested against this single source of truth.

Locked policies captured here as data (detection implemented later):
  - vague time expressions (recent/last few/…) -> unsupported_time_expression, never all-games;
  - bare "compare A and B" -> ambiguous, never auto head_to_head;
  - top_scoring_teams with no number omits ``n`` (downstream tool default); no invented window;
  - team strings are RAW candidates (the validator canonicalises).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

from src.rule_parser_types import (
    AMBIGUOUS_INTENT,
    MISSING_OPPONENT,
    MISSING_TEAM,
    PARSE_ERROR_CODES,
    PARSE_STATUS_PARSED,
    PARSE_STATUSES,
    UNSUPPORTED_QUERY,
    UNSUPPORTED_TIME_EXPRESSION,
)

SUPPORTED_TOOL_NAMES = (
    "team_average_points",
    "average_points_allowed",
    "team_record",
    "top_scoring_teams",
    "head_to_head",
    "team_efficiency_summary",
    "team_advanced_profile",
)


@dataclass(frozen=True)
class RuleQueryExample:
    """One catalogue row: a query and its expected parse outcome."""

    query: str
    expected_status: str
    expected_tool: Optional[str] = None
    expected_arguments: dict[str, object] | None = None
    expected_error_codes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query:
            raise ValueError("query must be a non-empty string.")
        if self.expected_status not in PARSE_STATUSES:
            raise ValueError(f"expected_status must be one of {PARSE_STATUSES}.")
        object.__setattr__(self, "expected_error_codes", tuple(self.expected_error_codes))
        object.__setattr__(self, "tags", tuple(self.tags))
        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("notes must be None or a string.")

        if self.expected_status == PARSE_STATUS_PARSED:
            if self.expected_tool not in SUPPORTED_TOOL_NAMES:
                raise ValueError("a parsed example must name a supported tool.")
            if not isinstance(self.expected_arguments, dict):
                raise TypeError("a parsed example must provide an arguments dict.")
            if self.expected_error_codes:
                raise ValueError("a parsed example must not have error codes.")
            object.__setattr__(
                self, "expected_arguments",
                MappingProxyType(copy.deepcopy(self.expected_arguments)),
            )
        else:
            if self.expected_tool is not None:
                raise ValueError("a non-parsed example must not name a tool.")
            if self.expected_arguments:
                raise ValueError("a non-parsed example must not carry arguments.")
            object.__setattr__(self, "expected_arguments", None)
            if not self.expected_error_codes:
                raise ValueError("a non-parsed example must have at least one error code.")
            for code in self.expected_error_codes:
                if code not in PARSE_ERROR_CODES:
                    raise ValueError(f"unknown parse error code {code!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "expected_status": self.expected_status,
            "expected_tool": self.expected_tool,
            "expected_arguments": (
                dict(self.expected_arguments) if self.expected_arguments is not None else None
            ),
            "expected_error_codes": list(self.expected_error_codes),
            "tags": list(self.tags),
            "notes": self.notes,
        }


def _parsed(query, tool, arguments, *, tags=(), notes=None) -> RuleQueryExample:
    return RuleQueryExample(query, PARSE_STATUS_PARSED, tool, arguments, (), tags, notes)


SUPPORTED_QUERY_EXAMPLES = (
    # team_average_points
    _parsed("How many points do the Warriors average over the last 5 games?",
            "team_average_points", {"team": "Warriors", "window": 5}),
    _parsed("What is the average points scored by the Warriors in their last 5 games?",
            "team_average_points", {"team": "Warriors", "window": 5}, tags=("brief_example",)),
    _parsed("What is GSW averaging over the last 5 games?",
            "team_average_points", {"team": "GSW", "window": 5}),
    _parsed("Average points for Boston Celtics last 10 games",
            "team_average_points", {"team": "Boston Celtics", "window": 10}),
    # average_points_allowed
    _parsed("How many points do GSW allow over the last 5 games?",
            "average_points_allowed", {"team": "GSW", "window": 5}),
    _parsed("Average points allowed by Warriors over the last 5 games",
            "average_points_allowed", {"team": "Warriors", "window": 5}),
    _parsed("How many points are Boston allowing?",
            "average_points_allowed", {"team": "Boston"}),
    # team_record
    _parsed("What is the Warriors record?", "team_record", {"team": "Warriors"}),
    _parsed("Boston Celtics win loss record", "team_record", {"team": "Boston Celtics"}),
    _parsed("Lakers record last 10 games", "team_record", {"team": "Lakers", "window": 10}),
    # top_scoring_teams
    _parsed("Top 5 scoring teams", "top_scoring_teams", {"n": 5}),
    _parsed("Highest scoring teams", "top_scoring_teams", {}, notes="n omitted -> tool default"),
    _parsed("Best offences by points", "top_scoring_teams", {}, notes="n omitted -> tool default"),
    _parsed("Top scoring teams in season 26", "top_scoring_teams", {"season_id": 26}),
    # head_to_head
    _parsed("Celtics vs Heat head to head", "head_to_head",
            {"team_a": "Celtics", "team_b": "Heat"}),
    _parsed("Boston against Miami", "head_to_head", {"team_a": "Boston", "team_b": "Miami"}),
    _parsed("How have the Celtics done against the Heat?", "head_to_head",
            {"team_a": "Celtics", "team_b": "Heat"}),
    _parsed("Celtics record against Heat", "head_to_head",
            {"team_a": "Celtics", "team_b": "Heat"}, tags=("record_against",)),
    _parsed("Celtics vs Heat last 5 meetings", "head_to_head",
            {"team_a": "Celtics", "team_b": "Heat", "window": 5}),
    # team_efficiency_summary
    _parsed("Boston Celtics efficiency last 10 games", "team_efficiency_summary",
            {"team": "Boston Celtics", "window": 10}),
    _parsed("Celtics offensive rating and defensive rating", "team_efficiency_summary",
            {"team": "Celtics"}),
    _parsed("Warriors net rating", "team_efficiency_summary", {"team": "Warriors"}),
    _parsed("Team efficiency summary for Lakers", "team_efficiency_summary", {"team": "Lakers"}),
    # team_advanced_profile (broad performance/profile queries; simple metric queries are unaffected)
    _parsed("How are the Warriors performing over the last 5 games?",
            "team_advanced_profile", {"team": "Warriors", "window": 5}),
    _parsed("Give me the Warriors advanced profile over the last 5 games.",
            "team_advanced_profile", {"team": "Warriors", "window": 5}),
    _parsed("Summarise the Celtics over the last 10 games.",
            "team_advanced_profile", {"team": "Celtics", "window": 10}),
    _parsed("Warriors performance profile", "team_advanced_profile", {"team": "Warriors"}),
    _parsed("Compare the Warriors offense and defense over the last 5 games.",
            "team_advanced_profile", {"team": "Warriors", "window": 5}, tags=("compare",)),
)


def _failed(query, status, codes, *, tags=(), notes=None) -> RuleQueryExample:
    return RuleQueryExample(query, status, None, None, codes, tags, notes)


UNSUPPORTED_QUERY_EXAMPLES = (
    _failed("Who is better?", "no_parse", (UNSUPPORTED_QUERY,)),
    _failed("Tell me about Boston", "no_parse", (UNSUPPORTED_QUERY,)),
    _failed("What happened last night?", "no_parse", (UNSUPPORTED_QUERY,)),
    _failed("Top teams", "no_parse", (UNSUPPORTED_QUERY,), notes="no 'scoring' -> not a ranking"),
    _failed("Compare Lakers and Celtics", "ambiguous", (AMBIGUOUS_INTENT,),
            tags=("compare",), notes="no h2h signal/metric -> ambiguous, not head_to_head"),
    _failed("Compare LA teams", "ambiguous", (AMBIGUOUS_INTENT,), tags=("compare",)),
    _failed("Warriors recent form", "incomplete", (UNSUPPORTED_TIME_EXPRESSION,), tags=("vague_time",)),
    _failed("Warriors last few games", "incomplete", (UNSUPPORTED_TIME_EXPRESSION,), tags=("vague_time",)),
    _failed("Show me recent Warriors games", "incomplete", (UNSUPPORTED_TIME_EXPRESSION,), tags=("vague_time",)),
    # clear intent + a vague time word (must not silently become all-games)
    _failed("Warriors average points lately", "incomplete", (UNSUPPORTED_TIME_EXPRESSION,), tags=("vague_time",)),
    _failed("GSW points allowed recently", "incomplete", (UNSUPPORTED_TIME_EXPRESSION,), tags=("vague_time",)),
    _failed("Lakers record of late", "incomplete", (UNSUPPORTED_TIME_EXPRESSION,), tags=("vague_time",)),
    _failed("Celtics efficiency latest games", "incomplete", (UNSUPPORTED_TIME_EXPRESSION,), tags=("vague_time",)),
    _failed("Celtics vs", "incomplete", (MISSING_OPPONENT,), tags=("h2h_incomplete",)),
    _failed("vs Heat", "incomplete", (MISSING_TEAM,), tags=("h2h_incomplete",)),
    # location splits are not supported -> unsupported, never silently ignored
    _failed("Warriors advanced profile at home", "no_parse", (UNSUPPORTED_QUERY,), tags=("location",)),
    _failed("Lakers record away", "no_parse", (UNSUPPORTED_QUERY,), tags=("location",)),
)

ALL_QUERY_EXAMPLES = SUPPORTED_QUERY_EXAMPLES + UNSUPPORTED_QUERY_EXAMPLES
