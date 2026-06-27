"""Raw slot extraction for the rule parser (Phase 8C).

Given a query already routed to a candidate tool (Phase 8B), extract RAW candidate slots:
team / team_a / team_b / window / n / season_id. This layer NEVER decides validity or
ambiguity, NEVER canonicalises, NEVER calls the resolver/validator/registry, and NEVER loads
data or computes statistics. It emits raw spans; the Phase 7 validator canonicalises later.

Team extraction = explicit team-surface gazetteer (longest-match) + a precision-gated
structural fallback so typo-like / unaliased spans still reach the validator. Numbers are
parsed only from explicit expressions; vague time fails as ``unsupported_time_expression``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

from src.rule_parser_types import (
    AMBIGUOUS_TEAM_MENTION,
    MISSING_OPPONENT,
    MISSING_TEAM,
    UNSUPPORTED_QUERY,
    UNSUPPORTED_TIME_EXPRESSION,
    ParseError,
    ParseWarning,
)
from src.rule_query_catalogue import SUPPORTED_TOOL_NAMES
from src.rule_query_normalisation import normalise_query_text
from src.team_surface_catalogue import get_team_surface_forms_by_length, normalise_surface

SLOT_STATUS_EXTRACTED = "extracted"
SLOT_STATUS_INCOMPLETE = "incomplete"
SLOT_STATUS_UNSUPPORTED = "unsupported"
SLOT_STATUSES = (SLOT_STATUS_EXTRACTED, SLOT_STATUS_INCOMPLETE, SLOT_STATUS_UNSUPPORTED)

SINGLE_TEAM_TOOLS = frozenset(
    {"team_average_points", "average_points_allowed", "team_record", "team_efficiency_summary",
     "team_advanced_profile"}
)
H2H_TOOL = "head_to_head"
COMPARE_TOOL = "compare_team_profiles"
RANKING_TOOL = "top_scoring_teams"
# Tools that accept an optional last-N window (the comparison applies it per team).
WINDOW_TOOLS = SINGLE_TEAM_TOOLS | {H2H_TOOL, COMPARE_TOOL}

# Explicit numbered time windows only (never invent a window). The unit noun ("games"/"meetings") is
# optional, so "last 10" reads as a 10-game window — a bare numbered "last N" is never silently
# dropped to all-games. A vague "last few"/"recent" (no number) is still an unsupported time phrase.
_WINDOW_RE = re.compile(r"\b(?:last|past|previous)\s+(\d+)(?:\s+(?:games?|meetings?))?\b")
# Top-N ranking number (top_scoring_teams only).
_TOP_N_RE = re.compile(r"\btop\s+(\d+)\b")
# Opaque dataset season id only ("season 26" / "season id 26"); never NBA season labels.
_SEASON_RE = re.compile(r"\bseason\s+(?:id\s+)?(\d+)\b")

# Vague recency expressions with no number -> unsupported, never silently all-games.
_VAGUE_TIME_EXPRESSIONS = (
    "recent form", "of late", "last few", "past few", "recently", "lately", "latest", "recent",
)

# Head-to-head separators (tolerant of hyphenated head-to-head and "vs.").
_H2H_SPLIT_RE = re.compile(
    r"\b(?:head[\s\-]+to[\s\-]+head|versus|vs|against|h2h|matchup)\b", re.IGNORECASE
)

# Two-team comparison connectors ("compare A and B", "compare A with B"). Bare "vs"/"versus" is NOT
# a comparison connector — it stays a head-to-head signal, so "compare A vs B" routes to head_to_head.
_COMPARE_SPLIT_RE = re.compile(r"\b(?:and|with)\b", re.IGNORECASE)

# Words that can never be a team candidate (stopwords + temporal + metric/intent vocabulary).
_BLOCKED_TOKENS = frozenset({
    # stopwords
    "how", "many", "much", "do", "does", "did", "is", "are", "was", "were", "be", "been",
    "being", "the", "a", "an", "of", "over", "in", "on", "for", "by", "their", "them", "they",
    "there", "here", "what", "whats", "which", "who", "whom", "show", "tell", "me", "us",
    "about", "get", "please", "this", "that", "these", "those", "with", "and", "or", "to",
    "from", "as", "at", "it", "its", "have", "has", "had", "i", "we", "you", "your", "my",
    # temporal
    "last", "past", "previous", "season", "game", "games", "meeting", "meetings", "recent",
    "recently", "lately", "latest", "form", "few", "late", "night", "tonight", "yesterday",
    "today", "current",
    # metric / intent
    "points", "point", "average", "averages", "averaging", "record", "records", "rating",
    "ratings", "offensive", "defensive", "efficiency", "scoring", "score", "scored", "scores",
    "net", "ortg", "drtg", "top", "best", "highest", "lowest", "allow", "allows", "allowed",
    "allowing", "give", "gives", "given", "up", "concede", "concedes", "conceded", "conceding",
    "win", "wins", "loss", "losses", "versus", "vs", "against", "matchup", "h2h", "team",
    "teams", "offence", "offences", "offense", "offenses", "compare", "better", "summary",
    "done", "doing", "happened",
    # broad-profile / comparison vocabulary (so these never leak as a typo team candidate)
    "advanced", "profile", "profiles", "performing", "performance", "summarise", "summarize",
    "defense", "defenses", "defence", "defences", "comparison", "comparisons", "between",
    # venue / location context (extracted separately as the location slot)
    "home", "away", "road", "neutral", "site", "court", "venue",
    # special/exhibition team components: full phrases ("Team World") are matched by the
    # gazetteer and extracted whole; these block a BARE remnant ("World") from leaking as a team.
    "stars", "stripes", "world",
})


@dataclass(frozen=True)
class SlotExtractionResult:
    """Raw extracted slots for a routed tool, or a structured extraction failure."""

    status: str
    arguments: Optional[dict] = None
    errors: tuple[ParseError, ...] = ()
    warnings: tuple[ParseWarning, ...] = ()
    raw_query: str = ""
    normalised_query: str = ""
    tool_name: str = ""
    matched_surfaces: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in SLOT_STATUSES:
            raise ValueError(f"status must be one of {SLOT_STATUSES}, got {self.status!r}.")
        for field_name in ("raw_query", "normalised_query", "tool_name"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string.")
        errors = tuple(self.errors)
        warnings = tuple(self.warnings)
        surfaces = tuple(self.matched_surfaces)
        for error in errors:
            if not isinstance(error, ParseError):
                raise TypeError("errors must contain only ParseError objects.")
        for warning in warnings:
            if not isinstance(warning, ParseWarning):
                raise TypeError("warnings must contain only ParseWarning objects.")
        for surface in surfaces:
            if not isinstance(surface, str):
                raise TypeError("matched_surfaces must contain only strings.")
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "matched_surfaces", surfaces)

        if self.status == SLOT_STATUS_EXTRACTED:
            if not isinstance(self.arguments, dict):
                raise TypeError("an extracted result must carry an arguments dict.")
            if errors:
                raise ValueError("an extracted result must not contain errors.")
            object.__setattr__(self, "arguments", MappingProxyType(copy.deepcopy(self.arguments)))
        else:
            if self.arguments:
                raise ValueError("a non-extracted result must not carry arguments.")
            object.__setattr__(self, "arguments", None)
            if not errors:
                raise ValueError("a non-extracted result must contain at least one error.")

    @classmethod
    def extracted(cls, arguments, **kw) -> "SlotExtractionResult":
        return cls(SLOT_STATUS_EXTRACTED, arguments, **kw)

    @classmethod
    def incomplete(cls, errors, **kw) -> "SlotExtractionResult":
        return cls(SLOT_STATUS_INCOMPLETE, None, errors, **kw)

    @classmethod
    def unsupported(cls, errors, **kw) -> "SlotExtractionResult":
        return cls(SLOT_STATUS_UNSUPPORTED, None, errors, **kw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "arguments": dict(self.arguments) if self.arguments is not None else None,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "raw_query": self.raw_query,
            "normalised_query": self.normalised_query,
            "tool_name": self.tool_name,
            "matched_surfaces": list(self.matched_surfaces),
        }


# --- surface patterns (compiled once) ---------------------------------------

def _compile_surface_patterns() -> tuple[tuple[str, "re.Pattern[str]"], ...]:
    patterns = []
    for surface in get_team_surface_forms_by_length():  # longest-match order
        words = surface.split()
        body = r"\s+".join(re.escape(w) for w in words)
        patterns.append((surface, re.compile(r"\b" + body + r"\b", re.IGNORECASE)))
    return tuple(patterns)


_SURFACE_PATTERNS = _compile_surface_patterns()


def _find_team_mentions(text: str) -> list[tuple[int, int, str, str]]:
    """Non-overlapping team-surface matches in `text` as (start, end, raw_span, surface),
    preferring longer matches, in left-to-right order."""
    candidates: list[tuple[int, int, str, str]] = []
    for surface, pattern in _SURFACE_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), match.group(), surface))
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    selected: list[tuple[int, int, str, str]] = []
    next_free = 0
    for start, end, span, surface in candidates:
        if start >= next_free:
            selected.append((start, end, span, surface))
            next_free = end
    selected.sort(key=lambda c: c[0])
    return selected


def _clean_candidate(token: str) -> str:
    """Strip surrounding punctuation from a raw token, preserving inner casing."""
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", token)


def _structural_fallback(text: str) -> Optional[str]:
    """Precision-gated typo/unaliased fallback: return a single clean candidate team string
    from `text` if one (and only one) plausible residual span exists, else None.

    Fires only when the gazetteer found nothing. A residual is a maximal run of tokens that are
    not stopwords/temporal/metric words and not numbers. Exactly one residual run of 1-2 tokens
    qualifies; anything else returns None (-> missing_team)."""
    runs: list[list[str]] = []
    current: list[str] = []
    for token in text.split():
        cleaned = _clean_candidate(token)
        norm = normalise_surface(cleaned)
        if not norm or norm.isdigit() or norm in _BLOCKED_TOKENS:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(cleaned)
    if current:
        runs.append(current)
    if len(runs) == 1 and 1 <= len(runs[0]) <= 2:
        return " ".join(runs[0])
    return None


# --- number extraction ------------------------------------------------------

def _extract_window(normalised: str) -> Optional[int]:
    match = _WINDOW_RE.search(normalised)
    return int(match.group(1)) if match else None


def _extract_n(normalised: str) -> Optional[int]:
    match = _TOP_N_RE.search(normalised)
    return int(match.group(1)) if match else None


def _extract_season_id(normalised: str) -> Optional[int]:
    match = _SEASON_RE.search(normalised)
    return int(match.group(1)) if match else None


def _has_vague_time(normalised: str) -> bool:
    padded = f" {normalised} "
    return any(f" {expr} " in padded for expr in _VAGUE_TIME_EXPRESSIONS)


# Venue split: "away"/"on the road" -> away; "home"/"at home" -> home. Whole-phrase, deterministic.
_AWAY_SIGNALS = ("away", "road")
_HOME_SIGNALS = ("home",)
# Venue modifiers we do NOT support (the dataset is home/away only). Extracted as a raw, invalid
# location so the validator rejects it (invalid_location) rather than silently ignoring the modifier.
_UNSUPPORTED_VENUE_SIGNALS = ("neutral",)


def _extract_location(normalised: str) -> Optional[str]:
    """Return a raw venue slot: ``"home"``/``"away"`` for a supported split, the literal modifier for
    an UNSUPPORTED venue (e.g. ``"neutral"`` for "neutral site"/"neutral court"), or ``None`` when no
    venue context is present. Raw slot only — the validator decides validity, so a venue modifier is
    never silently ignored. Unsupported venues are checked first; otherwise away wins ties."""
    padded = f" {normalised} "
    if any(f" {s} " in padded for s in _UNSUPPORTED_VENUE_SIGNALS):
        return "neutral"
    if any(f" {s} " in padded for s in _AWAY_SIGNALS):
        return "away"
    if any(f" {s} " in padded for s in _HOME_SIGNALS):
        return "home"
    return None


# --- team extraction --------------------------------------------------------

def _extract_one_team(text: str) -> tuple[Optional[str], tuple[str, ...]]:
    """Best single team candidate from `text`: first gazetteer mention, else fallback."""
    mentions = _find_team_mentions(text)
    if mentions:
        return mentions[0][2], (mentions[0][3],)
    candidate = _structural_fallback(text)
    return candidate, ()


def _extract_h2h(raw_query: str, normalised: str):
    """Extract team_a/team_b around the first head-to-head separator."""
    arguments: dict[str, object] = {}
    errors: list[ParseError] = []
    surfaces: list[str] = []

    split = _H2H_SPLIT_RE.search(raw_query)
    if split:
        left, right = raw_query[: split.start()], raw_query[split.end():]
    else:
        # No separator span (rare): split on the first/second mentions across the whole query.
        mentions = _find_team_mentions(raw_query)
        left = raw_query[: mentions[0][1]] if mentions else raw_query
        right = raw_query[mentions[0][1]:] if mentions else ""

    team_a, surf_a = _extract_one_team(left)
    team_b, surf_b = _extract_one_team(right)
    surfaces.extend(surf_a)
    surfaces.extend(surf_b)

    if team_a is not None:
        arguments["team_a"] = team_a
    else:
        errors.append(ParseError(MISSING_TEAM, "No first team found before the matchup signal.",
                                 field="team_a"))
    if team_b is not None:
        arguments["team_b"] = team_b
    else:
        errors.append(ParseError(MISSING_OPPONENT, "No opponent found after the matchup signal.",
                                 field="team_b"))
    return arguments, errors, tuple(surfaces)


def _extract_compare(raw_query: str, normalised: str):
    """Extract team_a/team_b around the first comparison connector ('and'/'with').

    Mirrors :func:`_extract_h2h` but splits on the comparison connector, so each side gets the
    structural fallback (a typo like 'Celics' still reaches the validator for a suggestion). With no
    connector, falls back to the first team mention boundary; a single team then fails as incomplete.
    """
    arguments: dict[str, object] = {}
    errors: list[ParseError] = []
    surfaces: list[str] = []

    split = _COMPARE_SPLIT_RE.search(raw_query)
    if split:
        left, right = raw_query[: split.start()], raw_query[split.end():]
    else:
        mentions = _find_team_mentions(raw_query)
        left = raw_query[: mentions[0][1]] if mentions else raw_query
        right = raw_query[mentions[0][1]:] if mentions else ""

    team_a, surf_a = _extract_one_team(left)
    team_b, surf_b = _extract_one_team(right)
    surfaces.extend(surf_a)
    surfaces.extend(surf_b)

    if team_a is not None:
        arguments["team_a"] = team_a
    else:
        errors.append(ParseError(MISSING_TEAM, "No first team found for the comparison.",
                                 field="team_a"))
    if team_b is not None:
        arguments["team_b"] = team_b
    else:
        errors.append(ParseError(MISSING_OPPONENT, "No second team found for the comparison.",
                                 field="team_b"))
    return arguments, errors, tuple(surfaces)


# --- public API -------------------------------------------------------------

def extract_slots(query: str, *, tool_name: str) -> SlotExtractionResult:
    """Extract raw candidate slots from `query` for an already-routed `tool_name`."""
    normalised = normalise_query_text(query)  # raises TypeError on non-str input

    if tool_name not in SUPPORTED_TOOL_NAMES:
        return SlotExtractionResult.unsupported(
            (ParseError(UNSUPPORTED_QUERY, f"Unknown tool {tool_name!r}."),),
            raw_query=query, normalised_query=normalised, tool_name=str(tool_name),
        )

    meta = dict(raw_query=query, normalised_query=normalised, tool_name=tool_name)

    # Location is a RAW slot, extracted for any tool. The validator rejects it for tools that do not
    # accept a venue split (top_scoring_teams / head_to_head), so it is never silently ignored.
    location = _extract_location(normalised)

    # Ranking tool: optional n + optional season_id, no team, no required slots.
    if tool_name == RANKING_TOOL:
        arguments: dict[str, object] = {}
        n = _extract_n(normalised)
        season_id = _extract_season_id(normalised)
        if n is not None:
            arguments["n"] = n
        if season_id is not None:
            arguments["season_id"] = season_id
        if location is not None:
            arguments["location"] = location
        return SlotExtractionResult.extracted(arguments, **meta)

    errors: list[ParseError] = []
    arguments = {}
    surfaces: tuple[str, ...] = ()

    # Window vs vague time (window tools only).
    window = _extract_window(normalised) if tool_name in WINDOW_TOOLS else None
    if window is not None:
        arguments["window"] = window
    elif tool_name in WINDOW_TOOLS and _has_vague_time(normalised):
        errors.append(ParseError(
            UNSUPPORTED_TIME_EXPRESSION,
            "Vague time expression; specify an explicit number of games (e.g. 'last 5 games').",
            field="window",
        ))

    # Location split (raw; validity decided by the validator per tool).
    if location is not None:
        arguments["location"] = location

    # Teams.
    if tool_name == H2H_TOOL:
        team_args, team_errors, surfaces = _extract_h2h(query, normalised)
        arguments.update(team_args)
        errors.extend(team_errors)
    elif tool_name == COMPARE_TOOL:
        team_args, team_errors, surfaces = _extract_compare(query, normalised)
        arguments.update(team_args)
        errors.extend(team_errors)
    else:  # single-team tools
        mentions = _find_team_mentions(query)
        if len(mentions) >= 2:
            return SlotExtractionResult.unsupported(
                (ParseError(
                    AMBIGUOUS_TEAM_MENTION,
                    "Multiple teams found for a single-team query; this tool takes one team.",
                    field="team",
                    suggestions=tuple(m[2] for m in mentions),
                ),),
                raw_query=query, normalised_query=normalised, tool_name=tool_name,
                matched_surfaces=tuple(m[3] for m in mentions),
            )
        team, surfaces = _extract_one_team(query)
        if team is not None:
            arguments["team"] = team
        else:
            errors.append(ParseError(MISSING_TEAM, "No team found in the query.", field="team"))

    if errors:
        return SlotExtractionResult.incomplete(
            tuple(errors), **meta, matched_surfaces=surfaces,
        )
    return SlotExtractionResult.extracted(arguments, **meta, matched_surfaces=surfaces)
