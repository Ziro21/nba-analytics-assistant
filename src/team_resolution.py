"""Deterministic team-name resolution (Phase 7B).

Pure and standard-library only: normalisation, curated alias/ambiguity handling,
special-team rejection, and `difflib`-based suggestions. No pandas, no registry, no data
loading, no validation execution. Fuzzy matching produces SUGGESTIONS ONLY — it never
auto-corrects an input into a resolved team.

The curated ``ALIAS_MAP`` and ``AMBIGUITY_MAP`` keys are normalised; every alias target and
ambiguity candidate is a canonical franchise name. ``validation_context`` validates these
maps against the dataset-derived canonical teams at build time (fail-fast).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Optional

TEAM_RESOLVED = "resolved"
TEAM_UNKNOWN = "unknown_team"
TEAM_AMBIGUOUS = "ambiguous_team"
TEAM_INVALID_SPECIAL = "invalid_special_team"

_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")

# Curated alias map: normalised key -> exact canonical franchise name.
# Standard NBA nicknames and tri-codes; each maps to exactly one team. Ambiguous
# city/market tokens (la, los angeles, ny, new york) are deliberately NOT aliases.
ALIAS_MAP: dict[str, str] = {
    "hawks": "Atlanta Hawks", "atl": "Atlanta Hawks",
    "celtics": "Boston Celtics", "bos": "Boston Celtics",
    "nets": "Brooklyn Nets", "bkn": "Brooklyn Nets",
    "hornets": "Charlotte Hornets", "cha": "Charlotte Hornets",
    "bulls": "Chicago Bulls", "chi": "Chicago Bulls",
    "cavaliers": "Cleveland Cavaliers", "cavs": "Cleveland Cavaliers", "cle": "Cleveland Cavaliers",
    "mavericks": "Dallas Mavericks", "mavs": "Dallas Mavericks", "dal": "Dallas Mavericks",
    "nuggets": "Denver Nuggets", "den": "Denver Nuggets",
    "pistons": "Detroit Pistons", "det": "Detroit Pistons",
    "warriors": "Golden State Warriors", "gsw": "Golden State Warriors",
    "rockets": "Houston Rockets", "hou": "Houston Rockets",
    "pacers": "Indiana Pacers", "ind": "Indiana Pacers",
    "clippers": "Los Angeles Clippers", "lac": "Los Angeles Clippers",
    "lakers": "Los Angeles Lakers", "lal": "Los Angeles Lakers",
    "grizzlies": "Memphis Grizzlies", "mem": "Memphis Grizzlies",
    "heat": "Miami Heat", "mia": "Miami Heat",
    "bucks": "Milwaukee Bucks", "mil": "Milwaukee Bucks",
    "timberwolves": "Minnesota Timberwolves", "wolves": "Minnesota Timberwolves",
    "min": "Minnesota Timberwolves",
    "pelicans": "New Orleans Pelicans", "nop": "New Orleans Pelicans",
    "knicks": "New York Knicks", "nyk": "New York Knicks",
    "thunder": "Oklahoma City Thunder", "okc": "Oklahoma City Thunder",
    "magic": "Orlando Magic", "orl": "Orlando Magic",
    "76ers": "Philadelphia 76ers", "sixers": "Philadelphia 76ers", "phi": "Philadelphia 76ers",
    "suns": "Phoenix Suns", "phx": "Phoenix Suns",
    "trail blazers": "Portland Trail Blazers", "blazers": "Portland Trail Blazers",
    "por": "Portland Trail Blazers",
    "kings": "Sacramento Kings", "sac": "Sacramento Kings",
    "spurs": "San Antonio Spurs", "sas": "San Antonio Spurs",
    "raptors": "Toronto Raptors", "tor": "Toronto Raptors",
    "jazz": "Utah Jazz", "uta": "Utah Jazz",
    "wizards": "Washington Wizards", "was": "Washington Wizards",
}

# Curated ambiguity map: normalised key -> candidate canonical names (never guessed).
AMBIGUITY_MAP: dict[str, tuple[str, ...]] = {
    "la": ("Los Angeles Lakers", "Los Angeles Clippers"),
    "los angeles": ("Los Angeles Lakers", "Los Angeles Clippers"),
    "ny": ("New York Knicks", "Brooklyn Nets"),
    "new york": ("New York Knicks", "Brooklyn Nets"),
}


def normalise_team_text(value: str) -> str:
    """Deterministically normalise a team string: lower-case, strip simple punctuation,
    and collapse whitespace. Digits are preserved (e.g. ``"76ers"``)."""
    if not isinstance(value, str):
        raise TypeError("team value must be a string.")
    text = value.lower().strip()
    text = _NON_ALNUM_SPACE.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


_VALID_STATUSES = frozenset(
    {TEAM_RESOLVED, TEAM_UNKNOWN, TEAM_AMBIGUOUS, TEAM_INVALID_SPECIAL}
)


@dataclass(frozen=True)
class TeamResolutionResult:
    """Outcome of resolving one team string. Public fields are validated so ``to_dict()``
    is always JSON-serialisable, even under direct construction."""

    status: str
    input_value: str
    canonical_name: Optional[str] = None
    suggestions: tuple[str, ...] = ()
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_STATUSES)}, got {self.status!r}."
            )
        if not isinstance(self.input_value, str):
            raise TypeError("input_value must be a string.")
        if self.canonical_name is not None and not isinstance(self.canonical_name, str):
            raise TypeError("canonical_name must be None or a string.")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be None or a string.")
        if isinstance(self.suggestions, str):
            raise TypeError("suggestions must be a sequence of strings, not a string.")
        coerced = tuple(self.suggestions)
        for item in coerced:
            if not isinstance(item, str):
                raise TypeError("suggestions must contain only strings.")
        object.__setattr__(self, "suggestions", coerced)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "input_value": self.input_value,
            "canonical_name": self.canonical_name,
            "suggestions": list(self.suggestions),
            "message": self.message,
        }


def resolve_team_name(
    value: str,
    *,
    canonical_teams: tuple[str, ...],
    special_teams: tuple[str, ...],
    alias_map: dict[str, str] | None = None,
    ambiguity_map: dict[str, tuple[str, ...]] | None = None,
    max_suggestions: int = 3,
) -> TeamResolutionResult:
    """Resolve a user-provided team string to a canonical franchise, deterministically.

    Resolution order: exact/normalised canonical → special rejection → curated alias →
    curated ambiguity → fuzzy suggestions (``unknown_team``). Fuzzy matches are returned
    only as suggestions; they never produce ``status="resolved"``.
    """
    alias_map = alias_map if alias_map is not None else {}
    ambiguity_map = ambiguity_map if ambiguity_map is not None else {}
    norm = normalise_team_text(value)

    canonical_by_norm = {normalise_team_text(t): t for t in canonical_teams}
    special_by_norm = {normalise_team_text(t): t for t in special_teams}

    # 1 & 2: exact / normalised canonical match.
    if norm in canonical_by_norm:
        canonical = canonical_by_norm[norm]
        return TeamResolutionResult(
            TEAM_RESOLVED, value, canonical, (), f"Resolved team input to {canonical}."
        )

    # 3: special / exhibition team rejection.
    if norm in special_by_norm:
        special = special_by_norm[norm]
        return TeamResolutionResult(
            TEAM_INVALID_SPECIAL, value, None, (),
            f"{special} is not a supported NBA franchise team for this assistant.",
        )

    # 4: curated alias.
    if norm in alias_map:
        canonical = alias_map[norm]
        return TeamResolutionResult(
            TEAM_RESOLVED, value, canonical, (), f"Resolved team input to {canonical}."
        )

    # 5: curated ambiguity (never guessed).
    if norm in ambiguity_map:
        candidates = tuple(ambiguity_map[norm])
        return TeamResolutionResult(
            TEAM_AMBIGUOUS, value, None, candidates, f"The team input {value!r} is ambiguous."
        )

    # 6: fuzzy suggestions only -> unknown.
    pool: dict[str, str] = dict(canonical_by_norm)
    for key, canonical in alias_map.items():
        pool.setdefault(key, canonical)
    matches = difflib.get_close_matches(norm, list(pool), n=max_suggestions, cutoff=0.6)
    suggestions: list[str] = []
    for match in matches:
        canonical = pool[match]
        if canonical not in suggestions:
            suggestions.append(canonical)
    return TeamResolutionResult(
        TEAM_UNKNOWN, value, None, tuple(suggestions[:max_suggestions]),
        f"I could not find a team matching {value!r}.",
    )
