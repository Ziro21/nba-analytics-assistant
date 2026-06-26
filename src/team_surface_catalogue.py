"""Explicit, static team-surface catalogue for extraction (Phase 8C).

The rule parser's recognised team vocabulary — the surface strings it may extract as RAW
candidates. This is an EXTRACTION vocabulary, NOT a resolver: it never decides validity,
ambiguity, or canonical identity. The Phase 7 validator/resolver does that.

Design: the catalogue is deliberately standalone and explicit (no import of the resolver's
internals). Drift tests (``tests/test_team_surface_catalogue.py``) assert it stays consistent
with its sources of truth:
  - CANONICAL_TEAM_NAMES == the dataset-derived ``ValidationContext.canonical_teams``;
  - TEAM_ALIAS_SURFACES == the resolver's ``ALIAS_MAP`` keys;
  - AMBIGUOUS_TEAM_SURFACES == the resolver's ``AMBIGUITY_MAP`` keys.
Because every alias surface equals a resolver alias key (and every alias resolves to a
canonical franchise), anything the parser extracts via the catalogue is resolvable later.
Special/exhibition teams (Team Stars/Stripes/World) are deliberately excluded.

No pandas, no dataset load, no registry/tools/validation_context/resolver imports, no execution.
"""

from __future__ import annotations

from src.rule_query_normalisation import normalise_query_text

# All 30 NBA franchises (exact names as they appear in the clean dataframe). Drift-tested.
CANONICAL_TEAM_NAMES: tuple[str, ...] = (
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls",
    "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons",
    "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers",
    "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks",
    "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder",
    "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers",
    "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards",
)

# Safe alias surfaces (normalised). Exactly the resolver's ALIAS_MAP keys — nicknames and
# tri-codes only. Ambiguous market tokens (la/ny/...) are deliberately NOT here. Drift-tested.
TEAM_ALIAS_SURFACES: tuple[str, ...] = (
    "hawks", "atl",
    "celtics", "bos",
    "nets", "bkn",
    "hornets", "cha",
    "bulls", "chi",
    "cavaliers", "cavs", "cle",
    "mavericks", "mavs", "dal",
    "nuggets", "den",
    "pistons", "det",
    "warriors", "gsw",
    "rockets", "hou",
    "pacers", "ind",
    "clippers", "lac",
    "lakers", "lal",
    "grizzlies", "mem",
    "heat", "mia",
    "bucks", "mil",
    "timberwolves", "wolves", "min",
    "pelicans", "nop",
    "knicks", "nyk",
    "thunder", "okc",
    "magic", "orl",
    "76ers", "sixers", "phi",
    "suns", "phx",
    "trail blazers", "blazers", "por",
    "kings", "sac",
    "spurs", "sas",
    "raptors", "tor",
    "jazz", "uta",
    "wizards", "was",
)

# Ambiguous market surfaces (normalised). Extracted RAW; the validator returns ambiguous_team.
# Exactly the resolver's AMBIGUITY_MAP keys. Drift-tested.
AMBIGUOUS_TEAM_SURFACES: tuple[str, ...] = ("la", "los angeles", "ny", "new york")

# Special/exhibition team surfaces (normalised). These ARE recognised so the parser extracts the
# FULL phrase raw (e.g. "Team World"), letting the validator reject it with invalid_special_team
# rather than the fallback mangling "Team World" into a partial "World". Drift-tested against the
# dataset's special_teams. They are surfaces (extractable) but never canonical (never valid).
SPECIAL_TEAM_SURFACES: tuple[str, ...] = ("team stars", "team stripes", "team world")


def normalise_surface(value: str) -> str:
    """Normalise a surface/candidate the same way queries are normalised (Phase 8B)."""
    return normalise_query_text(value)


def _build_surface_forms() -> tuple[str, ...]:
    forms: list[str] = []
    seen: set[str] = set()
    for source in (
        tuple(normalise_surface(name) for name in CANONICAL_TEAM_NAMES),
        TEAM_ALIAS_SURFACES,
        AMBIGUOUS_TEAM_SURFACES,
        SPECIAL_TEAM_SURFACES,
    ):
        for surface in source:
            if surface and surface not in seen:
                seen.add(surface)
                forms.append(surface)
    return tuple(forms)


# Every matchable surface: normalised canonical names + alias surfaces + ambiguous surfaces.
TEAM_SURFACE_FORMS: tuple[str, ...] = _build_surface_forms()


def get_team_surface_forms() -> tuple[str, ...]:
    """All recognised team surface forms (normalised)."""
    return TEAM_SURFACE_FORMS


def get_team_surface_forms_by_length() -> tuple[str, ...]:
    """Surface forms ordered for deterministic longest-match: more words first, then longer
    strings, then alphabetical for stability."""
    return tuple(
        sorted(TEAM_SURFACE_FORMS, key=lambda s: (-len(s.split()), -len(s), s))
    )
