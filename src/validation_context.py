"""Reference context for the shared validator (Phase 7B).

`build_validation_context` derives a pure, JSON-serialisable reference layer from the
clean dataframe and the tool registry: registered tools and their safe schemas, the
canonical franchise teams, valid (opaque) season ids, the special/exhibition teams, and
the curated alias/ambiguity maps. The dataframe is read for reference data only and is
NEVER stored. No dataset loading, no tool execution, no registry execution here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from src.team_resolution import ALIAS_MAP, AMBIGUITY_MAP, normalise_team_text


def _deep_freeze(obj: Any) -> Any:
    """Recursively convert dicts to read-only mappings and lists/tuples to tuples."""
    if isinstance(obj, MappingProxyType):
        return obj
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze(v) for v in obj)
    return obj


def _to_plain(obj: Any) -> Any:
    """Recursively convert read-only mappings/tuples back to plain dicts/lists (for JSON)."""
    if isinstance(obj, (MappingProxyType, dict)):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


@dataclass(frozen=True)
class ValidationContext:
    """Immutable reference data used by the Phase 7C validator (never holds the dataframe)."""

    registered_tools: tuple[str, ...]
    tool_schemas: tuple[dict, ...]
    tool_schema_by_name: dict
    canonical_teams: tuple[str, ...]
    normalised_team_lookup: dict
    valid_season_ids: tuple[int, ...]
    special_teams: tuple[str, ...]
    alias_map: dict
    ambiguity_map: dict

    def __post_init__(self) -> None:
        object.__setattr__(self, "registered_tools", tuple(self.registered_tools))
        object.__setattr__(self, "tool_schemas", _deep_freeze(list(self.tool_schemas)))
        object.__setattr__(
            self, "tool_schema_by_name", _deep_freeze(dict(self.tool_schema_by_name))
        )
        object.__setattr__(self, "canonical_teams", tuple(self.canonical_teams))
        object.__setattr__(
            self, "normalised_team_lookup", MappingProxyType(dict(self.normalised_team_lookup))
        )
        object.__setattr__(self, "valid_season_ids", tuple(int(s) for s in self.valid_season_ids))
        object.__setattr__(self, "special_teams", tuple(self.special_teams))
        object.__setattr__(self, "alias_map", MappingProxyType(dict(self.alias_map)))
        object.__setattr__(
            self, "ambiguity_map",
            MappingProxyType({k: tuple(v) for k, v in self.ambiguity_map.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "registered_tools": list(self.registered_tools),
            "tool_schemas": _to_plain(self.tool_schemas),
            "tool_schema_by_name": _to_plain(self.tool_schema_by_name),
            "canonical_teams": list(self.canonical_teams),
            "normalised_team_lookup": dict(self.normalised_team_lookup),
            "valid_season_ids": list(self.valid_season_ids),
            "special_teams": list(self.special_teams),
            "alias_map": dict(self.alias_map),
            "ambiguity_map": {k: list(v) for k, v in self.ambiguity_map.items()},
        }


def _validate_curated_maps(
    canonical_teams: tuple[str, ...],
    special_teams: tuple[str, ...],
    alias_map: dict[str, str],
    ambiguity_map: dict[str, tuple[str, ...]],
) -> None:
    """Fail-fast build-time validation of the curated alias/ambiguity maps."""
    canonical_set = set(canonical_teams)
    special_set = set(special_teams)
    overlap = set(alias_map) & set(ambiguity_map)
    if overlap:
        raise ValueError(f"alias/ambiguity key overlap: {sorted(overlap)}")
    for key, target in alias_map.items():
        if normalise_team_text(key) != key:
            raise ValueError(f"alias key {key!r} is not normalised.")
        if target in special_set:
            raise ValueError(f"alias {key!r} targets special team {target!r}.")
        if target not in canonical_set:
            raise ValueError(f"alias {key!r} targets non-canonical team {target!r}.")
    for key, candidates in ambiguity_map.items():
        if normalise_team_text(key) != key:
            raise ValueError(f"ambiguity key {key!r} is not normalised.")
        for candidate in candidates:
            if candidate not in canonical_set:
                raise ValueError(f"ambiguity {key!r} candidate {candidate!r} is not canonical.")


def build_validation_context(clean_df, *, registry) -> ValidationContext:
    """Build a `ValidationContext` from the clean dataframe and the tool registry.

    The dataframe is read for reference data only and is not stored. Schemas are copied
    defensively from the registry. The curated alias/ambiguity maps are validated against
    the dataset-derived canonical teams (raising on any inconsistency).
    """
    schemas = copy.deepcopy(registry.schemas())
    registered_tools = tuple(schema["name"] for schema in schemas)
    schema_by_name = {schema["name"]: schema for schema in schemas}

    is_exhibition = clean_df["is_exhibition"]
    special_teams = tuple(sorted(clean_df.loc[is_exhibition, "team_name"].unique().tolist()))
    canonical_teams = tuple(sorted(clean_df.loc[~is_exhibition, "team_name"].unique().tolist()))
    normalised_lookup = {normalise_team_text(team): team for team in canonical_teams}
    valid_season_ids = tuple(sorted(int(s) for s in clean_df["season_id"].unique().tolist()))

    _validate_curated_maps(canonical_teams, special_teams, ALIAS_MAP, AMBIGUITY_MAP)

    return ValidationContext(
        registered_tools=registered_tools,
        tool_schemas=tuple(schemas),
        tool_schema_by_name=schema_by_name,
        canonical_teams=canonical_teams,
        normalised_team_lookup=normalised_lookup,
        valid_season_ids=valid_season_ids,
        special_teams=special_teams,
        alias_map=dict(ALIAS_MAP),
        ambiguity_map=dict(AMBIGUITY_MAP),
    )
