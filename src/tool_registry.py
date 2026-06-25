"""Tool registry foundation: explicit specs, schemas, and safe dispatch.

A small, explicit registry. Each tool is described by a hand-written ``ToolSpec`` (the
single source of truth for its public schema — no docstring parsing). The registry maps
a tool name to its spec, exports JSON-serialisable schemas (the LLM parser will later
consume these), and dispatches execution.

Design boundaries (Phase 6A):
  - ``clean_df`` is an injected runtime dependency, passed keyword-only to ``execute`` and
    NEVER part of a public schema or a tool parameter.
  - The registry does SHALLOW argument-shape validation only (known tool, args is a mapping,
    required args present, no unexpected args, ``clean_df`` not smuggled in ``args``).
    Deep semantic validation (real team names, positive windows, season existence, …)
    belongs to the analytical tools and the later validator/parser layers.
  - Normal request problems return a structured ``status="error"`` result (never raise).
  - Developer/configuration mistakes (invalid spec, duplicate registration) raise.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.tool_results import ToolResult, error_result
from src.tools import (
    average_points_allowed,
    head_to_head,
    team_average_points,
    team_efficiency_summary,
    team_record,
    top_scoring_teams,
)

CLEAN_DF_PARAM = "clean_df"
REGISTRY_TOOL_NAME = "tool_registry"

# JSON-safe parameter type strings allowed in public schemas. Restricting to this set
# guarantees schemas stay serialisable and rules out raw Python type objects and typos.
ALLOWED_PARAM_TYPES = frozenset(
    {"str", "int", "int|null", "float", "float|null", "bool", "bool|null"}
)


@dataclass(frozen=True)
class ToolParameter:
    """One user-facing tool parameter (never ``clean_df``).

    ``type`` is a JSON-safe string (e.g. ``"str"``, ``"int"``, ``"int|null"``,
    ``"float"``, ``"bool"``) — not a raw Python type — so schemas stay serialisable.
    """

    name: str
    type: str
    required: bool
    description: str
    default: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ToolParameter name must be a non-empty string.")
        if self.name == CLEAN_DF_PARAM:
            raise ValueError(f"{CLEAN_DF_PARAM!r} must never be a tool parameter.")
        if not isinstance(self.type, str) or self.type not in ALLOWED_PARAM_TYPES:
            raise ValueError(
                f"ToolParameter type must be one of {sorted(ALLOWED_PARAM_TYPES)}, "
                f"got {self.type!r}."
            )
        if not isinstance(self.required, bool):
            raise ValueError("ToolParameter required must be a bool.")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("ToolParameter description must be a non-empty string.")
        try:
            json.dumps(self.default)
        except (TypeError, ValueError) as exc:
            raise ValueError("ToolParameter default must be JSON-serialisable.") from exc

    def to_schema(self) -> dict[str, Any]:
        """JSON-serialisable public schema for this parameter."""
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "description": self.description,
            "default": self.default,
        }


@dataclass(frozen=True)
class ToolSpec:
    """An explicit tool specification: public metadata + the callable to dispatch.

    The raw ``function`` is internal and is never exposed in a public schema.
    """

    name: str
    description: str
    parameters: tuple[ToolParameter, ...]
    function: Callable[..., ToolResult]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ToolSpec name must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("ToolSpec description must be a non-empty string.")
        if not callable(self.function):
            raise ValueError("ToolSpec function must be callable.")
        if not isinstance(self.parameters, tuple) or not all(
            isinstance(p, ToolParameter) for p in self.parameters
        ):
            raise ValueError("ToolSpec parameters must be a tuple of ToolParameter.")
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("ToolSpec has duplicate parameter names.")
        if CLEAN_DF_PARAM in names:
            raise ValueError(f"{CLEAN_DF_PARAM!r} must never be a tool parameter.")

    def to_schema(self) -> dict[str, Any]:
        """JSON-serialisable public schema (excludes the raw function and ``clean_df``)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_schema() for p in self.parameters],
        }


class ToolRegistry:
    """Registers tool specs and dispatches execution with shallow validation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # --- registration -------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        """Register a tool. Raises ``ValueError`` on a duplicate name (dev error)."""
        if not isinstance(spec, ToolSpec):
            raise ValueError("register() expects a ToolSpec.")
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name!r} is already registered.")
        self._tools[spec.name] = spec

    # --- introspection ------------------------------------------------------

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def schema(self, name: str) -> dict[str, Any] | None:
        spec = self._tools.get(name)
        return spec.to_schema() if spec is not None else None

    def schemas(self) -> list[dict[str, Any]]:
        """Public schemas in deterministic registration order."""
        return [spec.to_schema() for spec in self._tools.values()]

    # --- execution ----------------------------------------------------------

    def execute(
        self, name: str, args: Mapping[str, Any] | None = None, *, clean_df: Any
    ) -> ToolResult:
        """Dispatch a registered tool with shallow argument-shape validation.

        ``args`` may be any read-only mapping (e.g. the validator's MappingProxyType
        output) or ``None``; non-mappings are rejected. ``clean_df`` is injected
        keyword-only. Normal request problems return a structured registry-level error;
        the underlying tool's result is returned unchanged on success.
        """
        spec = self._tools.get(name)
        if spec is None:
            return self._error(
                "Unknown tool requested.",
                requested_tool=name,
                available_tools=sorted(self._tools),
            )

        if args is None:
            args = {}
        # Accept any read-only mapping (e.g. the validator's MappingProxyType output),
        # not just a plain dict — non-mappings (list/str/…) are still rejected.
        if not isinstance(args, Mapping):
            return self._error("Arguments must be a dictionary or None.", requested_tool=name)
        if CLEAN_DF_PARAM in args:
            return self._error(
                f"{CLEAN_DF_PARAM!r} must not be provided in args; it is injected.",
                requested_tool=name,
            )

        allowed = {p.name for p in spec.parameters}
        required = {p.name for p in spec.parameters if p.required}
        unexpected = sorted(set(args) - allowed)
        if unexpected:
            return self._error(
                "Unexpected argument(s).", requested_tool=name, unexpected=unexpected
            )
        missing = sorted(required - set(args))
        if missing:
            return self._error(
                "Missing required argument(s).", requested_tool=name, missing=missing
            )

        try:
            return spec.function(clean_df, **args)
        except Exception as exc:  # noqa: BLE001 - registry boundary; surfaced structurally
            return self._error(
                "Tool execution failed.",
                requested_tool=name,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            )

    @staticmethod
    def _error(message: str, **extra: Any) -> ToolResult:
        """Build a structured registry-level error result."""
        result = error_result(REGISTRY_TOOL_NAME, message)
        if extra:
            result["result"].update(extra)
        return result


# --- Default registry: the six analytical tools (explicit registration) -----

def _team_window_params() -> tuple[ToolParameter, ...]:
    """The standard (team, optional window) parameter pair shared by team-level tools."""
    return (
        ToolParameter(name="team", type="str", required=True, description="Canonical team name."),
        ToolParameter(
            name="window", type="int|null", required=False,
            description="Optional recent-game window. If omitted, all available games are used.",
            default=None,
        ),
    )


def build_default_registry() -> ToolRegistry:
    """Construct a registry with the six analytical tools, registered explicitly.

    Registration order is the logical project order (not alphabetical) and is the order
    in which ``schemas()`` lists them.
    """
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="team_average_points",
        description="Average points scored by a team over all available games or the last N games.",
        parameters=_team_window_params(),
        function=team_average_points,
    ))
    registry.register(ToolSpec(
        name="average_points_allowed",
        description="Average points allowed by a team over all available games or the last N games.",
        parameters=_team_window_params(),
        function=average_points_allowed,
    ))
    registry.register(ToolSpec(
        name="team_record",
        description="Win-loss record and win percentage for a team over all available games or the last N games.",
        parameters=_team_window_params(),
        function=team_record,
    ))
    registry.register(ToolSpec(
        name="top_scoring_teams",
        description="Rank teams by average points scored, optionally within a dataset season_id.",
        parameters=(
            ToolParameter(name="n", type="int", required=False,
                          description="Number of top teams to return.", default=5),
            ToolParameter(name="season_id", type="int|null", required=False,
                          description="Optional opaque dataset season identifier.", default=None),
        ),
        function=top_scoring_teams,
    ))
    registry.register(ToolSpec(
        name="head_to_head",
        description="Head-to-head meetings, record, and scoring summary for two teams from team_a's perspective.",
        parameters=(
            ToolParameter(name="team_a", type="str", required=True,
                          description="Canonical name of the first team. Results are from this team's perspective."),
            ToolParameter(name="team_b", type="str", required=True,
                          description="Canonical name of the opposing team."),
            ToolParameter(name="window", type="int|null", required=False,
                          description="Optional recent-meeting window. If omitted, all meetings are used.",
                          default=None),
        ),
        function=head_to_head,
    ))
    registry.register(ToolSpec(
        name="team_efficiency_summary",
        description="Average per-game offensive rating, defensive rating, net rating, and possessions for a team.",
        parameters=_team_window_params(),
        function=team_efficiency_summary,
    ))
    return registry


DEFAULT_REGISTRY = build_default_registry()


# --- thin module-level convenience wrappers (delegate to DEFAULT_REGISTRY) ---

def get(name: str) -> ToolSpec | None:
    return DEFAULT_REGISTRY.get(name)


def is_registered(name: str) -> bool:
    return DEFAULT_REGISTRY.is_registered(name)


def schema(name: str) -> dict[str, Any] | None:
    return DEFAULT_REGISTRY.schema(name)


def schemas() -> list[dict[str, Any]]:
    return DEFAULT_REGISTRY.schemas()


def execute(name: str, args: Mapping[str, Any] | None = None, *, clean_df: Any) -> ToolResult:
    return DEFAULT_REGISTRY.execute(name, args, clean_df=clean_df)
