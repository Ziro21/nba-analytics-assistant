"""Tool result contract and lightweight constructors.

Every analytical tool (built later, in Phases 5B–5G) returns the same JSON-serialisable
dict shape. These constructors keep that shape consistent and typo-free. No tool logic,
no prose, no rounding here — this is purely the structured envelope.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

Status = Literal["ok", "no_data", "error"]


class ResultMeta(TypedDict):
    """Standard metadata block carried by every tool result."""

    team: Optional[str]
    games_used: Optional[int]
    date_range: Optional[list[str]]
    window_requested: Optional[int]
    season_id: Optional[int]


class ToolResult(TypedDict):
    """The exact top-level shape every tool returns."""

    status: Status
    tool: str
    result: dict[str, Any]
    meta: ResultMeta
    warnings: list[str]


def build_meta(
    team: Optional[str] = None,
    games_used: Optional[int] = None,
    date_range: Optional[list[str]] = None,
    window_requested: Optional[int] = None,
    season_id: Optional[int] = None,
) -> ResultMeta:
    """Return a standard metadata block; missing fields default to ``None``.

    Numeric fields are coerced to plain ``int`` and ``date_range`` to a list of strings
    so the result stays JSON-serialisable even if callers pass numpy/pandas scalars.
    """
    return {
        "team": team,
        "games_used": int(games_used) if games_used is not None else None,
        "date_range": [str(d) for d in date_range] if date_range is not None else None,
        "window_requested": int(window_requested) if window_requested is not None else None,
        "season_id": int(season_id) if season_id is not None else None,
    }


def _normalise(
    meta: Optional[ResultMeta], warnings: Optional[list[str]]
) -> tuple[ResultMeta, list[str]]:
    return (meta if meta is not None else build_meta()), (list(warnings) if warnings else [])


def ok_result(
    tool: str,
    result: dict[str, Any],
    meta: Optional[ResultMeta] = None,
    warnings: Optional[list[str]] = None,
) -> ToolResult:
    """Build a successful result (``status == "ok"``)."""
    meta, warnings = _normalise(meta, warnings)
    return {"status": "ok", "tool": tool, "result": dict(result), "meta": meta, "warnings": warnings}


def no_data_result(
    tool: str,
    result: Optional[dict[str, Any]] = None,
    meta: Optional[ResultMeta] = None,
    warnings: Optional[list[str]] = None,
) -> ToolResult:
    """Build a no-data result (``status == "no_data"``)."""
    meta, warnings = _normalise(meta, warnings)
    return {
        "status": "no_data",
        "tool": tool,
        "result": dict(result) if result is not None else {},
        "meta": meta,
        "warnings": warnings,
    }


def error_result(
    tool: str,
    message: str,
    meta: Optional[ResultMeta] = None,
    warnings: Optional[list[str]] = None,
) -> ToolResult:
    """Build an error result (``status == "error"``) carrying a clear message."""
    meta, warnings = _normalise(meta, warnings)
    return {
        "status": "error",
        "tool": tool,
        "result": {"message": message},
        "meta": meta,
        "warnings": warnings,
    }
