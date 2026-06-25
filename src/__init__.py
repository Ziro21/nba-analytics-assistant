"""Sporting Risk NBA Analytics Assistant — source package.

A deterministic, tool-based NL analytics assistant over a structured NBA CSV.
pandas is the only source of truth for statistics; the LLM (optional mode) only
maps a question to a registered tool and its arguments — it never computes.

Note: once the tool registry exists (Phase 6), the tools module is imported here
so that `@tool` registration fires at import time. That import is intentionally
absent until the registry and tools are implemented.
"""

__all__: list[str] = []
