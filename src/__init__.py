"""Sporting Risk NBA Analytics Assistant — source package.

A deterministic, tool-based NL analytics assistant over a structured NBA CSV.
pandas is the only source of truth for statistics; the LLM (optional mode) only
maps a question to a registered tool and its arguments — it never computes.

Note: tool registration is explicit (no import side-effects). The registry is built
by explicitly registering each tool's `ToolSpec`; this package does not import tools to
trigger hidden registration.
"""

__all__: list[str] = []
