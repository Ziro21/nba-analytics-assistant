"""Sporting Risk NBA Analytics Assistant — source package.

A deterministic, tool-based natural-language analytics assistant over a structured NBA CSV.
pandas is the only source of truth for statistics; the rule-based parser only maps a question to a
registered tool and its arguments — nothing in the language layer ever computes a number.

Note: tool registration is explicit (no import side-effects). The registry is built by explicitly
registering each tool's `ToolSpec`; this package does not import tools to trigger hidden registration.
"""

from __future__ import annotations

# Single source of truth for the package version.
__version__ = "1.3.0"

__all__ = ["__version__"]
