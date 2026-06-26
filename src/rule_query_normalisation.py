"""Deterministic query normalisation for the rule parser (Phase 8B).

Shared by the intent router (8B) and the slot extractor (8C). Pure and offline: lower-case,
strip simple punctuation (so ``head-to-head`` -> ``head to head``, ``vs.`` -> ``vs``,
``win-loss`` -> ``win loss``), collapse whitespace, preserve digits (``last 5``/``top 5``/
``season 26``/``76ers``). No canonicalisation, no spelling correction, no fuzzy matching,
no data, no registry/validator/resolver/LLM imports.
"""

from __future__ import annotations

import re

_NON_ALNUM_SPACE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def normalise_query_text(query: str) -> str:
    """Lower-case, strip simple punctuation, and collapse whitespace. Digits preserved."""
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    text = query.lower()
    text = _NON_ALNUM_SPACE.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def query_tokens(query: str) -> tuple[str, ...]:
    """The normalised query split into whitespace-delimited tokens."""
    normalised = normalise_query_text(query)
    return tuple(normalised.split()) if normalised else ()
