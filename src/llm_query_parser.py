"""Optional, offline LLM-ready query interpretation (an extension point, disabled by default).

This module shows how a future large language model could turn flexible natural language into a
structured tool intent that then flows through the **existing** validator, registry, and pandas
tools — so the language model never computes a statistic, never resolves an ambiguous team, never
executes a tool, and never writes the final answer. The deterministic rule parser
(``src.rule_parser.parse_rule_query``) remains the default and only enabled parser.

This module is used **only when explicitly injected** (e.g. ``AssistantRuntime(parser=...)``) with a
provider. No provider is bundled: there is no API key, no network call, and no third-party SDK. The
provider is an injected ``Callable[[str], str]`` (prompt -> raw model text); tests inject a
deterministic fake. Provider output is parsed STRICTLY and fails closed; the validator remains the
single safety gate. Output reuses the existing ``RuleParseResult`` / ``ParsedIntent`` contracts with
``parser_mode="llm"``, so the rest of the pipeline is parser-agnostic.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Optional

from src.intent_types import PARSER_MODE_LLM, ParsedIntent
from src.rule_parser_types import UNSUPPORTED_QUERY, ParseError, RuleParseResult
from src.rule_query_catalogue import SUPPORTED_TOOL_NAMES

# A future LLM call: prompt -> raw model text. Injected for testing; NEVER bundled (no SDK, no key).
LLMProvider = Callable[[str], str]

# The only tool names the interpreter may emit (mirrors the registry's supported tools).
ALLOWED_TOOLS: tuple[str, ...] = tuple(SUPPORTED_TOOL_NAMES)

# Argument concepts the current system understands (for the prompt schema). The validator enforces,
# per tool, which of these are actually allowed and rejects anything else.
ALLOWED_ARGUMENT_CONCEPTS: tuple[str, ...] = (
    "team", "team_a", "team_b", "window", "location", "n", "season_id",
)

# The only top-level keys the interpreter may emit. The envelope is STRICT: any other key (e.g.
# "answer", "result", "explanation") fails closed, so a model can never smuggle a fabricated answer
# past the parser. "confidence" is optional advisory metadata only. Argument *semantics* remain the
# validator's responsibility — argument keys are NOT restricted here (they pass through to the
# validator, which rejects unexpected ones per tool).
ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"tool", "arguments", "confidence"})


def build_intent_prompt(query: str) -> str:
    """Build the instruction a future LLM would receive to map a query to a structured intent.

    Pure string construction — no provider call, no network, no side effects. The prompt encodes the
    safety contract: emit ONLY strict JSON, only the listed tools, preserve raw team text, never
    resolve ambiguity or fix typos, never compute a statistic, never give betting advice, never write
    a final answer.
    """
    tools = "\n".join(f"  - {name}" for name in ALLOWED_TOOLS)
    return (
        "You convert ONE NBA analytics question into a single structured tool intent.\n"
        "Output STRICT JSON ONLY — no prose, no markdown, no explanation, no final answer.\n\n"
        "Allowed tools (choose exactly one; never invent a tool):\n"
        f"{tools}\n\n"
        "Allowed argument keys: team, team_a, team_b, window (integer), location ('home' or 'away'),"
        " n (integer), season_id (integer). Include only the arguments the question specifies.\n\n"
        "Rules:\n"
        "  - Preserve the team exactly as written (e.g. 'LA', 'Warriors'); do NOT resolve, expand,"
        " or disambiguate it.\n"
        "  - Do NOT correct spelling or typos.\n"
        "  - Do NOT compute, estimate, or include any statistic, score, or number you were not"
        " explicitly given.\n"
        "  - Do NOT provide betting advice, odds, predictions, or recommendations.\n"
        "  - Do NOT add any other top-level field (no 'answer', 'result', 'explanation', or score).\n"
        "  - If the question is not answerable by exactly one listed tool, output {\"tool\": null}.\n\n"
        "Output shape (these top-level keys only):\n"
        "  {\"tool\": \"<tool_name>\", \"arguments\": { ... }}\n\n"
        f"Question: {query}\n"
        "JSON:"
    )


def _no_parse(query: str, message: str) -> RuleParseResult:
    """A safe, fail-closed no-parse result (the validator/formatter handle the user-facing message)."""
    return RuleParseResult.no_parse((ParseError(UNSUPPORTED_QUERY, message),), raw_query=query)


def _strip_one_code_fence(text: str) -> str:
    """Strip a single surrounding ``` or ```json fence if present; otherwise return text unchanged.

    A pragmatic allowance for models that wrap JSON in one code fence. Anything else (prose around
    the JSON, multiple objects) is left intact so strict ``json.loads`` rejects it."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:]  # drop the opening ``` / ```json line
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _coerce_confidence(value: object) -> Optional[float]:
    """Advisory only — captured as metadata, never used to gate validation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0.0 <= float(value) <= 1.0 else None


def parse_llm_query(query: str, provider: Optional[LLMProvider] = None) -> RuleParseResult:
    """Interpret a query into a candidate intent via an injected provider; fail closed otherwise.

    Returns the SAME ``RuleParseResult`` the rule parser returns: a ``parsed`` result wrapping a
    ``ParsedIntent(parser_mode="llm")`` for structurally-valid output, or a ``no_parse`` result for an
    empty query / missing provider / invalid JSON / unexpected top-level field / unknown tool /
    malformed shape. NEVER raises, NEVER executes a tool, NEVER resolves teams or computes
    statistics — the validator remains the safety gate.
    """
    if not isinstance(query, str):
        return _no_parse("", "Query must be a string.")
    if not query.strip():
        return _no_parse(query, "The query was empty.")  # never ask the provider to interpret nothing
    if provider is None:
        return _no_parse(query, "LLM query interpretation is not configured (no provider).")

    try:
        raw = provider(build_intent_prompt(query))
    except Exception:  # noqa: BLE001 - provider boundary; never leak provider failures
        return _no_parse(query, "The LLM provider failed; falling back safely.")
    if not isinstance(raw, str):
        return _no_parse(query, "The LLM provider returned a non-text response.")

    try:
        payload = json.loads(_strip_one_code_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return _no_parse(query, "The LLM response was not strict JSON.")

    if not isinstance(payload, Mapping):
        return _no_parse(query, "The LLM response was not a JSON object.")
    unexpected = set(payload) - ALLOWED_TOP_LEVEL_KEYS
    if unexpected:
        # A strict envelope: reject smuggled fields (e.g. a fabricated "answer") rather than ignore them.
        return _no_parse(query, f"The LLM response had unexpected top-level field(s): {sorted(unexpected)}.")
    tool = payload.get("tool")
    if tool is None:
        return _no_parse(query, "The query was not mapped to a supported tool.")
    if not isinstance(tool, str) or tool not in ALLOWED_TOOLS:
        return _no_parse(query, f"The LLM proposed an unsupported tool {tool!r}.")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, Mapping):
        return _no_parse(query, "The LLM 'arguments' field was not a JSON object.")

    # Raw candidate only: arguments pass through UNCHANGED (no canonicalisation, no typo fixing, no
    # argument filtering) so the existing validator detects ambiguity, unknown teams, and bad args.
    try:
        intent = ParsedIntent(
            tool_name=tool,
            arguments=dict(arguments),
            parser_mode=PARSER_MODE_LLM,
            raw_query=query,
            confidence=_coerce_confidence(payload.get("confidence")),
        )
    except (ValueError, TypeError):
        return _no_parse(query, "The LLM produced a malformed intent.")
    return RuleParseResult.parsed(intent, raw_query=query)


class LLMQueryParser:
    """A callable ``(str) -> RuleParseResult`` wrapping :func:`parse_llm_query` with a bound provider.

    Convenient for runtime injection, e.g. ``AssistantRuntime(parser=LLMQueryParser(provider))``.
    With no provider it fails closed (every query -> ``no_parse``)."""

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self._provider = provider

    def parse(self, query: str) -> RuleParseResult:
        return parse_llm_query(query, provider=self._provider)

    __call__ = parse
