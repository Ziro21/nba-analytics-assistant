# LLM Integration Design — Optional Query Interpretation

This document describes how a large language model (LLM) *could* improve the assistant's query
interpretation, and the small, **optional, offline, disabled-by-default** interface that demonstrates
it safely. It exists because the assignment asks the report to discuss LLM integration for query
interpretation or response generation. The implementation focuses on **query interpretation only**;
response generation is deliberately deferred (see §7).

This is a **validator-gated extension point**, not an enabled feature. No model, provider SDK, API
key, or network call is bundled. The default assistant remains fully deterministic and offline.

## 1. Why the default assistant remains deterministic and rule-based

The default parser is `src.rule_parser.parse_rule_query`: a deterministic rule engine (normalise →
route → extract slots). It is reproducible (same input → same output), needs no network or key, runs
offline, and is fully testable. For a graded analytics tool where every number must be trustworthy
and every test must run anywhere, a deterministic default is the correct, safe choice. The LLM
interface never replaces it — it is an alternative front end that produces the *same* structured
intent for the *same* downstream pipeline.

## 2. How an optional LLM can improve query interpretation

A rule parser only understands phrasings it was written for. An LLM front end could map flexible,
unseen natural language onto the existing tools — e.g. *"How have Golden State been doing on the road
lately?"* → `team_advanced_profile(team="Golden State", location="away")`. The LLM's only job is to
**propose structure**: pick one supported tool and copy the relevant argument strings out of the
question. It interprets; it never computes.

## 3. Why the LLM must output structured JSON tool intents

The boundary between an LLM and this deterministic system must be a strict, machine-checkable
contract. The interface (`src/llm_query_parser.py`) builds a prompt (`build_intent_prompt`) that
instructs the model to emit **strict JSON only**:

```json
{ "tool": "team_advanced_profile", "arguments": { "team": "Golden State", "location": "away", "window": 5 } }
```

`parse_llm_query` then parses that JSON **strictly** and **fails closed** on anything else (empty
query, invalid JSON, prose around JSON, non-object, missing/non-string `tool`, an unknown tool,
non-dict `arguments`, or no provider). The result reuses the existing `RuleParseResult` /
`ParsedIntent` contract with `parser_mode="llm"`, so the rest of the pipeline is parser-agnostic.

**Strict envelope, raw arguments.** The parser owns the *envelope* and the validator owns the
*argument semantics*:

- The JSON object may contain **only** the top-level keys `tool`, `arguments`, and an optional
  advisory `confidence`. Any other top-level field (e.g. a smuggled `"answer"`, `"result"`, or
  `"sql"`) is **rejected**, not ignored — a model can therefore never slip a fabricated answer past
  the parser. `confidence` is captured as metadata only and never gates validation.
- `arguments` are passed through **unchanged** (no canonicalisation, typo-fixing, or filtering), so
  the validator still sees raw values and rejects ambiguous teams, unknown teams, and unexpected
  argument keys exactly as for the rule parser.

**Deliberate code-fence tolerance.** One clean surrounding ```` ```json ```` fence is stripped before
parsing, because real models very commonly wrap JSON in a single fence even when told not to.
**Security note:** this is a parser-level convenience only — it is *not* accepted if any prose
surrounds the JSON, and a fence with trailing text or multiple JSON objects still fails strict
parsing and is rejected. The accepted contract is therefore "a strict JSON object, optionally inside
one clean code fence."

## 4. Why the validator remains the safety gate

The LLM parser performs only **structural** safety (valid JSON, a known tool, a dict of arguments).
All **semantic** safety stays with the existing `validate_intent`:

- Ambiguous teams are flagged, never auto-resolved. The parser **preserves the raw team string**
  (e.g. `"LA"`), so the validator still reports `ambiguous_team` and asks the user to choose.
- Unknown teams produce suggestions only; typos are not auto-corrected.
- Exhibition teams, invalid windows, invalid/`neutral` locations, and unexpected arguments are all
  rejected before any tool runs.

Because `parser_mode` is metadata only (the validator is parser-mode invariant by design), an
identical intent validates and executes identically whether it came from the rule parser or the LLM
parser. A test asserts this equivalence.

## 5. Why the registry remains the only tool execution route

The parser never calls a tool. A validated intent is dispatched through the existing `ToolRegistry`,
which is the single execution path. The LLM cannot invent a tool (the parser rejects unknown names)
and cannot reach a tool except through the registry after validation succeeds.

## 6. Why pandas tools remain the source of truth

Every statistic is computed by the pandas tools on the clean dataframe. The LLM never produces,
estimates, or includes a number — the prompt forbids it and the parser would reject answer-like text
(it is not valid intent JSON). The model proposes *which* question to ask; pandas answers it.

## 7. Why response generation is deferred

This phase implements query interpretation only. Response generation (an LLM rewriting the final
answer) is intentionally **not** built, because the deterministic formatter keeps answers
reproducible, an LLM could over-interpret or add unsupported claims, and betting-style narration is
exactly the kind of wording this project avoids. The current formatter is safer for assessment and
testing.

## 8. How response generation could be added safely in future

A future response layer could rephrase an **already-computed** tool result for readability — but it
must use **only the values returned by the pandas tools**, add no new numbers or claims, and pass a
"no new facts" check before display. It would live in a separate module (e.g.
`response_formatter_llm`, which **does not exist** today and is guarded against by tests), be
disabled by default, and never run before tool execution.

## 9. Risk register

| Risk | Mitigation in this design |
|---|---|
| **Prompt injection** (the question tries to change the instruction) | The query is untrusted input; even a manipulated model output is gated — the parser only accepts a known tool, and the validator re-checks teams/arguments. Blast radius is bounded to "a safe failure or a validated tool call". |
| **Hallucinated tool** | The parser rejects any tool not in the registry's supported set (`no_parse`). |
| **Hallucinated / wrong team** | The parser preserves raw team text; the validator resolves or rejects it (unknown → suggestions only). |
| **Ambiguous team names** (LA, NY) | Preserved raw → validator returns `ambiguous_team`; never auto-resolved. |
| **Typos** | Not corrected by the LLM layer; validator's suggestion-only fuzzy matching applies. |
| **Non-determinism of real models** | Output is re-validated on every call; nothing is cached or trusted. The deterministic rule parser remains the default. |
| **Latency / cost / availability** | Default stays offline and rule-based; the LLM path is optional and fails closed when no provider is configured. |
| **Unsupported betting/prediction requests** | The prompt forbids betting advice; invented betting tools are rejected by the parser; no betting tool exists in the registry. |
| **Response-generation risk** | Not implemented; guarded by scope tests (no `response_formatter_llm`). |

## 10. Limitations

- **Disabled by default.** The CLI default and `--parser rule` use the rule engine. `--parser llm`
  fails closed with a clear "not configured" message (no provider is bundled).
- **No provider, no key, no network, no SDK, no new dependency.** The provider is an injected
  `Callable[[str], str]`; tests use a deterministic fake.
- **Interpretation only.** No LLM-written answers, explanations, summaries, or betting narration.
- **No agent, RAG, tool-calling framework, or live data.**

## How it is exercised

- **Runtime injection** (the intended integration): `build_default_runtime(parser=LLMQueryParser(provider))`
  or `AssistantRuntime(parser=...)`. The candidate intent flows through `validate_intent → registry →
  pandas tools → deterministic formatter`.
- **Rule → LLM fallback** (recommended future wiring, not enabled by default): try the rule parser
  first; if it returns `no_parse`/`unsupported` and an LLM parser is injected, try the LLM parser, then
  validate its candidate with the same validator. The seam (`answer_query(parser=...)`) supports this;
  no provider is bundled.
- **Tests:** `tests/test_llm_query_parser.py` — strict parsing, unknown-tool/ambiguity/betting safety,
  no-provider fail-closed, and validator-gated end-to-end with a fake provider.
