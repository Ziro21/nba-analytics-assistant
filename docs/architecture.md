# Architecture

The Sporting Risk NBA Analytics Assistant is a deterministic pipeline of small, single-purpose
layers. Each layer has one responsibility and a narrow, tested boundary. No layer reaches past
the next; statistics are produced only by the pandas tools.

## High-level chain

```
Parser     extracts candidate structure from the query.
Validator  canonicalises teams and protects (rejects invalid/ambiguous requests).
Registry   dispatches a validated request to exactly one tool.
Tools      calculate from the clean dataframe (pandas — the only source of truth).
Formatter  explains an already-produced result; it never calculates.
Assistant  coordinates the pipeline; it loads no data and computes nothing.
Runtime    bootstraps dependencies (loads/validates the dataset, builds the context).
CLI        collects the query and displays the result.
```

## Component responsibilities

| Module | Responsibility |
| --- | --- |
| `src/rule_parser.py` | Parse a raw query into a `ParsedIntent` (tool + raw candidate arguments) or a structured parse failure. Extracts only — no validation, canonicalisation, execution, or data access. |
| `src/intent_validator.py` | Validate a `ParsedIntent` against a reference context: canonicalise team names, reject ambiguous/unknown/special teams, check argument types and domain rules. The single safety boundary. |
| `src/tool_registry.py` | Hold the six registered tools and dispatch a validated call. The only path to tool execution. |
| `src/tools.py` | The six pandas analytical tools. The only layer that computes statistics, all from the clean dataframe. |
| `src/response_formatter.py` | Convert a tool result / parse failure / validation failure into an `AssistantResult`. Pure formatting; computes nothing; fails closed on malformed input. |
| `src/assistant.py` | Orchestrate parse → validate → execute → format. Thin coordination; imports no pandas, loads no data, calls no tool directly. |
| `src/assistant_runtime.py` | Bootstrap: load and validate the dataset, build the clean view and validation context, and hold the prepared dependencies. The only place that loads data. |
| `src/cli.py` | A thin command-line demo: collect a query, call the runtime, print the result, return an exit code. No assistant logic. |
| `src/assistant_types.py` | The `AssistantIssue` and `AssistantResult` contracts: frozen, JSON-safe, mutation-safe. |

## Safety boundaries

- The **parser** does not validate, canonicalise, execute, or load data — it emits raw candidates
  (e.g. `LA` and typos pass through unchanged) for the validator to judge.
- The **validator** owns canonicalisation and all team-safety decisions (resolved / ambiguous /
  unknown / invalid special team) and argument/domain checks.
- The **registry** is the only dispatch path to a tool; nothing executes a tool directly.
- The **tools** are the only place statistics are calculated, and always from the clean dataframe.
- The **formatter** never executes a tool or computes a statistic; malformed input fails closed to
  an internal-error result.
- The **assistant** never loads data, never imports pandas, never calls a tool or the resolver
  directly; registry execution happens only after a successful parse and a successful validation.
- The **runtime** owns all data loading and dependency construction.
- The **CLI** only collects input and displays output.

These boundaries are enforced by import-scope tests (e.g. importing `src.assistant` pulls in no
pandas, data, tools, or registry; importing `src.cli` pulls in nothing heavy until it runs).

### Validator priority model

The validator can accumulate **multiple independent issues** for one query (for example an ambiguous
team *and* a malformed window). All of them are returned in the structured `errors` list, but the
formatter selects a single **primary** issue for the top-level clarification sentence, in this fixed
priority order (highest first):

1. `ambiguous_team` — the surface matches more than one franchise (e.g. "LA", "New York");
2. `unknown_team` — no confident match (suggestions are offered, never auto-applied);
3. `invalid_special_team` — an exhibition side (Team Stars/Stripes/World) is not a franchise;
4. `same_team_head_to_head` — a head-to-head needs two *different* teams;
5. `missing_required_argument`;
6. `invalid_window` / `invalid_n` / `invalid_season_id`;
7. any other validation error (fallback: the first issue reported).

Team-identity problems are surfaced before argument-shape problems because naming the right team is
the precondition for a meaningful answer. **Ambiguous and unknown teams are never auto-resolved** —
the assistant asks the user to choose, and suggestions are guidance only. Execution happens **only
after validation succeeds**, so a lower-priority issue can never leak a guessed statistic.

### Parser fallback design (safe-by-validator)

The rule parser is deterministic and conservative. It extracts a candidate tool plus **raw** argument
strings; it does **not** canonicalise teams, validate, or execute. A gazetteer handles known team
surfaces; a narrow, precision-gated structural fallback may additionally pick up a single
unrecognised token as a *team-like candidate* so a near-miss is surfaced rather than silently dropped.

This fallback is safe because **every raw argument still passes through the validator**. A fallback
false positive therefore becomes an `unknown_team` (or another validation error) and a clarification
request — never a guessed answer. For example, "Who are the champions?" does not map to a supported
metric and is reported as unsupported/unknown rather than executed. The parser never claims to
understand arbitrary natural language; it either maps a query to a supported family or fails safely.

## Data flow

```
0. Runtime fingerprints the dataset file    (validate_dataset_fingerprint, SHA-256 over raw bytes)
1. Runtime loads the raw dataset            (load_raw_dataset)
2. Runtime validates the raw dataset        (validate_dataset)
3. Runtime builds the clean view            (build_clean_view)
4. Runtime validates the clean view         (validate_clean_view)
5. Runtime builds the validation context    (build_validation_context, against the registry)
6. Runtime holds {clean_df, validation_context, registry, dataset_fingerprint}
7. Each query: assistant parses → validates → registry executes a tool on clean_df → formats
```

The step-0 fingerprint compares the dataset file's SHA-256 to `EXPECTED_DATASET_SHA256`. By default a
mismatch is recorded as bootstrap metadata (a warning on the runtime's `dataset_fingerprint`) and
startup proceeds, so a reviewer can intentionally run a different file; `build_default_runtime(...,
strict_dataset_hash=True)` instead raises before any work. The check reads raw bytes only and changes
no statistic — pandas remains the sole source of truth for every number.

## Result contract

Every response is an `AssistantResult`; `AssistantResult.to_dict()` is always JSON-serialisable:

```python
{
    "status": "answer | clarification_needed | unsupported | error",
    "message": "a user-facing sentence",
    "query": "the original query",
    "tool_name": "the tool used, or null",
    "data": { ... },        # tool result payload, or null
    "errors": [ ... ],      # structured AssistantIssue items (empty for an answer)
    "warnings": [ ... ],    # non-blocking notes, e.g. canonicalisation
    "meta": { ... },        # window / season / date-range metadata, or null
}
```

`status` is `answer` only when an analytical question was fully understood, validated, and
executed. Everything else is a safe, structured failure.

## Why deterministic design matters

- **Repeatability** — identical input always yields identical output; no hidden state, no model
  drift, no network.
- **Testability** — every layer is independently testable, and the whole chain is locked by
  oracle-backed acceptance tests.
- **No hallucinated statistics** — the language layer cannot invent a number; only pandas tools
  compute, and only from the dataset.
- **Controlled scope** — unsupported questions fail clearly instead of producing a plausible but
  wrong answer.
- **Clear failure modes** — ambiguous teams, unknown teams, vague time ranges, and same-team
  head-to-head each produce a specific, explainable result.
