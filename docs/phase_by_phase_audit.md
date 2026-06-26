# Phase-by-Phase Technical Audit

A post-release (v1.0.0) technical audit of the Sporting Risk NBA Analytics Assistant. This is a
review document only — it changes no production behaviour. Grades are deliberately not inflated.

## Executive summary

The released v1.0.0 is a complete, deterministic, well-tested NBA analytics assistant. The
architecture holds to strict single-responsibility boundaries (parser → validator → registry →
pandas tools → formatter → assistant → runtime → CLI), pandas is the only source of statistics,
and unsupported or ambiguous input fails safely with specific, actionable messages. The full test
suite (1128 tests) passes offline and deterministically, and import-scope guards keep the assistant,
formatter, and CLI free of pandas/data/tools.

There are **no must-fix defects and no release blockers**. The improvement opportunities are
quality/polish items for a future v1.1.0: a few dead configuration constants, the per-invocation
dataset bootstrap cost, a dataset-integrity (hash) guard, the precision of the typo fallback, and
minor CLI/docs polish. Overall the project is at a strong enterprise/portfolio grade.

## Repository state

- Branch `main`; `HEAD = 72e3eee`; `origin/main` aligned (0 ahead / 0 behind).
- Tag `v1.0.0` points to `72e3eee` (the final release commit).
- Working tree clean; no `_working/`, `.venv`, `.env`, or caches tracked.
- ~4,665 lines of `src/`, ~8,844 lines of `tests/`, 673 test functions.

## Test baseline

- `python -m pytest tests/ -q` → **1128 passed**, ~11s, offline.
- CLI smokes: Warriors avg → 114.4 (exit 0); `--json` Celtics vs Heat → `answer`/`head_to_head`
  (exit 0); New York → "Do you mean New York Knicks or Brooklyn Nets?" (exit 1); LA → Lakers/Clippers
  (exit 1); "Who is better?" → unsupported (exit 1). No tracebacks.
- Import checks: `src.assistant`, `src.response_formatter`, `src.cli` → all clean (no
  pandas/data/tools/LLM/web/api/rag/agent).

---

## Phase-by-phase audit

## Phase 4 — Data loading, validation, clean view

### Purpose
Load the raw CSV, validate it, and build a stable, validated "clean view" dataframe that every
analytical tool reads from.

### Main files
`src/data_loader.py`, `src/data_validation.py`, `src/data_model.py`, `src/config.py`.

### Evidence inspected
The three modules, `tests/test_data_validation.py`, `tests/test_data_model.py`, `config.py`
constants, and runtime confirmation (14,746 clean rows, 30 canonical + 3 special teams).

### Correctness assessment
Drops the exported `Unnamed: 0` index, derives a 17-column clean view with two rows per game
(home/away), opponent derivation, win flag, ratings; validates raw and clean shapes; `season_id`
is treated as an opaque integer; exhibition (Team Stars/Stripes/World) rows are flagged and excluded
from franchise tools. Deterministic sort keys. No silent corruption observed.

### Enterprise-grade strengths
Clear separation of load vs validate vs model; explicit expected row/column counts; opaque
`season_id`; exhibition handling is explicit and never silently special-cased.

### Risks / weaknesses
- No dataset **integrity/version guard** (e.g. a content hash). A swapped CSV with the same shape
  would silently change every answer and the oracle tests, with no single "this is the expected
  dataset" assertion beyond row/column counts.
- Dataset assumptions are encoded as magic counts; a short data dictionary would aid reviewers.

### Test coverage assessment
Strong: shape, clean-view columns, two-rows-per-game, date format, exhibition flags. Missing:
an explicit dataset-fingerprint test.

### Architecture-boundary assessment
Clean: data modules own loading/validation; nothing downstream re-loads.

### Production-readiness grade: 90 / 100

### Recommended improvements
- Should fix: add a dataset content-hash check surfaced at bootstrap.
- Nice to have: a `docs/data_dictionary.md` describing each clean-view column.
- Future roadmap: dataset versioning / multiple-dataset support.

### Verdict: Green

## Phase 5 — Analytical pandas tools

### Purpose
The six pandas tools that compute every statistic from the clean view.

### Main files
`src/tools.py`, `src/tool_results.py`.

### Evidence inspected
`src/tools.py`, `tests/test_tools.py` (94), `tests/test_tools_integration.py` (47), the
`{status, tool, result, meta, warnings}` contract, oracle values.

### Correctness assessment
Six tools with a uniform result contract and locked status semantics (invalid argument → error;
valid-but-no-rows → no_data). Oracle-backed (e.g. Warriors record 289-223; Celtics vs Heat 25-14
across 39). No internal rounding (display rounding is the formatter's job). Exhibition exclusion and
windowing handled.

### Enterprise-grade strengths
Pandas is the only calculation layer; oracle tests lock the numbers; the result contract is uniform
and JSON-safe.

### Risks / weaknesses
- Each call filters the full dataframe by team — fine at 14k rows, but linear and uncached; would
  slow on a much larger dataset.
- Some per-tool result assembly repeats shape; shared helpers reduce but do not eliminate it.

### Test coverage assessment
Excellent (positive, invalid-arg, no-data, edge windows, h2h symmetry). Missing: explicit
performance/scale tests (acceptable for the dataset size).

### Architecture-boundary assessment
Clean: tools take `clean_df`, never print/round/mutate, never load data.

### Production-readiness grade: 92 / 100

### Recommended improvements
- Nice to have: precompute team-indexed groupings once for repeated queries (future REPL/server).
- Future roadmap: possession-weighted variants (separately designed/validated).

### Verdict: Green

## Phase 6 — Tool registry

### Purpose
A single dispatch layer holding the six tools and their schemas.

### Main files
`src/tool_registry.py`.

### Evidence inspected
`ToolParameter`/`ToolSpec`/`ToolRegistry`, `ALLOWED_PARAM_TYPES`, `tests/test_tool_registry.py`
(64), `tests/test_tool_registry_integration.py` (12).

### Correctness assessment
Self-validating frozen specs; allowlisted JSON-safe parameter types; `execute(name, args, *,
clean_df)` injects the dataframe keyword-only and accepts any Mapping; dispatch is the only path to
a tool. Request problems return structured results, not raw exceptions.

### Enterprise-grade strengths
The registry is the sole execution path; schemas are validated at construction (fail fast); the 6A
`ALLOWED_PARAM_TYPES` fix prevents non-JSON types leaking into schemas.

### Risks / weaknesses
- Parameter metadata is minimal (no human descriptions/examples) — fine internally, but a richer
  schema would aid an external API later.

### Test coverage assessment
Strong (registration, schema validation, dispatch, contract, integration).

### Architecture-boundary assessment
Clean: the registry depends on tools but nothing above it.

### Production-readiness grade: 92 / 100

### Recommended improvements
- Nice to have: optional per-parameter description/example fields.
- Future roadmap: versioned tool schemas.

### Verdict: Green

## Phase 7 — Validation safety boundary and team resolution

### Purpose
The single canonicalisation/safety layer between the parser and the registry.

### Main files
`src/intent_types.py`, `src/team_resolution.py`, `src/validation_context.py`, `src/intent_validator.py`.

### Evidence inspected
The four modules; `tests/test_team_resolution.py` (50), `tests/test_validation_context.py` (16),
`tests/test_intent_validator.py` (51), `tests/test_intent_validation_integration.py` (21).

### Correctness assessment
Resolves nicknames/tri-codes/unambiguous cities to canonical franchises; rejects ambiguous markets
(LA/NY), unknown teams (fuzzy suggestions only — never auto-resolved), special teams, and same-team
head-to-head; validates arg types and the opaque `season_id`; executes nothing. The curated
alias/ambiguity maps are validated against dataset-derived canonical teams at build time (fail fast).

### Enterprise-grade strengths
This is the heart of the AI-safety design: fuzzy matching produces suggestions only, never
execution; the boundary is dataset-validated and parser-mode invariant.

### Risks / weaknesses
- Fuzzy unknown-team suggestions can be noisy (e.g. "Celics" suggests Boston Celtics **and** New
  Orleans Pelicans). Harmless (suggestion-only) but slightly odd UX.
- The alias map is hand-curated; correctness depends on the drift tests staying in place.

### Test coverage assessment
Excellent (resolution, ambiguity, unknown/fuzzy, special, same-team, arg/domain, drift).

### Architecture-boundary assessment
Clean: validator imports the resolver + context, never the parser/registry/tools.

### Production-readiness grade: 93 / 100

### Recommended improvements
- Nice to have: tune the fuzzy cutoff / rank suggestions to reduce noise.
- Should fix (docs): document the validator priority model and alias coverage.

### Verdict: Green

## Phase 8 — Deterministic rule parser

### Purpose
Map a natural-language query to a candidate tool + raw argument slots, or fail safely.

### Main files
`src/rule_parser_types.py`, `src/rule_query_catalogue.py`, `src/rule_query_normalisation.py`,
`src/rule_intent_router.py`, `src/team_surface_catalogue.py`, `src/rule_slot_extractor.py`,
`src/rule_parser.py`.

### Evidence inspected
All seven modules; the per-module test files; `tests/test_rule_parser_phase8_final.py` (25); the
executable query catalogue; the drift-tested team-surface catalogue.

### Correctness assessment
Catalogue-driven; deterministic normalisation; priority routing; gazetteer longest-match slot
extraction emitting raw spans; explicit-number-only windows; vague time → `unsupported_time_
expression` (never all-games, never invented window); two-teams-for-a-single-tool flagged; the
parser validates/executes nothing and imports no pandas.

### Enterprise-grade strengths
The executable catalogue is the single source of truth; the team-surface catalogue is drift-tested
against the dataset and the resolver; "fail loudly, never guess" is enforced throughout.

### Risks / weaknesses
- The **precision-gated structural fallback** can extract an arbitrary unrecognised word as a team
  candidate (e.g. "champions"); the validator then rejects it (`unknown_team`), so it fails safe,
  but it is the largest edge-surface in the parser. Heavily tested, but inherently heuristic.
- Documented catalogue-vs-parser nuance: a few vague-time examples were idealised as `incomplete`
  in the 8A catalogue but deterministically return `no_parse` (no routable metric). Honest and
  tested, but a small inconsistency between the catalogue's declared status and runtime.

### Test coverage assessment
Very strong (catalogue, routing, slots, fallback false-positives, vague time, h2h, scope guards).

### Architecture-boundary assessment
Clean: subprocess-verified that the parser stack imports no validator/registry/tools/pandas.

### Production-readiness grade: 90 / 100

### Recommended improvements
- Nice to have: a documented parser coverage matrix (query family × surface form).
- Should fix (docs): note the fallback's safe-by-validator design prominently.
- Future roadmap: an optional LLM parser strictly behind the same validator (separately designed).

### Verdict: Green

## Phase 9 — Assistant contracts, formatter, orchestration, integration

### Purpose
The contracts, the deterministic formatter, and the thin orchestrator that runs the pipeline.

### Main files
`src/assistant_types.py`, `src/response_formatter.py`, `src/assistant.py`.

### Evidence inspected
The three modules; `tests/test_assistant_types.py` (37), `tests/test_response_formatter.py` (49),
`tests/test_assistant.py` (31), `tests/test_assistant_integration.py` (41),
`tests/test_assistant_phase9_final.py` (42).

### Correctness assessment
`AssistantResult`/`AssistantIssue` are frozen, JSON-safe, mutation-safe, invariant-checked; the
formatter maps ok/no_data/error/malformed correctly and fails closed (incl. the 9B malformed-warning
fix); the orchestrator runs parse → validate → execute → format with fail-closed handling at every
boundary; registry execution happens only after a successful parse and validation. No data loading,
no globals.

### Enterprise-grade strengths
A single structured result for every outcome; an explicit "execution only after parse+validation"
gate; thorough internal-failure and bad-dependency coverage.

### Risks / weaknesses
- The orchestrator's broad `except Exception` boundaries are correct for fail-closed but could, in a
  future debug mode, optionally surface a correlation id for diagnosis.

### Test coverage assessment
Excellent (contracts, formatter status mapping + fail-closed, gating, fakes, determinism, scope).

### Architecture-boundary assessment
Clean and machine-checked.

### Production-readiness grade: 93 / 100

### Recommended improvements
- Nice to have: optional structured `hint`/diagnostic id in error results (future).
- Future roadmap: message localisation.

### Verdict: Green

## Phase 10 — Runtime, CLI, documentation, delivery QA

### Purpose
The bootstrap runtime, the CLI demo, the documentation, and the delivery acceptance gate.

### Main files
`src/assistant_runtime.py`, `src/cli.py`, `README.md`, `docs/*`, `tests/test_assistant_runtime.py`,
`tests/test_cli.py`, `tests/test_documentation.py`, `tests/test_delivery_final.py`.

### Correctness assessment
`AssistantRuntime` holds injected dependencies and delegates to `answer_query`; bootstrap failures
raise (config errors), per-query failures stay fail-closed; the CLI is a thin printer with
deterministic exit codes (0/1/2) and a lazy runtime import; docs are accurate to real output.

### Enterprise-grade strengths
One-directional import (`runtime → assistant`, never the reverse); `src.cli` import pulls nothing
heavy until `main()` runs; delivery gate runs the real CLI end-to-end.

### Risks / weaknesses
- The runtime **re-loads and re-validates the full dataset on every CLI invocation** (~1–2s). Fine
  for a demo, but there is no caching/REPL for repeated queries — the main performance note.
- No `--version`, no `--examples`, no packaging entry point (`pyproject.toml`).

### Test coverage assessment
Strong (runtime build/answer/safe-failure, CLI args/output/exit codes via a fake runtime,
import-scope, delivery subprocess smokes).

### Architecture-boundary assessment
Clean and machine-checked from both directions.

### Production-readiness grade: 90 / 100

### Recommended improvements
- Should fix: add `--version`.
- Nice to have: `--examples`; a `pyproject.toml` with `src.cli:main` entry point; a simple REPL mode
  that bootstraps once.
- Future roadmap: a lightweight performance benchmark in CI.

### Verdict: Green

## Phase 11A — Final submission / portfolio release package

### Purpose
Reviewer-facing submission, release notes, portfolio summary, reviewer quickstart, and a guard test.

### Main files
`SUBMISSION.md`, `RELEASE_NOTES.md`, `docs/project_summary.md`, `docs/reviewer_quickstart.md`,
`tests/test_release_package.py`.

### Correctness assessment
Accurate, professional, no overclaiming; the guard test enforces required commands, the six
families, limitation statements, no positive out-of-scope claims, and no AI-authorship language
(markers stored reversed so the repository stays literal-free).

### Risks / weaknesses
- Minor: a couple of docs/tests use substring scanning that could be brittle to future rewording.

### Production-readiness grade: 92 / 100

### Recommended improvements
- Nice to have: a GitHub Release body; a short screencast/gif; a portfolio case-study version.

### Verdict: Green

## Pre-11B UX hardening patch — Validation clarification messages

### Purpose
Replace the generic clarification line with specific, actionable messages.

### Main files
`src/response_formatter.py` (+ tests, `docs/usage_examples.md`).

### Correctness assessment
Formatter-only; composes a headline from the structured issues (ambiguous → "Do you mean X or Y?";
unknown → "Did you mean X?"; special team; same-team; plus clearer incomplete-parse messages);
deterministic, std-lib only, fail-safe fallback; the CLI still only prints `message`; no
auto-resolution; structured errors/suggestions and JSON unchanged; oracle numbers unchanged.

### Risks / weaknesses
- Inherits the noisy fuzzy-suggestion behaviour from Phase 7 (e.g. "Celics" → "Boston Celtics or
  New Orleans Pelicans").

### Test coverage assessment
Strong (+8 formatter unit tests, +2 real runtime/CLI tests).

### Production-readiness grade: 93 / 100

### Recommended improvements
- Nice to have: per-code templates for invalid-argument cases; a future structured `hint` field.

### Verdict: Green

## Phase 11B and final release closure

### Purpose
Release checklist, final verification document, readiness tests, and the local-then-pushed `v1.0.0`
tag.

### Main files
`RELEASE_CHECKLIST.md`, `docs/final_release_verification.md`, `tests/test_release_readiness.py`.

### Correctness assessment
Checklist frames the tag as a future action; the verification document states all eight architecture
boundaries; 13 readiness tests guard files/commands/tag-future-only/boundaries/limitations/no
positive claims/no AI-authorship/referenced-files. Tag `v1.0.0` annotated and pushed to `origin`.

### Risks / weaknesses
- None functional. (See cross-cutting: history-diff provenance.)

### Test coverage assessment
Strong for a documentation/release surface.

### Production-readiness grade: 92 / 100

### Recommended improvements
- Nice to have: a GitHub Release entry; an auto-generated changelog; signed tags; a CI badge.

### Verdict: Green

---

## Cross-cutting architecture findings

The single-responsibility split is consistently respected and machine-checked by import-scope tests.
No circular dependencies; one-directional imports (e.g. `runtime → assistant`, never reverse). The
"execution only after parse + validation" gate is the central safety invariant and is enforced and
tested. No responsibility leakage found.

## Cross-cutting test findings

673 test functions / 1128 cases, layered (unit → integration → acceptance gates), offline and
deterministic, with explicit import-scope and scope guards. Strengths: oracle-locked numbers,
fail-closed coverage, drift tests. Watch-items: oracle tests are intentionally dataset-coupled (a
dataset swap breaks them — which is desirable, but argues for a dataset-hash guard); a few
documentation/release tests use substring scanning that could be brittle to rewording. No flaky
tests observed.

## Cross-cutting documentation findings

Documentation is accurate to real output and free of overclaiming; limitations are explicit. The
release package and reviewer materials are complete. Minor: a data dictionary and an explicit
parser coverage matrix would help a reviewer.

## Security / safety / AI-risk findings

Strong. No LLM-generated statistics; no hallucinated values; fuzzy matching never auto-resolves or
auto-corrects into execution; ambiguous markets always ask for clarification; unsupported queries
fail safely; no live-data/betting/web-API capabilities exist or are claimed. One low-severity
provenance note: **pre-10C commit diffs still contain an AI-review-tool name and an LLM-provider
constant that were later removed from the working tree** — the current tree and all commit messages
are clean, but the historical diffs are not. Removing them requires a destructive history rewrite.

## Performance and scalability findings

Adequate for the dataset (14,746 rows). The two notable items: (1) the CLI re-bootstraps and
re-validates the dataset on every invocation (~1–2s), with no caching/REPL; (2) tools filter the
full dataframe per call (linear). Both are fine at this scale and would be the first targets if a
larger dataset or a long-running server were introduced.

## Improvement backlog summary

No must-fix defects. Should-fix: dataset content-hash guard; remove the dead config constants; CLI
`--version`; document the validator priority model + parser fallback design. Nice-to-have: fuzzy
suggestion tuning; REPL/caching; packaging entry point; GitHub Release body; data dictionary.
Future roadmap (separately designed, out of v1.0.0 scope): optional LLM parser behind the same
validator; possession-weighted analytics; performance benchmarks. See `docs/improvement_backlog.md`.

## Final recommendation

**v1.0.0 is acceptable as a completed portfolio/review release** — it is correct, deterministic,
safe, well-tested, and well-documented, with no release blockers. No urgent fixes are required.

If pursuing a **v1.1.0**, the highest-value first improvement is a **dataset content-hash guard
plus removing the dead config constants** (small, high-signal correctness/maintainability wins),
followed by **CLI `--version`** and **fuzzy-suggestion tuning**. Larger items (LLM parser, server
mode) should be designed separately and never mixed into the deterministic v1.0.0 scope.
