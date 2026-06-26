# Improvement Backlog

Post-v1.0.0 improvement backlog from the phase-by-phase technical audit
(`docs/phase_by_phase_audit.md`). The shipped v1.0.0 is complete and has no release blockers; these
are prioritised improvements for a future minor release. Each item is specific and actionable.

> **Status update:** the four highest-priority should-fix items (S1–S4) were implemented in
> **v1.1.0-A** (Quality Hardening). They are marked **Done** in place below and summarised in the
> next section; their full detail is retained for traceability.

## Completed in v1.1.0-A

- **S1 — Dataset content-hash guard.** SHA-256 over the raw CSV bytes, centralised in `config.py`
  (`EXPECTED_DATASET_SHA256`), validated at runtime bootstrap — warning by default, with
  `build_default_runtime(strict_dataset_hash=True)` to fail fast.
- **S2 — Config cleanup.** Removed the dead `DEFAULT_WINDOW`/`MIN_WINDOW`/`MAX_WINDOW`; `DEFAULT_TOP_N`
  retained and wired into both `top_scoring_teams` and its registry schema (one source of truth).
- **S3 — Architecture explainability.** Validator priority model and parser fallback (safe-by-validator)
  documented in `docs/architecture.md`.
- **S4 — CLI `--version`.** Prints `sporting-risk-nba-assistant 1.1.0-dev`, exit 0, no dataset load.

## Must fix

**None.** No defects or release blockers were found in the audit. v1.0.0 is acceptable as released.

## Should fix

_All four items below were implemented in v1.1.0-A (see "Completed in v1.1.0-A" above); the detail is
kept for traceability._

### S1 — Dataset integrity / content-hash guard
- **Status:** ✅ Done in v1.1.0-A.
- **Phase:** 4 (data loading/validation).
- **Severity:** medium.
- **Reason:** the system trusts `data/nba_dataset.csv` by shape (row/column counts) only. A swapped
  or corrupted CSV with the same shape would silently change every answer and the oracle tests, with
  no single "this is the expected dataset" assertion.
- **Approach:** compute a content hash of the raw CSV at load; expose it (e.g. in runtime metadata);
  add a config constant for the expected hash and a clear, non-fatal warning (or opt-in strict
  failure) on mismatch.
- **Risk:** low; additive. Must not change tool outputs.
- **Test coverage:** a unit test asserting the expected hash matches the bundled dataset; a test that
  a mismatch is detected/surfaced.

### S2 — Remove (or wire) the dead configuration constants
- **Status:** ✅ Done in v1.1.0-A (window constants removed; `DEFAULT_TOP_N` wired into the tool and
  the registry schema).
- **Phase:** 4 / `config.py`.
- **Severity:** low–medium (maintainability/correctness clarity).
- **Reason:** `DEFAULT_WINDOW`, `MIN_WINDOW`, `MAX_WINDOW`, and `DEFAULT_TOP_N` are defined but used
  in **zero** non-config source files. `DEFAULT_WINDOW`'s comment ("used when a query says 'recently'")
  contradicts the actual behaviour (vague time is rejected). They are dead/misleading.
- **Approach:** either delete the unused constants, or make the relevant default genuinely sourced
  from config (e.g. have `top_scoring_teams` read `DEFAULT_TOP_N`). Update/remove the stale comment.
- **Risk:** very low; verify no import breaks (confirmed unused today).
- **Test coverage:** existing tool/validator tests must still pass; if wiring `DEFAULT_TOP_N`, a test
  that the tool default equals the config value.

### S3 — Document the validator priority model and the parser fallback design
- **Status:** ✅ Done in v1.1.0-A (`docs/architecture.md`).
- **Phase:** 7 / 8 (documentation only).
- **Severity:** low.
- **Reason:** the validator's multi-error priority and the parser's precision-gated fallback are the
  two most subtle areas; a reviewer benefits from an explicit explanation that the fallback is
  safe-by-validator. Architectural explainability is prioritised above CLI polish, so it precedes S4.
- **Approach:** add a short section to `docs/architecture.md` (or a dedicated doc).
- **Risk:** none (docs).
- **Test coverage:** a documentation-existence/keyword test if desired.

### S4 — CLI `--version`
- **Status:** ✅ Done in v1.1.0-A.
- **Phase:** 10 (CLI).
- **Severity:** low.
- **Reason:** a released tool should report its version; useful for bug reports and reviewers.
- **Approach:** add a `--version` flag printing `v1.0.0` (sourced from a single `__version__`).
- **Risk:** very low; argparse-only, no behaviour change to queries.
- **Test coverage:** `main(["--version"])` exits 0 and prints the version.

## Nice to have

### N1 — Tune fuzzy unknown-team suggestions
- **Phase:** 7. **Severity:** low. **Reason:** "Celics" suggests "Boston Celtics **and** New Orleans
  Pelicans"; the second is noise. **Approach:** raise the difflib cutoff or rank/limit suggestions to
  the closest match. **Risk:** must not turn suggestions into auto-resolution (keep suggestion-only).
  **Test coverage:** assert the top suggestion is correct and noise is reduced; assert no
  auto-resolution.

### N2 — Bootstrap-once REPL / cached runtime
- **Phase:** 10. **Severity:** low. **Reason:** the CLI re-loads/re-validates the dataset (~1–2s) per
  invocation. **Approach:** an optional interactive mode that builds the runtime once and answers
  many queries. **Risk:** keep it simple and non-core; no multi-turn state. **Test coverage:** a
  lightweight test that the loop builds the runtime once and answers/fails safely.

### N3 — Packaging entry point
- **Phase:** 10. **Severity:** low. **Reason:** `python -m src.cli` works, but a console script is
  friendlier. **Approach:** a minimal `pyproject.toml` with `[project.scripts]` → `src.cli:main`, no
  new dependencies. **Risk:** low. **Test coverage:** the existing `test_delivery_packaging_entry_
  point_is_consistent_if_present` already asserts any entry point targets `src.cli:main`.

### N4 — Data dictionary + parser coverage matrix (docs)
- **Phase:** 4 / 8. **Severity:** low. **Reason:** aids reviewers. **Approach:** `docs/data_
  dictionary.md` (clean-view columns) and a query-family × surface-form matrix. **Risk:** none.
  **Test coverage:** existence test optional.

### N5 — GitHub Release write-up / screencast
- **Phase:** 11A/B. **Severity:** low. **Reason:** portfolio polish. **Approach:** a Release body from
  the `v1.0.0` tag and a short gif. **Risk:** none. **Test coverage:** none needed.

### N6 — Per-code invalid-argument clarification templates + structured hint
- **Phase:** 9 / UX patch. **Severity:** low. **Reason:** invalid-argument cases still use generic
  text. **Approach:** extend the formatter's per-code templates; optionally add a structured `hint`
  field to `AssistantIssue` in a future version. **Risk:** keep formatter-only and deterministic.
  **Test coverage:** formatter unit tests for the new templates.

### N7 — Git history provenance scrub
- **Phase:** cross-cutting. **Severity:** low. **Reason:** pre-10C commit **diffs** still contain an
  AI-review-tool name and an LLM-provider constant that were later removed from the tree; the current
  tree and all commit messages are clean. **Approach:** a destructive history rewrite (rebase/filter)
  + force-push — only with explicit sign-off, as it rewrites shared history and the tag. **Risk:**
  high (destructive; invalidates existing clones and the tag). **Test coverage:** the repo-wide
  AI-mention guard already protects the working tree.

## Future roadmap

Larger features that must be designed and validated separately and never mixed into the deterministic
v1.0.0 scope:

### F1 — Optional LLM parser strictly behind the same validator
- **Phase:** 8 (additive). **Reason:** broader natural-language coverage. **Approach:** an alternate
  front-end parser whose output passes through the **unchanged** Phase 7 validator (no LLM-computed
  statistics; LLM proposes structure only); fully mocked in tests, flag-gated, off by default.
  **Risk:** must never weaken validation or compute statistics. **Test coverage:** parser-mode
  invariance; mocked, network-free.

### F2 — Possession-weighted / advanced analytics
- **Phase:** 5 (additive tools). **Reason:** richer analysis. **Approach:** new registered tools with
  their own oracles. **Risk:** keep deterministic and oracle-backed. **Test coverage:** new oracle tests.

### F3 — Performance benchmark + larger-dataset support
- **Phase:** 5/10. **Reason:** scale beyond the demo dataset. **Approach:** precomputed team indices,
  a benchmark in CI, optional server mode. **Risk:** must not change numeric outputs. **Test
  coverage:** a benchmark guard and unchanged-oracle regression.

### F4 — Message localisation
- **Phase:** 9. **Reason:** non-English users. **Approach:** a message catalogue keyed by code.
  **Risk:** keep deterministic. **Test coverage:** catalogue completeness test.

## Explicitly out of scope for v1.0.0

The following are intentionally **not** part of v1.0.0 and must be separately designed and validated
before any future inclusion:

- Live data / real-time feeds.
- Betting odds prediction or any betting model.
- Arbitrary basketball question answering.
- An enabled LLM parser (the build is fully deterministic).
- Web app / HTTP API / database / vector search / agent framework.
- Machine-learning prediction of any kind.
