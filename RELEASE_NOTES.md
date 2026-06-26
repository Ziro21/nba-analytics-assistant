# Release Notes

## Current release

A complete, deterministic NBA analytics assistant over a structured CSV dataset, with a
command-line demo and a comprehensive test suite. The system is offline and reproducible: no
network access or API key is required.

### Added

- Deterministic rule parser (query normalisation, intent routing, slot extraction).
- Validation and canonicalisation boundary (team resolution, argument and domain checks).
- pandas-based analytical tools (six registered tools).
- Tool registry with a uniform tool-result contract.
- Assistant result contracts (`AssistantResult`, `AssistantIssue`).
- Deterministic response formatter.
- Production assistant orchestrator (`answer_query`).
- Runtime bootstrap (`AssistantRuntime`, `build_default_runtime`).
- Command-line demo (`python -m src.cli`) with deterministic exit codes.
- Documentation (README, architecture, usage examples, testing/quality) and a final delivery
  acceptance gate.

### Supported query families

- Team average points.
- Average points allowed.
- Team record.
- Top scoring teams.
- Head-to-head record.
- Team efficiency summary.

### Quality and safety

- pandas is the only source of truth; the language layer never computes a statistic.
- The parser extracts only; the validator canonicalises and protects; the registry is the only
  dispatch path; the tools are the only calculation layer; the formatter explains.
- A tool runs only after a successful parse and a successful validation.
- Every response is a structured, JSON-serialisable result; failures are explained, not guessed.
- Import-scope guards keep the assistant and CLI free of pandas, data loaders, and direct tools.
- The full test suite passes, including end-of-phase acceptance gates and a final delivery gate.

### Known limitations

- No live data; the dataset is a fixed bundled CSV.
- No betting odds model and no prediction engine.
- No arbitrary basketball Q&A; only the six supported families are answered.
- No LLM parser is enabled in this build.
- No web/API/RAG/agent layer; the only interface is the command line.
- `season_id` values are opaque internal identifiers.
