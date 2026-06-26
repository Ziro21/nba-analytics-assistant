# Final release verification

This document summarises what was verified at the end of the project, to support the final
independent review and the release-tag decision. It does not create a release tag.

## Final project state

The project is a complete, deterministic NBA analytics assistant over a structured CSV dataset.
A natural-language query is parsed, validated, dispatched to a registered pandas tool, and
returned as a structured result, available through a command-line demo and a reusable runtime.
It supports six natural-language query families, has a layered single-responsibility
architecture, and ships with a comprehensive offline test suite and a release package.

## Verified delivery surface

- `src/cli.py` — command-line demo.
- `src/assistant_runtime.py` — runtime bootstrap and `AssistantRuntime`.
- `src/assistant.py` — assistant orchestrator (`answer_query`).
- `src/response_formatter.py` — deterministic formatter.
- `src/assistant_types.py` — `AssistantResult` / `AssistantIssue` contracts.
- `README.md`, `SUBMISSION.md`, `RELEASE_NOTES.md`, `docs/reviewer_quickstart.md`.

## Verified commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# full test suite
python -m pytest tests/ -q
# CLI (human-readable)
python -m src.cli "How many points do the Warriors average over the last 5 games?"
# CLI (JSON)
python -m src.cli --json "Celtics vs Heat head to head"
```

Observed: the suite passes; the human-readable command answers and exits `0`; the JSON command
returns a valid `answer` result for `head_to_head` and exits `0`; ambiguous/unsupported queries
return safe, structured clarifications with a non-zero exit code and no traceback.

## Verified safety boundaries

Each layer has one responsibility and a tested boundary:

- The **parser extracts** candidate structure only — it does not validate, canonicalise, or execute.
- The **validator canonicalises** team names and protects the tools (rejects ambiguous/unknown/
  special teams and invalid arguments).
- The **registry dispatches** validated tool calls — it is the only dispatch path.
- The **tools calculate** from the clean dataframe — the only place statistics are produced (pandas).
- The **formatter explains** already-produced results — it computes nothing and fails closed.
- The **assistant coordinates** the pipeline — it loads no data and computes nothing.
- The **runtime bootstraps** the dependencies — the only place the dataset is loaded.
- The **CLI displays** the result — it builds no assistant logic.

## Verified limitations

- No live data — a fixed bundled CSV is the only input.
- No betting odds model and no prediction engine.
- No arbitrary basketball Q&A — only the six supported families are answered.
- No LLM parser is enabled in this build.
- No web/API/RAG/agent system.

## Final release decision note

This document prepares the project for final independent review. A release tag should be created
only after the final review is Green and the working tree is clean. No tag has been created as
part of this verification.
