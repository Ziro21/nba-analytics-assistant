# Reviewer quickstart

A five-minute path through the project. Everything runs offline; no API key or network is needed.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # pandas + pytest only (the full assistant runs on these)
pip install -r requirements-rich.txt   # optional: enables the `--pretty` Rich terminal mode
```

## 2. Run representative CLI commands

```bash
python -m src.cli "How many points do the Warriors average over the last 5 games?"
python -m src.cli "Top 5 scoring teams"
python -m src.cli "Compare Warriors and Celtics over the last 10 games"
python -m src.cli --json "Celtics vs Heat head to head"
python -m src.cli --pretty "Top 5 scoring teams"   # optional Rich table (needs requirements-rich.txt)
python -m src.cli "How many points do LA average?"
```

Expected: the analytics queries return answers (`--json` as structured JSON, `--pretty` as a Rich
table); `LA` is ambiguous and returns a safe clarification rather than guessing. Exit codes: `0` for
an answer, `1` for clarification/unsupported, `2` for an error or invalid arguments.

## 3. Run the tests

```bash
python -m pytest tests/ -q
```

The full suite is offline and deterministic.

## 4. Files to inspect

| File | What to look for |
| --- | --- |
| [src/cli.py](../src/cli.py) | thin CLI: collects a query, calls the runtime, prints, returns an exit code |
| [src/assistant_runtime.py](../src/assistant_runtime.py) | bootstrap: loads/validates the dataset, builds dependencies |
| [src/assistant.py](../src/assistant.py) | orchestrator: parse → validate → execute → format (loads no data) |
| [src/response_formatter.py](../src/response_formatter.py) | pure formatter; fails closed on malformed input |
| [src/intent_validator.py](../src/intent_validator.py) | canonicalisation + safety boundary |
| [src/rule_parser.py](../src/rule_parser.py) | deterministic parser (extracts only) |
| [src/tool_registry.py](../src/tool_registry.py) | the only dispatch path |
| [src/tools.py](../src/tools.py) | the only statistics-calculation layer (pandas) |
| [tests/test_delivery_final.py](../tests/test_delivery_final.py) | final delivery acceptance gate |
| [tests/test_oracle_correctness.py](../tests/test_oracle_correctness.py) | independent oracle: recomputes every tool's answer a second way and checks derivations against the raw CSV |
| [tests/test_adversarial_robustness.py](../tests/test_adversarial_robustness.py) | fail-closed robustness, cross-feature parity, and seeded fuzz |
| [README.md](../README.md) | overview, usage, limitations |
| [docs/architecture.md](architecture.md) | layer responsibilities and safety boundaries |

## 5. What to look for

- A **deterministic pipeline**: the same query always yields the same result.
- **Validation before execution**: a tool runs only after a successful parse and validation.
- **pandas-only calculations**: the language layer never computes a statistic.
- A **structured result contract**: every response is a JSON-safe `AssistantResult`.
- **Safe failures**: ambiguous/unknown teams, vague time ranges, and unsupported queries are
  explained, not guessed.
- A **comprehensive test suite**: layered unit tests, integration tests, and acceptance gates.
