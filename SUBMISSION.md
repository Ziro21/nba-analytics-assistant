# Sporting Risk NBA Analytics Assistant — Submission

## Summary

A deterministic, tool-based NBA analytics assistant that answers a controlled set of
natural-language questions over a structured NBA CSV dataset. A query flows through a fixed
pipeline — **rule parser → validator → tool registry → pandas tools → formatter → assistant
result** — and is exposed through a command-line demo. The language layer never computes a
statistic: pandas and the analytical tools are the only source of truth. There is no live data,
no betting prediction, no arbitrary question answering, no enabled LLM parser, and no web/API.

## What this project demonstrates

- **Deterministic natural-language routing** — a rule parser maps a query to one of eight tools; the
  same input always produces the same result (no LLM, no network, no randomness).
- **Safe validation and canonicalisation** — team names are resolved (`Warriors` →
  `Golden State Warriors`); ambiguous, unknown, and special teams are rejected with clear messages.
- **pandas analytics over a clean dataframe** — every number is a deterministic pandas calculation.
- **Tool registry and structured tool results** — eight registered tools dispatched through one
  registry, each returning a uniform result contract.
- **Assistant result contract** — a single JSON-safe `AssistantResult` for every outcome.
- **CLI demo** — a thin, scriptable interface with deterministic exit codes; an optional
  `--pretty` Rich terminal mode (presentation only, optional dependency) for nicer output.
- **Layered tests and final acceptance gates** — a large regression suite plus end-of-phase
  acceptance gates and a final delivery gate.
- **Strong scope and safety boundaries** — enforced and machine-checked (import-scope guards,
  no data loading in the assistant, registry-only dispatch).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # pandas + pytest only; no API key, no network

python -m src.cli "How many points do the Warriors average over the last 5 games?"
python -m src.cli --json "Celtics vs Heat head to head"
python -m pytest tests/ -q
```

## Supported query families

| Family | Example |
| --- | --- |
| Team average points | `How many points do the Warriors average over the last 5 games?` |
| Average points allowed | `How many points do GSW allow over the last 5 games?` |
| Team record | `What is the Warriors record?` |
| Top scoring teams | `Top 5 scoring teams` |
| Head-to-head record | `Celtics vs Heat head to head` |
| Team efficiency summary | `Boston Celtics efficiency last 10 games` |
| Team advanced profile | `How are the Warriors performing over the last 5 games?` |
| Two-team comparison | `Compare Warriors and Celtics over the last 10 games` |

## Key limitations

- **No live data** — a fixed bundled CSV is the only input.
- **No betting odds model** and **no prediction engine** — historical facts only.
- **No arbitrary basketball Q&A** — only the eight families above; anything else fails safely.
- **No LLM parser** is enabled in this build — the parser is a deterministic rule engine.
- **No web/API/RAG/agent layer** — the only interface is the command line.
- `season_id` values are opaque internal identifiers, not decoded calendar seasons.

## Review checklist

1. Run a CLI query: `python -m src.cli "What is the Warriors record?"`
2. Run a JSON query: `python -m src.cli --json "Celtics vs Heat head to head"`
3. Run the full test suite: `python -m pytest tests/ -q`
4. Read the architecture: [docs/architecture.md](docs/architecture.md)
5. Read the safety boundaries: [docs/architecture.md](docs/architecture.md) and
   [docs/testing_and_quality.md](docs/testing_and_quality.md)
6. Inspect the final delivery gate: [tests/test_delivery_final.py](tests/test_delivery_final.py)
