# Sporting Risk NBA Analytics Assistant

A deterministic, tool-based assistant that answers natural-language questions about a
structured NBA game dataset. A question is mapped to one registered analytical tool,
the request is validated, and a deterministic **pandas** calculation produces the answer.
The language layer never computes a statistic — pandas is the only source of truth.

> **Status: work in progress.** The repository skeleton is in place (Phase 2). The full
> README — quick-start, the six tools, example queries, and the deterministic-calculation
> statement — is completed in Phase 14. The sections below are the minimum needed to run
> what exists today.

## Quick start (skeleton)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py "What is the average points scored by the Warriors in their last 5 games?"
```
At this phase `main.py` is a scaffold that confirms the CLI wiring; the analytical
pipeline is implemented in later phases.

## Design in one line
Two interchangeable front ends — a default **rule-based** parser (no API key) and an
optional **LLM** tool-calling parser (`--mode llm`) — feed **one** shared pipeline:
`ParsedIntent → validator → tool registry → pandas tool → formatter`.

## Project layout (target)
```
main.py            thin CLI
src/               config, data layer, tools, registry, parsers, validator, formatter
tests/             oracle + edge + end-to-end tests (no network)
examples/          one-command demo
data/              nba_dataset.csv
```

## Requirements
- Python 3.12+
- `pandas`, `pytest` (core). `openai`, `python-dotenv` are optional, only for `--mode llm`.
