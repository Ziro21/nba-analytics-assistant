# Sporting Risk NBA Analytics Assistant

A deterministic, tool-based NBA analytics assistant that answers a controlled set of
natural-language questions over a structured NBA dataset. The assistant parses a query,
validates the team and arguments, executes a registered **pandas**-based analytical tool, and
returns a structured result through a command-line demo.

The language layer never computes a statistic. **pandas and the analytical tools are the only
source of truth** — every number in an answer comes from a deterministic calculation over the
clean dataframe.

## What it is (and is not)

It **is**: a deterministic pipeline — rule parser → validator → tool registry → pandas tools →
formatter → structured result — exposed through a small CLI.

It is **not**: a live-data service, a betting model, a machine-learning predictor, or an
LLM-driven chatbot. It does not answer arbitrary basketball questions — only the eight supported
analytical families below.

## Key features

- **Deterministic rule-based parsing** — the same query always produces the same result; no LLM,
  no network, no randomness.
- **Validation and canonicalisation boundary** — team names are resolved (e.g. `Warriors` →
  `Golden State Warriors`), ambiguous/unknown/special teams are rejected with clear messages.
- **Registered analytical tools** — eight pandas tools dispatched through a single registry.
- **Structured result contract** — every response is an `AssistantResult` (`status`, `message`,
  `data`, `errors`, `warnings`, `meta`), always JSON-serialisable.
- **Deterministic formatter** — turns tool output into a user-facing message; computes nothing.
- **Runtime bootstrap** — one place that loads/validates the dataset and builds dependencies.
- **CLI demo** — a thin command-line interface with deterministic exit codes.
- **Strong regression and safety tests** — the full suite passes (1000+ tests), including
  import-scope guards and final acceptance gates.

## Supported question families

| Family | Tool | Example query |
| --- | --- | --- |
| Team **average points** | `team_average_points` | `How many points do the Warriors average over the last 5 games?` |
| Average **points allowed** | `average_points_allowed` | `How many points do GSW allow over the last 5 games?` |
| Team **record** | `team_record` | `What is the Warriors record?` |
| **Top scoring teams** | `top_scoring_teams` | `Top 5 scoring teams` |
| **Head-to-head** record | `head_to_head` | `Celtics vs Heat head to head` |
| Team **efficiency** summary | `team_efficiency_summary` | `Boston Celtics efficiency last 10 games` |
| Team **advanced profile** | `team_advanced_profile` | `How are the Warriors performing over the last 5 games?` |
| Two-team **comparison** | `compare_team_profiles` | `Compare Warriors and Celtics over the last 10 games` |

Teams can be referenced by nickname (`Warriors`), tri-code (`GSW`), full name
(`Boston Celtics`), or unambiguous city (`Boston`). An optional `last N games` window is
supported where it applies.

**Simple vs broad:** a single-metric question keeps a simple answer (e.g. *"…averaged 114.4 points
…"*). A broad performance question — *"How are the Warriors performing…"*, *"advanced profile"*, or
*"summarise the Warriors over the last 10 games"* — returns a fuller **profile**: record, points
scored and allowed, and pace-adjusted ratings (ORTG/DRTG/net rating).

**Home/away splits:** the five single-team families above (average points, points allowed, record,
efficiency, advanced profile) — and the two-team comparison — also accept an optional **home/away**
filter — *"Warriors home record"*, *"Celtics efficiency away last 10 games"*, *"How many points do the
Lakers allow at home?"*, *"Compare Lakers and Knicks at home"*. `last N` then means the last N
home/away games (applied per team for a comparison). Top scoring teams and head-to-head do not
support location splits; such a query fails safely with a clear message.

**Comparison vs head-to-head:** a **comparison** describes two teams side by side over the same
sample type and is **descriptive, not predictive** — phrase it as *"compare X and Y"*, *"how do X and
Y compare"*, or *"comparison between X and Y"*. A **head-to-head** reports the games the two teams
played against each other — *"X vs Y head to head"*, *"X against Y"*. Bare *"X vs Y"* keeps its
existing head-to-head meaning. Subjective or predictive phrasings (*"who is better"*, *"who will
win"*, betting questions) remain unsupported.

## Limitations (by design)

- **No live data** — the assistant reads a fixed bundled CSV (`data/nba_dataset.csv`), not a feed.
- **No betting model** — it does not estimate or forecast betting markets.
- **No prediction engine** — it reports historical facts from the dataset; it does not project.
- **No LLM parser is enabled by default** — the default is a deterministic rule engine. An optional,
  offline, disabled-by-default **LLM-ready query-interpretation** interface exists as a documented
  extension point (no API key, no network, no SDK); it is gated by the same validator. See
  [docs/llm_integration_design.md](docs/llm_integration_design.md).
- **No arbitrary Q&A** — only the eight families above are supported; anything else fails safely.
- **No web app or service** — the only interface is the command line.
- `season_id` values are treated as **opaque identifiers** and are not decoded into NBA season
  labels (e.g. there is no "2023-24" interpretation).

## Installation

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # pandas + pytest only — the full assistant runs on these alone
```

No API key and no network access are required.

**Optional** — for the Rich pretty terminal mode only (`--pretty`):

```bash
pip install -r requirements-rich.txt  # adds 'rich'; presentation only, not needed for the core
```

## Running the CLI

The supported entry point is `python -m src.cli`. The repository root also provides a thin
convenience shim — `python main.py "question"` runs exactly the same command.

Human-readable answer:

```bash
python -m src.cli "How many points do the Warriors average over the last 5 games?"
# Golden State Warriors averaged 114.4 points over the last 5 games.
```

Structured JSON (the full `AssistantResult.to_dict()`):

```bash
python -m src.cli --json "Celtics vs Heat head to head"
```

**Optional Rich pretty mode** (`--pretty`) renders the same answer in a polished terminal layout —
panels for messages and tables for comparisons/rankings:

```bash
python -m src.cli --pretty "Compare Warriors and Celtics over the last 10 games."   # a comparison table
python -m src.cli --pretty "Top 5 scoring teams."                                   # a ranking table
```

`--pretty` is a **terminal presentation layer only**: it does not change query parsing, validation,
analytics, result calculation, exit codes, or JSON output. It needs the optional `rich` dependency
(install above); without it, `--pretty` prints an install hint and exits — the plain and `--json`
modes always work. `--pretty` and `--json` cannot be combined.

Safe failures (the assistant explains, it never guesses):

```bash
python -m src.cli "How many points do LA average?"   # ambiguous team  -> clarification
python -m src.cli "Who is better?"                    # unsupported     -> safe message
```

**Exit codes:** `0` = answer · `1` = clarification needed / unsupported · `2` = assistant error,
bootstrap failure, or invalid command-line arguments.

See [docs/usage_examples.md](docs/usage_examples.md) for worked examples and JSON shape.

## Architecture overview

```
query → parser → validator → registry → tool → formatter → AssistantResult
```

Through the application surface:

```
CLI → runtime → assistant → (parser → validator → registry → tools → formatter)
```

Responsibility split: the **parser** extracts candidate structure; the **validator**
canonicalises and protects; the **registry** dispatches; the **tools** calculate from the clean
dataframe; the **formatter** explains; the **assistant** coordinates; the **runtime** bootstraps
dependencies; the **CLI** collects the query and displays the result.

Full detail: [docs/architecture.md](docs/architecture.md).

## Running the tests

```bash
python -m pytest tests/ -q
```

Focused examples:

```bash
python -m pytest tests/test_cli.py -q
python -m pytest tests/test_assistant_runtime.py -q
python -m pytest tests/test_assistant_phase9_final.py -q
```

Test strategy and quality gates: [docs/testing_and_quality.md](docs/testing_and_quality.md).

## Project status

The implemented system includes: a deterministic rule parser, a validation/canonicalisation
safety layer, a tool registry, eight pandas analytical tools, structured result contracts, a
deterministic formatter, the assistant orchestrator, a runtime bootstrap, a CLI demo, and a
comprehensive regression/safety test suite. Each layer was implemented and independently
reviewed phase by phase.

## Project layout

```
data/nba_dataset.csv     the structured NBA dataset (only source of truth)
src/                     config, data layer, tools, registry, parser, validator,
                         formatter, assistant, runtime, CLI
tests/                   oracle, edge, integration, and acceptance tests (no network)
docs/                    usage, architecture, and testing documentation
```
