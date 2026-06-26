# Project summary

## Overview

The Sporting Risk NBA Analytics Assistant is a deterministic, tool-based system that answers a
controlled set of natural-language questions about NBA team performance from a structured CSV
dataset. A natural-language query is parsed, validated, dispatched to a registered pandas tool,
and returned as a structured, user-facing result through a command-line demo.

## Problem statement

A sports-analytics user often wants fast, repeatable answers to common team-level NBA performance
questions from a structured dataset — average points, points allowed, record, top scoring teams,
head-to-head, efficiency. The challenge is to accept natural-language-style queries while
preventing two failure modes: **hallucinated statistics** (a number the data does not support) and
**unsafe free-form answers** (responding confidently to an unsupported or ambiguous request).

## Technical solution

A fixed, single-direction pipeline, with one responsibility per layer:

- **Deterministic rule parser** — normalises the query, routes it to one of six tools, and extracts
  raw candidate slots (team, opponent, window, n, season). It validates nothing and executes nothing.
- **Validator** — canonicalises team names and rejects ambiguous/unknown/special teams; checks
  argument types and domain rules. This is the safety boundary.
- **Tool registry** — the only path that dispatches a validated call to a tool.
- **pandas tools** — the only layer that computes statistics, always from the clean dataframe.
- **Formatter** — turns a tool result (or a parse/validation failure) into a user-facing message.
  It computes nothing and fails closed on malformed input.
- **Assistant result** — one JSON-safe contract (`answer | clarification_needed | unsupported |
  error`) for every outcome.
- **Runtime** — loads and validates the dataset once and builds the dependencies the assistant needs.
- **CLI** — collects a query, calls the runtime, prints the result, and returns a deterministic exit code.

## Why the design is safe

- **No LLM-calculated statistics** — there is no LLM in the build; every figure is a pandas
  calculation over the dataset.
- **The parser does not execute** — it only proposes structure for the validator to judge.
- **The validator protects the tools** — only canonicalised, valid requests reach the registry.
- **The registry controls dispatch** — nothing calls a tool directly.
- **The tools are the only calculation layer** — a single, testable place where numbers come from.
- **Outputs are structured and tested** — every response is a typed result, and the behaviour is
  locked by a large regression suite and acceptance gates.

## Skills demonstrated

- Python and pandas; data validation and a stable clean-data model.
- Software architecture with strict, single-responsibility layer boundaries.
- Deterministic natural-language / rule-based parsing.
- Test-driven development (oracle-backed unit tests, integration tests, acceptance gates).
- CLI design with deterministic exit codes.
- Defensive programming and fail-closed error handling.
- AI-safety-aware system design (no hallucinated statistics; clear scope and failure modes).
- Sports analytics over a structured dataset.

## Interview explanation (≈90–120 seconds)

"I built a deterministic NBA analytics assistant. The idea was to let someone ask natural-language
questions — like 'how many points do the Warriors average over their last five games?' — but to
guarantee the answers come only from the data, never from a model guessing. So instead of an LLM, I
used a rule-based pipeline: a parser extracts the structure of the query, a validator canonicalises
the team name and rejects anything ambiguous or unsupported, a registry dispatches to one of six
pandas tools that do the actual calculation, and a formatter turns the result into a clear sentence.
Every layer has one job and a tested boundary — the parser never executes, the assistant never
loads data or computes statistics, and pandas is the only source of truth. If a query is
unsupported or a team is ambiguous, it fails safely with an explanation instead of a made-up number.
It's all offline and reproducible, exposed through a small command-line demo, and backed by a large
test suite with final acceptance gates. The interesting engineering challenge was the safety
boundary: making natural-language input convenient while making it impossible to hallucinate a
statistic or answer something out of scope."
