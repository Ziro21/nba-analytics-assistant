# Testing and quality

The project is test-first and deterministic. The full suite runs offline (no network, no API
key) and passes end to end.

```bash
python -m pytest tests/ -q
```

## Test strategy

Tests are layered to match the architecture — each layer is verified in isolation, then the
layers are verified together, then the whole pipeline is locked by acceptance gates.

| Layer | Test files (examples) | What it proves |
| --- | --- | --- |
| Data validation | `test_data_validation.py`, `test_data_model.py` | the dataset loads and the clean view is correct and stable |
| Tools | `test_tools.py`, `test_tools_integration.py` | each pandas tool computes the right statistics (oracle-backed) |
| Registry | `test_tool_registry.py`, `test_tool_registry_integration.py` | dispatch, schemas, and the tool result contract |
| Validator | `test_intent_validator.py`, `test_validation_context.py`, `test_team_resolution.py` | canonicalisation and safety decisions |
| Parser | `test_rule_parser*.py`, `test_rule_slot_extractor.py`, `test_rule_intent_router.py`, ... | routing, slot extraction, raw-candidate preservation |
| Parser ↔ validator | `test_rule_parser_validation_integration.py` | parsed intents validate and canonicalise correctly |
| Assistant contracts | `test_assistant_types.py` | `AssistantIssue` / `AssistantResult` invariants and JSON safety |
| Formatter | `test_response_formatter.py` | status mapping and fail-closed handling |
| Assistant orchestrator | `test_assistant.py` | parse → validate → execute → format, fail-closed |
| Assistant integration/safety | `test_assistant_integration.py` | real full-chain answers, registry gating, internal failures |
| Runtime | `test_assistant_runtime.py` | bootstrap pipeline and dependency injection |
| CLI | `test_cli.py` | argument parsing, output, exit codes (with a fake runtime) |
| Final acceptance | `test_rule_parser_phase8_final.py`, `test_assistant_phase9_final.py` | whole-of-phase invariants and scope guards |
| Documentation | `test_documentation.py` | docs exist, list the right commands/families, and make no out-of-scope claims |

## Running tests

Full suite:

```bash
python -m pytest tests/ -q
```

Focused examples:

```bash
python -m pytest tests/test_tools.py -q
python -m pytest tests/test_rule_parser_phase8_final.py -q
python -m pytest tests/test_assistant_phase9_final.py -q
python -m pytest tests/test_assistant_runtime.py -q
python -m pytest tests/test_cli.py -q
```

## Quality gates

The project was built phase by phase. Each phase had an implementation step and an independent
review before being accepted, plus consolidated acceptance gates at the end of major stages:

- **Parser acceptance** (`test_rule_parser_phase8_final.py`) — the deterministic parser stack is
  correct, deterministic, and correctly scoped (no validator/registry/tool/data imports).
- **Assistant acceptance** (`test_assistant_phase9_final.py`) — the full assistant chain answers
  supported queries, fails safely, gates registry execution, and stays in scope.
- **Runtime checks** (`test_assistant_runtime.py`) — bootstrap builds valid dependencies and keeps
  the assistant pure.
- **CLI checks** (`test_cli.py`) — deterministic output and exit codes; lightweight import.

## Safety checks

Several tests exist specifically to keep the safety boundaries honest:

- **Import/scope guards** — importing `src.assistant` pulls in no pandas, data loader, data model,
  data validation, tools, or registry; importing `src.cli` pulls in nothing heavy until it runs.
- **No data loading in the assistant** — `src/assistant.py` contains no dataset loading and no
  pandas import; data loading lives only in the runtime.
- **No direct tool calls** — neither the assistant nor the CLI calls a tool directly; the registry
  is the only dispatch path.
- **No out-of-scope modules** — there is no LLM parser, web, API, database, RAG, or agent module.
- **Malformed input fails closed** — malformed tool results or warning shapes become an
  internal-error result, never a crash or a fabricated answer.
- **Registry execution gating** — a tool runs only after a successful parse and a successful
  validation; parse and validation failures execute nothing.

## Determinism

Every documented answer is reproducible: the same query always yields the same `AssistantResult`.
Statistics come only from the clean dataframe via the pandas tools, so the assistant cannot
produce a number the dataset does not support.
