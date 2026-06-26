"""Command-line demo interface (Phase 10B).

A thin, deterministic demo over the Phase 10A runtime. It collects a query, builds the default
runtime, calls ``runtime.answer(query)``, prints the result, and returns an exit code. It contains
NO assistant logic — it never parses, validates, executes, formats, or computes statistics. The
runtime (and therefore data loading) is imported lazily inside ``main`` so importing ``src.cli``
stays lightweight.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Optional

from src import __version__
from src.assistant_types import (
    ASSISTANT_STATUS_ANSWER,
    ASSISTANT_STATUS_CLARIFICATION_NEEDED,
    ASSISTANT_STATUS_ERROR,
    ASSISTANT_STATUS_UNSUPPORTED,
    AssistantResult,
)

PROG = "sporting-risk-nba-assistant"
SETUP_ERROR_MESSAGE = (
    "Could not start the assistant: the dataset could not be loaded or validated."
)
EMPTY_QUERY_MESSAGE = (
    'Please provide a question, e.g. '
    '"How many points do the Warriors average over the last 5 games?"'
)

EXIT_OK = 0          # answer
EXIT_CLARIFY = 1     # clarification_needed / unsupported
EXIT_ERROR = 2       # assistant error / bootstrap failure / CLI argument error

_EXIT_BY_STATUS = {
    ASSISTANT_STATUS_ANSWER: EXIT_OK,
    ASSISTANT_STATUS_CLARIFICATION_NEEDED: EXIT_CLARIFY,
    ASSISTANT_STATUS_UNSUPPORTED: EXIT_CLARIFY,
    ASSISTANT_STATUS_ERROR: EXIT_ERROR,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Ask the deterministic NBA analytics assistant one question.",
    )
    parser.add_argument("query", nargs="*", help="the natural-language question")
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="print the full structured AssistantResult as JSON",
    )
    parser.add_argument(
        "--version", action="version", version=f"{PROG} {__version__}",
        help="print the program version and exit (does not load the dataset)",
    )
    return parser


def _print_result(result: AssistantResult, *, as_json: bool) -> None:
    """Print the assistant result: JSON when requested, else the user-facing message + notes."""
    if as_json:
        print(json.dumps(result.to_dict(), sort_keys=True))
        return
    print(result.message)
    for warning in result.warnings:
        print(f"  note: {warning.message}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a deterministic exit code; never raises for normal paths."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse printed help/usage; surface its code (0 for --help, 2 for errors)
        return int(exc.code) if exc.code is not None else EXIT_OK

    query = " ".join(args.query).strip()
    if not query:
        print(EMPTY_QUERY_MESSAGE, file=sys.stderr)
        return EXIT_ERROR

    try:
        from src.assistant_runtime import build_default_runtime  # lazy: data load happens here, not at import
        runtime = build_default_runtime()
    except Exception:  # noqa: BLE001 - bootstrap is configuration; fail closed, no traceback
        print(SETUP_ERROR_MESSAGE, file=sys.stderr)
        return EXIT_ERROR

    result = runtime.answer(query)
    _print_result(result, as_json=args.as_json)
    return _EXIT_BY_STATUS.get(result.status, EXIT_ERROR)


if __name__ == "__main__":
    sys.exit(main())
