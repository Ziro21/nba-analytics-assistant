"""Thin CLI entry point: `python main.py "question" [--mode rule|llm]`.

SCAFFOLDING ONLY at this phase. The full pipeline (parser -> validator -> registry
-> tool -> formatter) is wired in later phases; the orchestrator lands in Phase 10.
This stub exists so the skeleton is runnable and the import path is verified early.
"""

from __future__ import annotations

import argparse

from src.config import DEFAULT_MODE, SUPPORTED_MODES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sporting Risk NBA Analytics Assistant (deterministic, tool-based).",
    )
    parser.add_argument("question", nargs="?", help="Natural-language question about the dataset.")
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=DEFAULT_MODE,
        help="Parser front end. 'rule' (default, no API key) or 'llm' (optional).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(
        "Skeleton in place — the assistant pipeline is not implemented yet "
        f"(arrives in Phase 10). Received mode='{args.mode}', "
        f"question={args.question!r}."
    )


if __name__ == "__main__":
    main()
