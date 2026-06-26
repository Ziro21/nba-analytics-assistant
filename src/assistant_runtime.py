"""Application runtime / bootstrap (Phase 10A).

The bootstrap layer that prepares the dependencies the pure assistant orchestrator needs:

    raw dataset -> validated raw -> clean view -> validated clean view -> validation context

It is the ONLY assistant-layer module allowed to load data. ``src/assistant.py`` stays pure
orchestration (no pandas, no data loading). The runtime computes no statistics itself — it builds
dependencies and delegates every query to ``answer_query``.

Error policy: bootstrap is a configuration step, so setup failures (missing/invalid dataset)
raise normally — no user query is being answered yet. Per-query fail-closed handling remains
``answer_query``'s responsibility; ``AssistantRuntime.answer`` returns whatever it produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from src.assistant import answer_query
from src.assistant_types import AssistantResult
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import validate_dataset
from src.tool_registry import DEFAULT_REGISTRY
from src.validation_context import build_validation_context


@dataclass(frozen=True)
class AssistantRuntime:
    """Prepared, reusable assistant dependencies. Built once, used for many queries."""

    clean_df: object
    validation_context: object
    registry: object

    def answer(self, query: str) -> AssistantResult:
        """Answer one query by delegating to the pure orchestrator. Computes nothing itself."""
        return answer_query(
            query,
            clean_df=self.clean_df,
            validation_context=self.validation_context,
            registry=self.registry,
        )


def build_default_runtime(dataset_path: Optional[Union[str, Path]] = None) -> AssistantRuntime:
    """Build the assistant runtime from the on-disk dataset using the existing project pipeline.

    Steps: load raw -> validate raw -> build clean view -> validate clean view -> build the
    validation context against ``DEFAULT_REGISTRY``. Any setup failure raises (configuration
    error); a partially constructed runtime is never returned.
    """
    raw = load_raw_dataset() if dataset_path is None else load_raw_dataset(dataset_path)
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    context = build_validation_context(clean, registry=DEFAULT_REGISTRY)
    return AssistantRuntime(
        clean_df=clean,
        validation_context=context,
        registry=DEFAULT_REGISTRY,
    )
