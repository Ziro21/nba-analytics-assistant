"""Application runtime / bootstrap (Phase 10A).

The bootstrap layer that prepares the dependencies the pure assistant orchestrator needs:

    dataset fingerprint -> raw dataset -> validated raw -> clean view -> validated clean view
    -> validation context

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
from src.config import DATASET_PATH, EXPECTED_DATASET_SHA256
from src.data_loader import load_raw_dataset
from src.data_model import build_clean_view, validate_clean_view
from src.data_validation import (
    DatasetFingerprintResult,
    validate_dataset,
    validate_dataset_fingerprint,
)
from src.tool_registry import DEFAULT_REGISTRY
from src.validation_context import build_validation_context


@dataclass(frozen=True)
class AssistantRuntime:
    """Prepared, reusable assistant dependencies. Built once, used for many queries."""

    clean_df: object
    validation_context: object
    registry: object
    dataset_fingerprint: Optional[DatasetFingerprintResult] = None  # bootstrap metadata only.
    parser: object = None  # optional injected query interpreter; None -> the default rule parser.

    def answer(self, query: str) -> AssistantResult:
        """Answer one query by delegating to the pure orchestrator. Computes nothing itself.

        If a ``parser`` was injected (e.g. an LLM-ready interpreter) it is used in place of the
        default rule parser; the validator and registry remain the only safety/execution gates.
        """
        kwargs = dict(
            clean_df=self.clean_df,
            validation_context=self.validation_context,
            registry=self.registry,
        )
        if self.parser is not None:
            kwargs["parser"] = self.parser
        return answer_query(query, **kwargs)


def build_default_runtime(
    dataset_path: Optional[Union[str, Path]] = None, *, strict_dataset_hash: bool = False,
    parser: object = None,
) -> AssistantRuntime:
    """Build the assistant runtime from the on-disk dataset using the existing project pipeline.

    Steps: fingerprint dataset -> load raw -> validate raw -> build clean view -> validate clean
    view -> build the validation context against ``DEFAULT_REGISTRY``. The fingerprint compares the
    file's SHA-256 to ``EXPECTED_DATASET_SHA256``; by default a mismatch is recorded as metadata
    (warning) and bootstrap proceeds, so a reviewer can run a different dataset. With
    ``strict_dataset_hash=True`` a mismatch raises before any work. The check changes no statistic.

    ``parser`` is an optional injected query interpreter (default ``None`` -> the deterministic rule
    parser). This function never imports or configures any LLM provider itself; an LLM-ready parser
    is supplied by the caller. Any setup failure raises (configuration error); a partially
    constructed runtime is never returned.
    """
    path = DATASET_PATH if dataset_path is None else Path(dataset_path)
    fingerprint = validate_dataset_fingerprint(
        path, EXPECTED_DATASET_SHA256, strict=strict_dataset_hash
    )
    raw = load_raw_dataset(path)
    validate_dataset(raw)
    clean = build_clean_view(raw)
    validate_clean_view(clean, raw)
    context = build_validation_context(clean, registry=DEFAULT_REGISTRY)
    return AssistantRuntime(
        clean_df=clean,
        dataset_fingerprint=fingerprint,
        validation_context=context,
        registry=DEFAULT_REGISTRY,
        parser=parser,
    )
