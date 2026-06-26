"""v1.1.0-A tests: dataset content-hash guard (Part A) and config-constant hygiene (Part B).

These additions detect a swapped/corrupted dataset and lock the config cleanup. They change no
analytics: the fingerprint is computed over raw file bytes and pandas remains the only source of
truth for every statistic.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

import src.assistant_runtime as runtime_module
from src import config
from src.config import DATASET_PATH, EXPECTED_DATASET_SHA256
from src.data_validation import (
    DatasetFingerprintResult,
    DatasetIntegrityError,
    compute_file_sha256,
    validate_dataset_fingerprint,
)
from src.tool_registry import DEFAULT_REGISTRY
from src.tools import top_scoring_teams


# --- Part A: dataset fingerprint --------------------------------------------

def test_compute_file_sha256_matches_expected_dataset() -> None:
    assert compute_file_sha256(DATASET_PATH) == EXPECTED_DATASET_SHA256


def test_compute_file_sha256_reads_raw_bytes(tmp_path: Path) -> None:
    payload = b"raw,bytes\n1,2\n"
    target = tmp_path / "sample.csv"
    target.write_bytes(payload)
    assert compute_file_sha256(target) == hashlib.sha256(payload).hexdigest()


def test_dataset_hash_validation_accepts_bundled_dataset() -> None:
    result = validate_dataset_fingerprint(DATASET_PATH, EXPECTED_DATASET_SHA256)
    assert isinstance(result, DatasetFingerprintResult)
    assert result.matches is True
    assert result.algorithm == "sha256"
    assert result.warning is None
    assert result.actual_hash == EXPECTED_DATASET_SHA256


def test_strict_mode_accepts_bundled_dataset() -> None:
    result = validate_dataset_fingerprint(DATASET_PATH, EXPECTED_DATASET_SHA256, strict=True)
    assert result.matches is True  # strict must NOT raise for the real released dataset


def test_dataset_hash_validation_detects_mismatch(tmp_path: Path) -> None:
    swapped = tmp_path / "swapped.csv"
    swapped.write_bytes(b"team,points\nFoo,100\n")  # same-ish shape, different bytes
    result = validate_dataset_fingerprint(swapped, EXPECTED_DATASET_SHA256)  # non-strict default
    assert result.matches is False
    assert result.actual_hash != EXPECTED_DATASET_SHA256
    assert result.warning and "integrity" in result.warning.lower()


def test_dataset_hash_validation_warns_non_strictly_on_mismatch(tmp_path: Path) -> None:
    swapped = tmp_path / "swapped.csv"
    swapped.write_bytes(b"different bytes")
    result = validate_dataset_fingerprint(swapped, EXPECTED_DATASET_SHA256, strict=False)
    assert result.matches is False and result.warning is not None  # warns, does not raise


def test_dataset_hash_validation_can_fail_strictly_on_mismatch(tmp_path: Path) -> None:
    swapped = tmp_path / "swapped.csv"
    swapped.write_bytes(b"different bytes")
    with pytest.raises(DatasetIntegrityError):
        validate_dataset_fingerprint(swapped, EXPECTED_DATASET_SHA256, strict=True)


def test_build_default_runtime_records_matching_fingerprint() -> None:
    runtime = runtime_module.build_default_runtime()
    fingerprint = runtime.dataset_fingerprint
    assert isinstance(fingerprint, DatasetFingerprintResult)
    assert fingerprint.matches is True
    assert fingerprint.algorithm == "sha256"


def test_runtime_strict_dataset_hash_fails_before_loading(tmp_path, monkeypatch) -> None:
    swapped = tmp_path / "swapped.csv"
    swapped.write_bytes(b"not the real dataset")
    loaded: list[int] = []
    monkeypatch.setattr(runtime_module, "load_raw_dataset", lambda *a, **k: loaded.append(1))
    with pytest.raises(DatasetIntegrityError):
        runtime_module.build_default_runtime(dataset_path=swapped, strict_dataset_hash=True)
    assert loaded == []  # strict mismatch fails fast — before any CSV is loaded


# --- Part B: config-constant hygiene ----------------------------------------

@pytest.mark.parametrize("name", ["DEFAULT_WINDOW", "MIN_WINDOW", "MAX_WINDOW"])
def test_dead_window_constants_are_removed(name: str) -> None:
    # These were unused and DEFAULT_WINDOW's comment contradicted the "reject vague time" behaviour.
    assert not hasattr(config, name), f"{name} should have been removed (dead/misleading)"


def test_default_top_n_is_defined_and_wired_into_top_scoring() -> None:
    assert config.DEFAULT_TOP_N == 5
    default_n = inspect.signature(top_scoring_teams).parameters["n"].default
    assert default_n == config.DEFAULT_TOP_N  # the tool default is now sourced from config


def test_tool_registry_top_scoring_default_matches_config() -> None:
    # The public registry schema default for `n` must stay coupled to DEFAULT_TOP_N (single source).
    spec = DEFAULT_REGISTRY.get("top_scoring_teams")
    assert spec is not None
    n_param = next(p for p in spec.parameters if p.name == "n")
    assert n_param.default == config.DEFAULT_TOP_N


def test_dataset_hash_constants_present_and_well_formed() -> None:
    assert config.DATASET_HASH_ALGORITHM == "sha256"
    assert isinstance(config.EXPECTED_DATASET_SHA256, str)
    assert len(config.EXPECTED_DATASET_SHA256) == 64  # SHA-256 hex digest length
