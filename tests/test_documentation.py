"""Phase 10C tests: keep the committed documentation honest and in scope."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
DOC_FILES = (
    README,
    REPO_ROOT / "docs" / "usage_examples.md",
    REPO_ROOT / "docs" / "architecture.md",
    REPO_ROOT / "docs" / "testing_and_quality.md",
)


def _text(path: Path) -> str:
    return path.read_text().lower()


def _all_docs_text() -> str:
    return "\n".join(_text(p) for p in DOC_FILES if p.exists())


# --- 10.1 required files exist ----------------------------------------------

@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_required_documentation_files_exist(path: Path) -> None:
    assert path.exists(), f"missing documentation file: {path.relative_to(REPO_ROOT)}"


# --- 10.2 README has the key commands ---------------------------------------

def test_readme_contains_cli_and_test_commands() -> None:
    text = _text(README)
    assert "python -m src.cli" in text
    assert "--json" in text
    assert "python -m pytest tests/ -q" in text


# --- 10.3 README documents the supported families ---------------------------

def test_readme_documents_supported_question_families() -> None:
    text = _text(README)
    for phrase in ("average points", "points allowed", "record",
                   "top scoring teams", "head-to-head", "efficiency", "advanced profile"):
        assert phrase in text, f"README should mention {phrase!r}"


# --- 10.4 no out-of-scope CAPABILITY claims (limitations are fine) -----------

FORBIDDEN_POSITIVE_CLAIMS = (
    "supports live data", "live data feed", "real-time data", "live nba data",
    "predicts betting", "betting prediction", "predict betting outcomes", "betting odds prediction",
    "answers any question", "answer any question", "any basketball question",
    "powered by an llm", "llm reasoning", "llm-powered",
    "machine learning model", "neural network", "trained model",
    "vector search", "vector store", "rag pipeline", "agent framework",
    "rest api", "web api available", "graphql",
)


def test_documentation_does_not_claim_out_of_scope_features() -> None:
    text = _all_docs_text()
    found = [claim for claim in FORBIDDEN_POSITIVE_CLAIMS if claim in text]
    assert not found, f"docs make out-of-scope capability claims: {found}"


# --- 10.5 no AI authorship / provenance language ----------------------------

# AI-system / authorship markers are stored REVERSED so this repository itself contains no
# literal AI-vendor or authorship token anywhere (the project requires the repo to stay free of
# any AI-system or authorship reference). The test reverses them back to scan the source.
_REVERSED_MARKERS = (
    "edualc", "ciporhtna", "tpgtahc", "tolipoc", "xedoc", "ianepo", "inimeg",
    "derohtua-oc", "htiw detareneg", "yb-detareneg", "detareneg ia",
)
AUTHORSHIP_TERMS = tuple(marker[::-1] for marker in _REVERSED_MARKERS)


def test_documentation_has_no_ai_authorship_provenance_language() -> None:
    text = _all_docs_text()
    found = [term for term in AUTHORSHIP_TERMS if term in text]
    assert not found, f"docs contain AI-authorship/provenance language: {found}"


def test_repository_has_no_ai_system_or_authorship_mentions() -> None:
    # Whole-repo guard: no AI-vendor / AI-reviewer / authorship token anywhere in source or docs.
    targets = (
        sorted((REPO_ROOT / "src").glob("*.py"))
        + sorted((REPO_ROOT / "tests").glob("*.py"))
        + sorted((REPO_ROOT / "docs").glob("*.md"))
        + [README, REPO_ROOT / "main.py", REPO_ROOT / "requirements.txt"]
    )
    blob = "\n".join(p.read_text() for p in targets if p.exists()).lower()
    found = [term for term in AUTHORSHIP_TERMS if term in blob]
    assert not found, f"repository contains an AI-system/authorship mention: {found}"


# --- legacy / stale delivery artifacts --------------------------------------

def test_no_stale_legacy_entrypoint_claims() -> None:
    # The root entry point and requirements must not advertise the obsolete scaffold / LLM mode.
    stale = ("--mode llm", "not implemented", "scaffold")
    for name in ("main.py", "requirements.txt"):
        text = (REPO_ROOT / name).read_text().lower()
        found = [marker for marker in stale if marker in text]
        assert not found, f"{name} contains stale references: {found}"


# --- 10.6 documented entry points are importable ----------------------------

def test_documented_entry_points_are_importable() -> None:
    assert importlib.util.find_spec("src.cli") is not None
    assert importlib.util.find_spec("src.assistant_runtime") is not None


# --- v1.1.0-A: architecture explainability (validator priority + parser fallback) ----

def test_architecture_documents_validator_priority() -> None:
    text = _text(REPO_ROOT / "docs" / "architecture.md")
    assert "validator priority" in text
    for code in ("ambiguous_team", "unknown_team", "same_team_head_to_head"):
        assert code in text, f"architecture should document {code!r} in the priority model"
    assert "never auto-resolved" in text  # ambiguous/unknown teams are never auto-resolved


def test_architecture_documents_parser_fallback_safe_by_validator() -> None:
    text = _text(REPO_ROOT / "docs" / "architecture.md")
    assert "parser fallback" in text
    assert "safe-by-validator" in text
    assert "unknown_team" in text  # a fallback false positive becomes unknown_team, not an answer
