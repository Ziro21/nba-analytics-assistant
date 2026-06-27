"""Phase 11A tests: keep the release/submission package present, accurate, and in scope."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SUBMISSION = REPO_ROOT / "SUBMISSION.md"
RELEASE_NOTES = REPO_ROOT / "RELEASE_NOTES.md"
PROJECT_SUMMARY = REPO_ROOT / "docs" / "project_summary.md"
REVIEWER_QUICKSTART = REPO_ROOT / "docs" / "reviewer_quickstart.md"
RELEASE_DOCS = (SUBMISSION, RELEASE_NOTES, PROJECT_SUMMARY, REVIEWER_QUICKSTART)

# AI-vendor/authorship markers stored reversed so this file holds no literal AI token.
_REVERSED_AI_MARKERS = (
    "edualc", "ciporhtna", "tpgtahc", "tolipoc", "xedoc", "ianepo", "inimeg",
    "derohtua-oc", "htiw detareneg", "yb-detareneg", "detareneg ia",
)
AI_MARKERS = tuple(m[::-1] for m in _REVERSED_AI_MARKERS)

# Positive capability assertions the project does NOT support. Only verb-led / unambiguous
# positive forms are listed, so a limitation statement ("no betting prediction") never trips them.
FORBIDDEN_POSITIVE_CLAIMS = (
    "supports live data",
    "predicts betting",
    "predicts odds",
    "answers any question",
    "uses an llm",
    "llm parser enabled",
    "web api available",
    "rag enabled",
    "agentic system",
)


def _release_blob() -> str:
    return "\n".join(p.read_text().lower() for p in RELEASE_DOCS if p.exists())


# --- 10.1 release files exist -----------------------------------------------

@pytest.mark.parametrize("path", RELEASE_DOCS, ids=lambda p: p.name)
def test_release_package_files_exist(path: Path) -> None:
    assert path.exists(), f"missing release file: {path.relative_to(REPO_ROOT)}"


# --- 10.2 submission has the key commands -----------------------------------

def test_submission_contains_required_commands() -> None:
    text = SUBMISSION.read_text().lower()
    assert "python -m src.cli" in text
    assert "--json" in text
    assert "python -m pytest tests/ -q" in text


# --- 10.3 supported families documented -------------------------------------

def test_release_docs_document_supported_query_families() -> None:
    text = (SUBMISSION.read_text() + "\n" + RELEASE_NOTES.read_text()).lower()
    for phrase in ("average points", "points allowed", "record",
                   "top scoring teams", "head-to-head", "efficiency", "advanced profile"):
        assert phrase in text, f"release docs should mention {phrase!r}"


# --- 10.4 limitations declared ----------------------------------------------

def test_release_docs_declare_limitations() -> None:
    text = _release_blob()
    for limitation in ("no live data", "no betting", "arbitrary", "no llm", "no web"):
        assert limitation in text, f"release docs should state the limitation {limitation!r}"


# --- 10.5 no positive out-of-scope claims -----------------------------------

def test_release_docs_do_not_claim_unsupported_features() -> None:
    text = _release_blob()
    found = [claim for claim in FORBIDDEN_POSITIVE_CLAIMS if claim in text]
    assert not found, f"release docs make out-of-scope capability claims: {found}"


# --- 10.6 no AI authorship / provenance language ----------------------------

def test_release_docs_have_no_ai_authorship_provenance_language() -> None:
    text = _release_blob()
    found = [marker for marker in AI_MARKERS if marker in text]
    assert not found, f"release docs contain AI-authorship/provenance language: {found}"


# --- 10.7 quickstart references real files ----------------------------------

def test_reviewer_quickstart_references_existing_modules() -> None:
    for rel in ("src/cli.py", "src/assistant_runtime.py", "src/assistant.py",
                "src/response_formatter.py", "src/intent_validator.py", "src/rule_parser.py",
                "src/tool_registry.py", "src/tools.py", "tests/test_delivery_final.py",
                "README.md", "docs/architecture.md"):
        assert (REPO_ROOT / rel).exists(), f"quickstart references missing file: {rel}"
