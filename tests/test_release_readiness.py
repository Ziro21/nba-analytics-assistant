"""Phase 11B tests: final release-readiness at the documentation/repository surface."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

RELEASE_CHECKLIST = REPO_ROOT / "RELEASE_CHECKLIST.md"
FINAL_VERIFICATION = REPO_ROOT / "docs" / "final_release_verification.md"
SUBMISSION = REPO_ROOT / "SUBMISSION.md"
RELEASE_NOTES = REPO_ROOT / "RELEASE_NOTES.md"
PROJECT_SUMMARY = REPO_ROOT / "docs" / "project_summary.md"
REVIEWER_QUICKSTART = REPO_ROOT / "docs" / "reviewer_quickstart.md"

RELEASE_FILES = (
    RELEASE_CHECKLIST, FINAL_VERIFICATION, SUBMISSION, RELEASE_NOTES,
    PROJECT_SUMMARY, REVIEWER_QUICKSTART,
)

# AI-vendor/authorship markers stored reversed so this file holds no literal AI token.
_REVERSED_AI_MARKERS = (
    "edualc", "ciporhtna", "tpgtahc", "tolipoc", "xedoc", "ianepo", "inimeg",
    "derohtua-oc", "htiw detareneg", "yb-detareneg", "detareneg ia",
)
AI_MARKERS = tuple(m[::-1] for m in _REVERSED_AI_MARKERS)

# Only verb-led / unambiguous positive forms, so limitation statements never trip them.
FORBIDDEN_POSITIVE_CLAIMS = (
    "supports live data", "predicts betting", "predicts odds", "answers any question",
    "uses an llm", "llm parser enabled", "web api available", "rag enabled", "agentic system",
)


def _release_blob() -> str:
    return "\n".join(p.read_text().lower() for p in RELEASE_FILES if p.exists())


# --- 8.1 release files exist ------------------------------------------------

@pytest.mark.parametrize("path", RELEASE_FILES, ids=lambda p: p.name)
def test_release_readiness_files_exist(path: Path) -> None:
    assert path.exists(), f"missing release file: {path.relative_to(REPO_ROOT)}"


# --- 8.2 checklist has the required commands --------------------------------

def test_release_checklist_contains_required_commands() -> None:
    text = RELEASE_CHECKLIST.read_text().lower()
    for fragment in ("git status --short", "python -m pytest tests/ -q", "python -m src.cli", "--json"):
        assert fragment in text, f"checklist should contain {fragment!r}"


# --- 8.3 tag documented as a future action only -----------------------------

def test_release_checklist_documents_tag_as_future_action_only() -> None:
    text = RELEASE_CHECKLIST.read_text().lower()
    assert "git tag -a v1.0.0" in text                       # the command is documented
    assert "do not run" in text                              # explicitly future-only
    assert "after final review" in text
    for claim in ("tag has been created", "already tagged", "release tag was created"):
        assert claim not in text                             # never claims the tag exists


# --- 8.4 final verification documents architecture boundaries ---------------

def test_final_release_verification_documents_architecture_boundaries() -> None:
    text = FINAL_VERIFICATION.read_text().lower()
    for phrase in ("parser extracts", "validator canonicalises", "registry dispatches",
                   "tools calculate", "formatter explains", "assistant coordinates",
                   "runtime bootstraps", "cli displays"):
        assert phrase in text, f"final verification should state {phrase!r}"


# --- 8.5 limitations declared -----------------------------------------------

def test_release_readiness_docs_declare_limitations() -> None:
    text = _release_blob()
    for limitation in ("no live data", "no betting", "arbitrary", "no llm", "no web"):
        assert limitation in text, f"release docs should state the limitation {limitation!r}"


# --- 8.6 no positive out-of-scope claims ------------------------------------

def test_release_readiness_docs_do_not_claim_unsupported_features() -> None:
    text = _release_blob()
    found = [claim for claim in FORBIDDEN_POSITIVE_CLAIMS if claim in text]
    assert not found, f"release docs make out-of-scope capability claims: {found}"


# --- 8.7 no AI authorship / provenance language -----------------------------

def test_release_readiness_public_files_have_no_ai_authorship_provenance_language() -> None:
    targets = (
        sorted((REPO_ROOT / "src").glob("*.py"))
        + sorted((REPO_ROOT / "tests").glob("*.py"))
        + sorted((REPO_ROOT / "docs").glob("*.md"))
        + [REPO_ROOT / "README.md", SUBMISSION, RELEASE_NOTES, RELEASE_CHECKLIST,
           REPO_ROOT / "main.py", REPO_ROOT / "requirements.txt"]
    )
    blob = "\n".join(p.read_text() for p in targets if p.exists()).lower()
    found = [marker for marker in AI_MARKERS if marker in blob]
    assert not found, f"public files contain AI-authorship/provenance language: {found}"


# --- 8.8 referenced core files exist ----------------------------------------

def test_release_readiness_referenced_core_files_exist() -> None:
    for rel in ("src/cli.py", "src/assistant_runtime.py", "src/assistant.py",
                "src/response_formatter.py", "src/tool_registry.py", "src/tools.py",
                "tests/test_delivery_final.py", "tests/test_release_package.py"):
        assert (REPO_ROOT / rel).exists(), f"referenced core file missing: {rel}"
