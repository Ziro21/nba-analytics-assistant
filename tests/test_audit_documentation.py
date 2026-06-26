"""Phase 12A tests: keep the post-release audit documents present, complete, and in scope."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT = REPO_ROOT / "docs" / "phase_by_phase_audit.md"
BACKLOG = REPO_ROOT / "docs" / "improvement_backlog.md"
AUDIT_DOCS = (AUDIT, BACKLOG)

# AI-vendor/authorship markers stored reversed so this file holds no literal AI token.
_REVERSED_AI_MARKERS = (
    "edualc", "ciporhtna", "tpgtahc", "tolipoc", "xedoc", "ianepo", "inimeg",
    "derohtua-oc", "htiw detareneg", "yb-detareneg", "detareneg ia",
)
AI_MARKERS = tuple(m[::-1] for m in _REVERSED_AI_MARKERS)

# Only verb-led / unambiguous positive forms, so limitation/out-of-scope statements never trip them.
FORBIDDEN_POSITIVE_CLAIMS = (
    "supports live data", "predicts betting", "predicts odds", "answers any question",
    "uses an llm", "llm parser enabled", "web api available", "rag enabled", "agentic system",
)


def _audit_blob() -> str:
    return "\n".join(p.read_text().lower() for p in AUDIT_DOCS if p.exists())


@pytest.mark.parametrize("path", AUDIT_DOCS, ids=lambda p: p.name)
def test_audit_documentation_files_exist(path: Path) -> None:
    assert path.exists(), f"missing audit doc: {path.relative_to(REPO_ROOT)}"


def test_audit_covers_all_major_phases() -> None:
    text = AUDIT.read_text().lower()
    for phase in ("phase 4", "phase 5", "phase 6", "phase 7", "phase 8", "phase 9",
                  "phase 10", "phase 11a", "pre-11b", "phase 11b"):
        assert phase in text, f"audit should cover {phase!r}"


def test_backlog_has_all_severity_categories() -> None:
    text = BACKLOG.read_text().lower()
    for category in ("must fix", "should fix", "nice to have", "future roadmap"):
        assert category in text, f"backlog should include {category!r}"


def test_backlog_marks_v1_1_0_a_items_completed() -> None:
    # Guard against the backlog drifting back to presenting the now-implemented S1-S4 as open work.
    text = BACKLOG.read_text().lower()
    assert "completed in v1.1.0-a" in text
    assert "done in v1.1.0-a" in text


def test_audit_docs_do_not_claim_unsupported_features() -> None:
    text = _audit_blob()
    found = [claim for claim in FORBIDDEN_POSITIVE_CLAIMS if claim in text]
    assert not found, f"audit docs make out-of-scope capability claims: {found}"


def test_audit_docs_have_no_ai_authorship_provenance_language() -> None:
    text = _audit_blob()
    found = [marker for marker in AI_MARKERS if marker in text]
    assert not found, f"audit docs contain AI-authorship/provenance language: {found}"
