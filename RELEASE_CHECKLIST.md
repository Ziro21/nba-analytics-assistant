# Release checklist

This checklist supports the final review and release-tag decision for the deterministic NBA
Analytics Assistant. A release tag should be cut **only after** the final independent review is
Green and the working tree is clean. This document does not create a tag.

## Release candidate status

The repository is **prepared for final independent review**. It is not yet released. The release
tag is a future action to be taken by the project lead after the review is approved.

## Pre-release checks

- [ ] Working tree is clean (`git status --short` shows nothing tracked/uncommitted).
- [ ] Full test suite passes (`python -m pytest tests/ -q`).
- [ ] CLI human-readable smoke passes (returns an answer, exit code 0).
- [ ] CLI JSON smoke passes (valid JSON, status `answer`, exit code 0).
- [ ] Documentation exists (`README.md`, `docs/architecture.md`, `docs/usage_examples.md`,
      `docs/testing_and_quality.md`).
- [ ] Release package exists (`SUBMISSION.md`, `RELEASE_NOTES.md`, `docs/project_summary.md`,
      `docs/reviewer_quickstart.md`).
- [ ] No private `_working/` files are tracked.
- [ ] No `.env` or `.venv` is tracked.
- [ ] No unnecessary dependencies (`requirements.txt` is pandas + pytest only).
- [ ] No unsupported feature claims in the documentation.
- [ ] No AI-authorship/provenance language anywhere in the repository.

## Commands to run

```bash
git status --short
python -m pytest tests/ -q
python -m src.cli "How many points do the Warriors average over the last 5 games?"
python -m src.cli --json "Celtics vs Heat head to head"
```

Expected: the suite passes; the first CLI command prints an answer and exits `0`; the JSON command
prints a valid `answer` result for `head_to_head` and exits `0`.

## Release tag preparation (future action — do not run during Phase 11B)

After the final independent review is Green and the working tree is clean, the project lead may
cut the release tag:

```bash
git tag -a v1.0.0 -m "Release v1.0.0: deterministic NBA analytics assistant"
git push origin v1.0.0
```

**Do not run these commands during Phase 11B.** Run them only after final review approval.

## Final scope confirmation

The release is intentionally bounded:

- No live data — a fixed bundled CSV is the only input.
- No betting odds prediction and no prediction engine.
- No arbitrary basketball Q&A — only the seven supported families are answered.
- No LLM parser is enabled in this build.
- No web/API/RAG/agent system — the only interface is the command line.
