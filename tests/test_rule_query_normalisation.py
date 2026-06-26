"""Phase 8B tests: deterministic query normalisation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from src.rule_query_normalisation import normalise_query_text, query_tokens

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_MODULES = (  # 8C/8D legitimately added the catalogue, slot extractor, and parser
    "src.rule_parser_validation_integration",
    "src.llm_query_parser",
    "src.response_formatter",
    "src.assistant",
)


# --- Normalisation basics ---------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("  How many POINTS do GSW allow?  ", "how many points do gsw allow"),
    ("Celtics vs. Heat", "celtics vs heat"),
    ("Celtics head-to-head with Heat", "celtics head to head with heat"),
    ("Boston Celtics win-loss record", "boston celtics win loss record"),
    ("Top 5 scoring teams in season 26", "top 5 scoring teams in season 26"),
    ("Philadelphia 76ers record", "philadelphia 76ers record"),
    ("Warriors   net    rating", "warriors net rating"),
])
def test_normalisation_examples(raw, expected) -> None:
    assert normalise_query_text(raw) == expected


def test_lowercases_and_strips_and_collapses() -> None:
    assert normalise_query_text("   LAKERS   Record   ") == "lakers record"


def test_digits_preserved() -> None:
    assert normalise_query_text("last 5 games top 10 season 26") == "last 5 games top 10 season 26"
    assert "76ers" in normalise_query_text("Philadelphia 76ers")


def test_punctuation_removed_or_spaced() -> None:
    assert normalise_query_text("Who is better?!") == "who is better"
    assert normalise_query_text("points, allowed.") == "points allowed"


# --- Intent phrase preservation ---------------------------------------------

@pytest.mark.parametrize("raw,phrase", [
    ("Warriors points allowed", "points allowed"),
    ("Celtics head-to-head Heat", "head to head"),
    ("Boston win-loss record", "win loss"),
    ("Celtics vs. Heat", "vs"),
    ("Celtics offensive rating", "offensive rating"),
    ("Celtics defensive rating", "defensive rating"),
    ("Warriors net rating", "net rating"),
])
def test_route_relevant_phrases_survive(raw, phrase) -> None:
    normalised = normalise_query_text(raw)
    assert f" {phrase} " in f" {normalised} "


# --- Type / error behaviour -------------------------------------------------

@pytest.mark.parametrize("bad", [None, 5, ["a"], {"q": 1}, b"bytes"])
def test_non_string_raises_typeerror(bad) -> None:
    with pytest.raises(TypeError):
        normalise_query_text(bad)


def test_empty_and_whitespace_normalise_to_empty() -> None:
    assert normalise_query_text("") == ""
    assert normalise_query_text("   ") == ""
    assert normalise_query_text("\t\n ") == ""


def test_query_tokens() -> None:
    assert query_tokens("  Top 5 SCORING teams ") == ("top", "5", "scoring", "teams")
    assert query_tokens("   ") == ()


# --- Import / scope safety --------------------------------------------------

def test_forbidden_modules_absent() -> None:
    for module in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(module) is None, f"{module} should not exist yet"


def test_normalisation_import_is_lightweight() -> None:
    code = (
        "import sys; import src.rule_query_normalisation;"
        "forbidden = ['pandas', 'src.data_loader', 'src.tool_registry', 'src.tools',"
        " 'src.validation_context', 'src.team_resolution', 'src.intent_validator'];"
        "assert not any(m in sys.modules for m in forbidden), [m for m in forbidden if m in sys.modules];"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
