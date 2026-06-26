"""Project-wide configuration: paths, defaults, and named constants.

Importing this module has no side effects and requires no environment variables, so the
system imports cleanly and runs fully offline. Values here are deliberate design defaults,
not dataset statistics — pandas remains the only source of truth for any computed figure.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
DATASET_PATH: Path = DATA_DIR / "nba_dataset.csv"

# --- Raw-data shape constants (confirmed by read-only pre-flight) -----------
# 125 raw columns including the exported pandas index; 124 analytical after the drop.
INDEX_COLUMN: str = "Unnamed: 0"  # leftover export index; dropped on load.
RAW_DATE_COLUMN: str = "match_date"  # parsed to datetime and exposed as `game_date`.

# All-Star exhibition fingerprint: 8 rows, season_id 36, dated 16 Feb 2026.
# Excluded from franchise/league tools by default; never silently special-cased.
SPECIAL_TEAMS: tuple[str, ...] = ("Team Stars", "Team Stripes", "Team World")

# `season_id` is an OPAQUE integer index — never decoded to a calendar season.
EXPECTED_SEASON_IDS: tuple[int, ...] = (26, 28, 30, 32, 34, 36)

# --- Tool defaults ---------------------------------------------------------
# Default number of teams returned by ``top_scoring_teams`` when N is unspecified.
# There is deliberately NO default window and NO global window bound: a window query must carry an
# explicit number, and vague time such as "recently" is rejected by the parser rather than defaulted.
DEFAULT_TOP_N: int = 5

# --- Dataset integrity fingerprint -----------------------------------------
# SHA-256 of the bundled raw CSV bytes, used to detect a swapped or corrupted dataset at bootstrap.
# Computed from data/nba_dataset.csv; pandas remains the only source of truth for any statistic.
DATASET_HASH_ALGORITHM: str = "sha256"
EXPECTED_DATASET_SHA256: str = "090d9ad663022e2fd94b166f2155d9e2b29ab7ae4b65d57dbcff1719cbfbe69f"
