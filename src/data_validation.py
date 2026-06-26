"""Dataset validators: prove the structural assumptions of the NBA dataset in code.

Each validator is a small, focused function that either confirms a fact or raises
``DataValidationError`` with a specific message. ``validate_dataset`` runs them in
order and returns a structured summary of the verified facts.

pandas is the only source of truth: every expected number below was computed from
``data/nba_dataset.csv`` before being encoded here — none is guessed. ``season_id``
is treated as an opaque integer index and is never decoded into a season label.

Scope note (Phase 3): this module validates the *raw* analytical frame. It does NOT
derive opponent names, build a clean model, filter exhibition rows, or zero-fill nulls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.config import (
    DATASET_HASH_ALGORITHM,
    EXPECTED_SEASON_IDS,
    INDEX_COLUMN,
    RAW_DATE_COLUMN,
    SPECIAL_TEAMS,
)

# --- Expected dataset facts (computed from data/nba_dataset.csv) ------------
EXPECTED_ROW_COUNT: int = 14_746
EXPECTED_COLUMN_COUNT: int = 124  # after dropping the Unnamed: 0 export index
EXPECTED_UNIQUE_GAMES: int = 7_373
ROWS_PER_GAME: int = 2
EXPECTED_DATE_MIN: str = "2020-12-23"
EXPECTED_DATE_MAX: str = "2026-03-05"
EXPECTED_SPECIAL_TEAM_ROWS: int = 8
MATCH_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
FIXTURE_ID_COLUMN: str = "our_fixture_id"
VALID_HOME_FLAGS: frozenset[int] = frozenset({0, 1})

REQUIRED_COLUMNS: tuple[str, ...] = (
    "_id", "season_id", "game_id", "match_date", "team_id", "team_name",
    "is_home", "team_points", "opponent_points", "total_points", "plus_minus",
    "possessions", "ORTG", "DRTG",
)

# Verified-identical column relationships (safe to treat as one if ever needed).
IDENTITY_CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("team_points", ("fg_points", "ft_points")),
    ("FTA", ("ft_attempts",)),
    ("FTM", ("ft_makes",)),
)

# The three-point TRAP: these pairs are NOT interchangeable. Never collapsed or used
# in the committed tools; we count the disagreement to document the distinction.
THREE_POINT_PAIRS: tuple[tuple[str, str], ...] = (
    ("3FGA", "three_attempts"),
    ("3FGM", "three_makes"),
)

# Core columns that downstream phases depend on and that must be complete (zero nulls).
# (Confirmed complete on the real file; '_id' is validated separately for uniqueness.)
CORE_COMPLETENESS_COLUMNS: tuple[str, ...] = (
    "game_id", "match_date", "season_id", "team_id", "team_name", "is_home",
    "team_points", "opponent_points", "total_points", "plus_minus",
    "possessions", "ORTG", "DRTG",
)

# Minimum RAW columns each committed tool will depend on later. Documentation/verification
# only — the tools are NOT implemented here. ``opponent_team_name`` is derived in Phase 4
# and so is intentionally absent from these raw dependencies.
TOOL_COLUMN_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "team_average_points": ("team_name", "match_date", "team_points", "game_id"),
    "team_record": ("team_name", "match_date", "plus_minus", "game_id"),
    "top_scoring_teams": ("team_name", "team_points", "season_id"),
    "head_to_head": ("game_id", "team_name", "team_points", "opponent_points",
                     "plus_minus", "match_date"),
    "team_efficiency_summary": ("team_name", "match_date", "ORTG", "DRTG",
                                "possessions", "plus_minus"),
    "average_points_allowed": ("team_name", "match_date", "opponent_points", "game_id"),
}


class DataValidationError(Exception):
    """Raised when the dataset violates a proven structural assumption."""


class DatasetIntegrityError(DataValidationError):
    """Raised in strict mode when the dataset file does not match the expected fingerprint."""


# --- Dataset integrity fingerprint ------------------------------------------
# A defence against a silently swapped/corrupted CSV: shape and schema can match while the bytes
# (and therefore every computed statistic) differ. The hash is taken over the RAW FILE BYTES, never
# a dataframe representation, so it is independent of pandas parsing. This changes no analytics.

@dataclass(frozen=True)
class DatasetFingerprintResult:
    """Outcome of comparing a dataset file's content hash against the expected fingerprint."""

    algorithm: str
    actual_hash: str
    expected_hash: str
    matches: bool
    warning: Optional[str] = None


def compute_file_sha256(path: Union[str, Path]) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes (streamed; constant memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_fingerprint(
    path: Union[str, Path], expected_hash: str, *, strict: bool = False
) -> DatasetFingerprintResult:
    """Compare the file's SHA-256 against ``expected_hash``.

    Match → a result with ``matches=True``. Mismatch → ``matches=False`` with a clear ``warning``;
    in ``strict`` mode a mismatch raises :class:`DatasetIntegrityError` instead. This never mutates
    the file and never changes any computed statistic — it only reports on the bytes present.
    """
    actual = compute_file_sha256(path)
    matches = actual == expected_hash
    warning: Optional[str] = None
    if not matches:
        warning = (
            f"Dataset integrity check: {Path(path).name} does not match the expected "
            f"{DATASET_HASH_ALGORITHM} fingerprint (expected {expected_hash[:12]}…, got "
            f"{actual[:12]}…). Results reflect the file actually present, not the released dataset."
        )
        if strict:
            raise DatasetIntegrityError(warning)
    return DatasetFingerprintResult(
        algorithm=DATASET_HASH_ALGORITHM,
        actual_hash=actual,
        expected_hash=expected_hash,
        matches=matches,
        warning=warning,
    )


# --- Individual validators --------------------------------------------------

def validate_required_columns(df: pd.DataFrame) -> int:
    """Confirm all required core columns are present. Returns the count present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    return len(REQUIRED_COLUMNS)


def validate_shape(df: pd.DataFrame) -> dict[str, int]:
    """Confirm row/column counts, index-column removal, and ``_id`` presence."""
    if INDEX_COLUMN in df.columns:
        raise DataValidationError(f"Index column {INDEX_COLUMN!r} was not dropped on load.")
    if "_id" not in df.columns:
        raise DataValidationError("Expected '_id' column is missing.")
    if len(df) != EXPECTED_ROW_COUNT:
        raise DataValidationError(f"Expected {EXPECTED_ROW_COUNT} rows, found {len(df)}.")
    if df.shape[1] != EXPECTED_COLUMN_COUNT:
        raise DataValidationError(
            f"Expected {EXPECTED_COLUMN_COUNT} columns, found {df.shape[1]}."
        )
    unique_games = int(df["game_id"].nunique())
    if unique_games != EXPECTED_UNIQUE_GAMES:
        raise DataValidationError(
            f"Expected {EXPECTED_UNIQUE_GAMES} unique games, found {unique_games}."
        )
    return {"rows": len(df), "columns": df.shape[1], "unique_games": unique_games}


def validate_no_duplicate_ids(df: pd.DataFrame) -> int:
    """Confirm ``_id`` is unique across all rows."""
    dupes = int(df["_id"].duplicated().sum())
    if dupes:
        raise DataValidationError(f"Found {dupes} duplicate '_id' value(s).")
    return dupes


def validate_no_duplicate_rows(df: pd.DataFrame) -> int:
    """Confirm there are no fully duplicated rows."""
    dupes = int(df.duplicated().sum())
    if dupes:
        raise DataValidationError(f"Found {dupes} fully duplicated row(s).")
    return dupes


def validate_game_pair_structure(df: pd.DataFrame) -> dict[str, object]:
    """Confirm the long-format two-rows-per-game structure on the FULL frame.

    Every ``game_id`` must appear exactly twice, ``is_home`` must be 0/1, and each
    game must have exactly one home row and one away row. No exhibition pre-filtering.
    """
    unexpected_flags = set(df["is_home"].unique()) - VALID_HOME_FLAGS
    if unexpected_flags:
        raise DataValidationError(f"Unexpected is_home values: {sorted(unexpected_flags)}")

    counts = df.groupby("game_id").size()
    bad_counts = counts[counts != ROWS_PER_GAME]
    if not bad_counts.empty:
        raise DataValidationError(
            f"{len(bad_counts)} game_id(s) do not appear exactly {ROWS_PER_GAME} times; "
            f"examples: {bad_counts.head(3).to_dict()}"
        )

    home = df.groupby("game_id")["is_home"].agg(["sum", "count"])
    bad_home = home[(home["sum"] != 1) | (home["count"] != ROWS_PER_GAME)]
    if not bad_home.empty:
        raise DataValidationError(
            f"{len(bad_home)} game(s) lack exactly one home and one away row; "
            f"examples: {bad_home.head(3).to_dict('index')}"
        )
    return {
        "all_games_paired": True,
        "home_away_split": {0: int((df["is_home"] == 0).sum()),
                            1: int((df["is_home"] == 1).sum())},
    }


def validate_points_identities(df: pd.DataFrame) -> dict[str, int]:
    """Confirm ``plus_minus == team_points - opponent_points`` and
    ``team_points == total_points`` on every row."""
    pm_viol = int((df["plus_minus"] != df["team_points"] - df["opponent_points"]).sum())
    if pm_viol:
        raise DataValidationError(
            f"plus_minus != team_points - opponent_points on {pm_viol} row(s)."
        )
    tp_viol = int((df["team_points"] != df["total_points"]).sum())
    if tp_viol:
        raise DataValidationError(f"team_points != total_points on {tp_viol} row(s).")
    return {"plus_minus_violations": pm_viol, "team_points_total_violations": tp_viol}


def validate_opponent_points_consistency(df: pd.DataFrame) -> int:
    """Verify (not derive) opponent points: for each game, each row's
    ``opponent_points`` equals the paired row's ``team_points`` (self-merge on game_id)."""
    left = df[["game_id", "team_id", "team_points", "opponent_points"]]
    right = df[["game_id", "team_id", "team_points"]].rename(
        columns={"team_id": "team_id_other", "team_points": "team_points_other"}
    )
    merged = left.merge(right, on="game_id")
    merged = merged[merged["team_id"] != merged["team_id_other"]]
    if len(merged) != len(df):
        raise DataValidationError(
            f"Opponent pairing is not one-to-one: {len(merged)} paired rows vs {len(df)} rows."
        )
    viol = int((merged["opponent_points"] != merged["team_points_other"]).sum())
    if viol:
        raise DataValidationError(
            f"opponent_points != paired team_points on {viol} row(s)."
        )
    return viol


def parse_match_date(df: pd.DataFrame) -> pd.Series:
    """Parse ``match_date`` explicitly with a fixed format. Raises on any failure."""
    try:
        return pd.to_datetime(df[RAW_DATE_COLUMN], format=MATCH_DATE_FORMAT)
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Failed to parse '{RAW_DATE_COLUMN}': {exc}") from exc


def validate_dates(df: pd.DataFrame) -> dict[str, str]:
    """Parse dates and confirm the observed range matches the proven dataset bounds."""
    parsed = parse_match_date(df)
    if parsed.isna().any():
        raise DataValidationError(f"'{RAW_DATE_COLUMN}' contains unparseable values.")
    date_min = parsed.min().date().isoformat()
    date_max = parsed.max().date().isoformat()
    if date_min != EXPECTED_DATE_MIN:
        raise DataValidationError(
            f"Minimum date {date_min} != expected {EXPECTED_DATE_MIN}."
        )
    if date_max != EXPECTED_DATE_MAX:
        raise DataValidationError(
            f"Maximum date {date_max} != expected {EXPECTED_DATE_MAX}."
        )
    return {"date_min": date_min, "date_max": date_max}


def validate_season_ids(df: pd.DataFrame) -> list[int]:
    """Confirm the observed ``season_id`` set matches the expected opaque set exactly.

    Fails if any expected id is missing or any unexpected id appears. ``season_id`` is
    an opaque index; it is never decoded into an NBA season label.
    """
    observed = sorted(int(x) for x in df["season_id"].unique())
    expected = sorted(EXPECTED_SEASON_IDS)
    if observed != expected:
        raise DataValidationError(
            f"season_id set mismatch: observed {observed}, expected {expected}."
        )
    return observed


def validate_special_teams(df: pd.DataFrame) -> dict[str, object]:
    """Validate/report special (exhibition) teams. Does NOT filter them here.

    Enforces the expected special-team row count and that the special teams present
    are exactly the configured set, then — if ``our_fixture_id`` exists — confirms its
    nulls align exactly with the special-team rows. Reports the unique team-name count.
    """
    special_mask = df["team_name"].isin(SPECIAL_TEAMS)
    present_special = sorted(set(df.loc[special_mask, "team_name"]))
    special_rows = int(special_mask.sum())

    if special_rows != EXPECTED_SPECIAL_TEAM_ROWS:
        raise DataValidationError(
            f"Expected {EXPECTED_SPECIAL_TEAM_ROWS} special-team rows, found {special_rows}."
        )
    if set(present_special) != set(SPECIAL_TEAMS):
        raise DataValidationError(
            f"Special teams present {present_special} do not match configured "
            f"{sorted(SPECIAL_TEAMS)}."
        )

    summary: dict[str, object] = {
        "team_name_count": int(df["team_name"].nunique()),
        "special_team_rows": special_rows,
        "special_teams_present": present_special,
        "fixture_null_alignment_ok": None,
    }
    if FIXTURE_ID_COLUMN in df.columns:
        null_mask = df[FIXTURE_ID_COLUMN].isna()
        aligned = bool((null_mask.to_numpy() == special_mask.to_numpy()).all())
        summary["fixture_null_alignment_ok"] = aligned
        if not aligned:
            raise DataValidationError(
                f"'{FIXTURE_ID_COLUMN}' nulls do not align exactly with special-team rows."
            )
    return summary


def validate_known_identities(df: pd.DataFrame) -> dict[str, int]:
    """Confirm verified-identical column relationships hold where the columns exist."""
    results: dict[str, int] = {}
    for lhs, rhs in IDENTITY_CHECKS:
        if lhs not in df.columns or any(c not in df.columns for c in rhs):
            continue
        expected = df[rhs[0]] if len(rhs) == 1 else df[rhs[0]] + df[rhs[1]]
        viol = int((df[lhs] != expected).sum())
        label = f"{lhs}=={'+'.join(rhs)}"
        if viol:
            raise DataValidationError(f"Identity {label} violated on {viol} row(s).")
        results[label] = viol
    return results


def validate_three_point_trap(df: pd.DataFrame) -> dict[str, int]:
    """Document that the three-point column pairs are NOT interchangeable.

    Informational: returns the disagreement count for each pair. These columns are
    never collapsed or used by the committed tools.
    """
    disagreements: dict[str, int] = {}
    for col_a, col_b in THREE_POINT_PAIRS:
        if col_a in df.columns and col_b in df.columns:
            disagreements[f"{col_a} vs {col_b}"] = int((df[col_a] != df[col_b]).sum())
    return disagreements


def validate_core_completeness(df: pd.DataFrame) -> dict[str, int]:
    """Confirm the core columns contain no missing values.

    Returns a {column: null_count} map (all zeros on success). Raises if any core
    column is absent or contains nulls. No imputation or zero-filling is performed.
    """
    absent = [c for c in CORE_COMPLETENESS_COLUMNS if c not in df.columns]
    if absent:
        raise DataValidationError(f"Core columns absent: {absent}")
    null_counts = {c: int(df[c].isna().sum()) for c in CORE_COMPLETENESS_COLUMNS}
    offenders = {c: v for c, v in null_counts.items() if v > 0}
    if offenders:
        raise DataValidationError(f"Core columns contain missing values: {offenders}")
    return null_counts


def profile_missingness(df: pd.DataFrame) -> dict[str, object]:
    """Summarise missing values per column without imputing or dropping anything.

    Separates core columns (expected complete) from advanced/vendor columns where
    nulls are concentrated. The input frame is never modified.
    """
    total = len(df)
    null_counts = df.isna().sum()
    with_nulls = null_counts[null_counts > 0]
    core_set = set(CORE_COMPLETENESS_COLUMNS)

    core_with_nulls = {c: int(with_nulls[c]) for c in with_nulls.index if c in core_set}
    advanced = [
        {"column": c, "count": int(with_nulls[c]), "pct": round(100 * with_nulls[c] / total, 4)}
        for c in with_nulls.index
        if c not in core_set
    ]
    # Stable ordering: most-null first, then alphabetical — never depends on input order.
    advanced.sort(key=lambda row: (-row["count"], row["column"]))

    return {
        "rows": total,
        "columns_with_nulls": int((null_counts > 0).sum()),
        "null_count_min": int(with_nulls.min()) if not with_nulls.empty else 0,
        "null_count_max": int(with_nulls.max()) if not with_nulls.empty else 0,
        "core_columns_with_nulls": core_with_nulls,
        "core_complete": len(core_with_nulls) == 0,
        "advanced_columns_with_nulls": advanced,
    }


def summarise_tool_column_readiness(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """For each committed tool, report whether its minimum RAW columns exist and are
    complete. Documentation/verification only — no tool is implemented here."""
    report: dict[str, dict[str, object]] = {}
    for tool, columns in TOOL_COLUMN_DEPENDENCIES.items():
        present = [c for c in columns if c in df.columns]
        absent = [c for c in columns if c not in df.columns]
        null_cols = [c for c in present if int(df[c].isna().sum()) > 0]
        report[tool] = {
            "columns": list(columns),
            "all_present": not absent,
            "absent": absent,
            "all_complete": not absent and not null_cols,
            "null_columns": null_cols,
        }
    return report


# --- High-level orchestrator -----------------------------------------------

def validate_dataset(df: pd.DataFrame) -> dict[str, object]:
    """Run all validators in order on the raw analytical frame.

    Returns a structured summary of the verified facts. Raises ``DataValidationError``
    with a specific message on the first failed check.
    """
    validate_required_columns(df)
    shape = validate_shape(df)
    validate_core_completeness(df)
    summary: dict[str, object] = {
        **shape,
        "core_complete": True,
        "duplicate_ids": validate_no_duplicate_ids(df),
        "duplicate_rows": validate_no_duplicate_rows(df),
        **validate_game_pair_structure(df),
        **validate_points_identities(df),
        "opponent_points_violations": validate_opponent_points_consistency(df),
        **validate_dates(df),
        "season_ids": validate_season_ids(df),
        **validate_special_teams(df),
        "known_identities": validate_known_identities(df),
        "three_point_disagreements": validate_three_point_trap(df),
    }
    return summary


def build_dataset_profile(df: pd.DataFrame) -> dict[str, object]:
    """Combine the validated structural facts with the missingness and tool-readiness
    summaries into one profile dict (for human confirmation against §2)."""
    return {
        "facts": validate_dataset(df),
        "missingness": profile_missingness(df),
        "tool_readiness": summarise_tool_column_readiness(df),
    }


def format_profile(profile: dict[str, object]) -> str:
    """Render a dataset profile as a readable multi-line report."""
    facts = profile["facts"]  # type: ignore[index]
    miss = profile["missingness"]  # type: ignore[index]
    readiness = profile["tool_readiness"]  # type: ignore[index]
    lines = [
        "=== NBA dataset profile (raw analytical frame) ===",
        f"rows: {facts['rows']}  columns: {facts['columns']}  unique games: {facts['unique_games']}",
        f"home/away split: {facts['home_away_split']}",
        f"duplicate ids: {facts['duplicate_ids']}  duplicate rows: {facts['duplicate_rows']}",
        f"plus_minus violations: {facts['plus_minus_violations']}  "
        f"team_points==total_points violations: {facts['team_points_total_violations']}  "
        f"opponent_points violations: {facts['opponent_points_violations']}",
        f"date range: {facts['date_min']} .. {facts['date_max']}",
        f"season_ids (opaque): {facts['season_ids']}",
        f"team names: {facts['team_name_count']}  special-team rows: {facts['special_team_rows']} "
        f"{facts['special_teams_present']}  fixture-null alignment: {facts['fixture_null_alignment_ok']}",
        f"three-point disagreements: {facts['three_point_disagreements']}",
        "",
        "--- missingness ---",
        f"core columns complete (zero nulls): {miss['core_complete']}",
        f"columns with nulls: {miss['columns_with_nulls']}  "
        f"null-count range: [{miss['null_count_min']}, {miss['null_count_max']}]  "
        "(all confined to advanced/vendor columns; NOT zero-filled)",
        "top nullable advanced columns:",
    ]
    for row in miss["advanced_columns_with_nulls"][:5]:  # type: ignore[index]
        lines.append(f"  {row['column']:42s} {row['count']:4d}  {row['pct']}%")
    lines.append("")
    lines.append("--- committed-tool column readiness ---")
    for tool, info in readiness.items():  # type: ignore[union-attr]
        lines.append(
            f"  {tool:24s} present={info['all_present']} complete={info['all_complete']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - manual profiling entry point
    from src.data_loader import load_raw_dataset

    print(format_profile(build_dataset_profile(load_raw_dataset())))
