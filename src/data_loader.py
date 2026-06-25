"""Dataset loader: read the raw CSV and remove the exported index column only.

Deliberately minimal. The loader's single responsibility is to return the raw
analytical dataframe. It does NOT:
  - perform feature engineering;
  - derive opponent names;
  - filter exhibition / special-team rows;
  - zero-fill or impute missing values;
  - parse dates (that happens in the date validator);
  - mutate or rewrite the source file.

Column interpretation (confirmed by read-only pre-flight):
  - the raw CSV has 125 columns including a leftover pandas export index, ``Unnamed: 0``;
  - dropping that single index column yields the 124-column analytical dataframe;
  - ``_id`` is a separate vendor/record identifier and is kept.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH, INDEX_COLUMN


def load_raw_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Read the dataset CSV and drop the ``Unnamed: 0`` export index if present.

    Args:
        path: Path to the dataset CSV. Defaults to the project dataset.

    Returns:
        The raw analytical dataframe (124 columns) with ``_id`` retained and the
        ``Unnamed: 0`` index column removed. No other transformation is applied.
    """
    df = pd.read_csv(path)
    if INDEX_COLUMN in df.columns:
        df = df.drop(columns=[INDEX_COLUMN])
    return df
