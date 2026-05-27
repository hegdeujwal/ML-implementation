"""
Shared I/O and validation helpers used across all pipeline modules.

Functions
---------
load_parquet(path)         -- load a parquet file with a helpful error on missing
save_parquet(df, path)     -- save a parquet, creating parent dirs if needed
validate_schema(df, cols)  -- raise ValueError if required columns are absent
"""

from pathlib import Path

import pandas as pd


def load_parquet(path: str) -> pd.DataFrame:
    """Load a parquet file, raising a clear error if it does not exist.

    Args:
        path: File path (str or Path).

    Returns:
        DataFrame loaded from the parquet file.

    Raises:
        FileNotFoundError: with a message naming the missing file and suggesting
            which upstream step needs to run.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {path}\n"
            "Ensure the upstream pipeline step has run and written this file."
        )
    return pd.read_parquet(p)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as parquet, creating parent directories if missing.

    Args:
        df:   DataFrame to persist.
        path: Destination file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def validate_schema(df: pd.DataFrame, required_columns: list) -> None:
    """Raise ValueError if any required columns are absent from df.

    Args:
        df:               DataFrame to check.
        required_columns: List of column name strings that must be present.

    Raises:
        ValueError: listing the missing columns and the columns that are present.
    """
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}\n"
            f"Present columns: {list(df.columns)}"
        )
