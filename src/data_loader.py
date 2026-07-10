"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, REQUIRED_FEATURE_COLUMNS, TARGET_COLUMN
from src.logging import get_logger


logger = get_logger(__name__)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the cleaned customer dataset."""

    if not path.exists():
        logger.error("Dataset not found at %s", path)
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    return pd.read_csv(path)


def validate_prediction_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that a dataframe contains all model input columns."""

    missing_columns = [
        column for column in REQUIRED_FEATURE_COLUMNS if column not in df.columns
    ]
    return len(missing_columns) == 0, missing_columns


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe columns in the order expected by the model."""

    valid, missing_columns = validate_prediction_columns(df)
    if not valid:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")
    return df[REQUIRED_FEATURE_COLUMNS].copy()


def get_target_series(df: pd.DataFrame) -> pd.Series:
    """Return binary encoded target values."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
