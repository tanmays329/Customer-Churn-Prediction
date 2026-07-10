"""Preprocessing utilities for model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.config import TARGET_COLUMN


def load_data(path: Path) -> pd.DataFrame:
    """Load a dataset from CSV."""

    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and a binary target."""

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    return X, y


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create the preprocessing transformer."""

    categorical_columns = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        ],
        remainder="passthrough"
    )

    return preprocessor
