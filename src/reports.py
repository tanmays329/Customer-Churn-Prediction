"""Helpers for loading persisted training artifacts."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    ROC_PATH,
)
from src.logging import get_logger


logger = get_logger(__name__)


class ReportLoadError(RuntimeError):
    """Raised when a required report artifact cannot be loaded."""


def _ensure_non_empty(path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ReportLoadError(f"Missing or empty report artifact: {path.name}")


def load_metrics() -> dict[str, float]:
    """Load model KPI metrics from disk."""

    _ensure_non_empty(METRICS_PATH)
    return joblib.load(METRICS_PATH)


def load_classification_report() -> pd.DataFrame:
    """Load the saved classification report."""

    _ensure_non_empty(CLASSIFICATION_REPORT_PATH)
    return pd.read_csv(CLASSIFICATION_REPORT_PATH, index_col=0)


def load_feature_importance() -> pd.DataFrame:
    """Load feature importance values."""

    _ensure_non_empty(FEATURE_IMPORTANCE_PATH)
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def load_confusion_matrix() -> np.ndarray:
    """Load the saved confusion matrix."""

    _ensure_non_empty(CONFUSION_MATRIX_PATH)
    return np.load(CONFUSION_MATRIX_PATH)


def load_roc() -> dict[str, Any]:
    """Load ROC curve arrays and AUC."""

    _ensure_non_empty(ROC_PATH)
    return joblib.load(ROC_PATH)


def load_model_comparison() -> pd.DataFrame:
    """Load model comparison results."""

    _ensure_non_empty(MODEL_COMPARISON_PATH)
    return pd.read_csv(MODEL_COMPARISON_PATH)


def load_all_reports() -> dict[str, Any]:
    """Load all dashboard report artifacts."""

    logger.info("Loading report artifacts")
    return {
        "metrics": load_metrics(),
        "classification_report": load_classification_report(),
        "feature_importance": load_feature_importance(),
        "confusion_matrix": load_confusion_matrix(),
        "roc": load_roc(),
        "model_comparison": load_model_comparison(),
    }
