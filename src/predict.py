"""Prediction API for churn scoring."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config import CHURN_LABELS, MODEL_PATH
from src.data_loader import get_feature_frame
from src.logging import get_logger
from src.utils import confidence_from_probability, get_recommendation, get_risk


logger = get_logger(__name__)
_MODEL_CACHE: Any | None = None


def load_model() -> Any:
    """Load and cache the trained churn model."""

    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not MODEL_PATH.exists():
        logger.error("Model file is missing at %s", MODEL_PATH)
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.train` first."
        )

    logger.info("Loading model from %s", MODEL_PATH)
    _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def predict_customer(customer_df: pd.DataFrame) -> dict[str, Any]:
    """Predict churn for one customer and return business-facing outputs."""

    model = load_model()
    features = get_feature_frame(customer_df)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])
    risk = get_risk(probability)

    logger.info(
        "Prediction completed | prediction=%s | probability=%.4f | risk=%s",
        CHURN_LABELS[prediction],
        probability,
        risk,
    )

    return {
        "prediction": CHURN_LABELS[prediction],
        "probability": round(probability * 100, 2),
        "probability_raw": probability,
        "risk": risk,
        "recommendation": get_recommendation(risk),
        "confidence": round(confidence_from_probability(probability) * 100, 2),
    }


def predict_batch(customers_df: pd.DataFrame) -> pd.DataFrame:
    """Score a batch of customers and append churn outputs."""

    model = load_model()
    features = get_feature_frame(customers_df)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    scored = customers_df.copy()
    scored["ChurnPrediction"] = [CHURN_LABELS[int(value)] for value in predictions]
    scored["ChurnProbability"] = (probabilities * 100).round(2)
    scored["Risk"] = [get_risk(float(probability)) for probability in probabilities]
    scored["Recommendation"] = [
        " ".join(get_recommendation(risk)) for risk in scored["Risk"]
    ]
    logger.info("Batch prediction completed for %s rows", len(scored))
    return scored
