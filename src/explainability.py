"""Model explainability helpers with optional SHAP support."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_loader import get_feature_frame, load_data
from src.logging import get_logger
from src.predict import load_model


logger = get_logger(__name__)


def _transformed_frame(model: Any, customer_df: pd.DataFrame) -> pd.DataFrame:
    """Return the transformed model input as a dataframe."""

    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(get_feature_frame(customer_df))
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return pd.DataFrame(transformed, columns=feature_names)


def explain_prediction(customer_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Return top local feature contributions for a prediction."""

    model = load_model()
    transformed = _transformed_frame(model, customer_df)
    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "coef_"):
        contributions = transformed.iloc[0] * classifier.coef_[0]
        explanation = pd.DataFrame({
            "Feature": transformed.columns,
            "Contribution": contributions,
        })
        explanation["Absolute Contribution"] = explanation["Contribution"].abs()
        return explanation.sort_values(
            "Absolute Contribution",
            ascending=False,
        ).head(top_n)

    return pd.DataFrame(columns=["Feature", "Contribution", "Absolute Contribution"])


def create_shap_waterfall(customer_df: pd.DataFrame):
    """Create a SHAP waterfall figure when SHAP is installed."""

    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        logger.warning("SHAP is not installed: %s", exc)
        return None

    model = load_model()
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    background = load_data().sample(n=100, random_state=42)
    background_features = get_feature_frame(background)
    transformed_background = preprocessor.transform(background_features)
    transformed_customer = preprocessor.transform(get_feature_frame(customer_df))

    if hasattr(transformed_background, "toarray"):
        transformed_background = transformed_background.toarray()
    if hasattr(transformed_customer, "toarray"):
        transformed_customer = transformed_customer.toarray()

    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.LinearExplainer(classifier, transformed_background)
    shap_values = explainer(
        pd.DataFrame(transformed_customer, columns=feature_names)
    )

    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    return plt.gcf()
