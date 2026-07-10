"""Train the churn model and persist dashboard artifacts."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    ROC_PATH,
    TEST_SIZE,
)
from src.evaluate import evaluate_model
from src.logging import get_logger
from src.preprocessing import create_preprocessor, load_data, split_features_target


logger = get_logger(__name__)


def build_pipeline(model) -> Pipeline:
    """Build a modeling pipeline with the project preprocessor."""

    df = load_data(DATA_PATH)
    X, _ = split_features_target(df)
    return Pipeline([
        ("preprocessor", create_preprocessor(X)),
        ("classifier", model),
    ])


def save_feature_importance(pipeline: Pipeline) -> None:
    """Persist coefficient-based feature importance for the trained model."""

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "coef_"):
        raw_importance = classifier.coef_[0]
        column_name = "Coefficient"
    elif hasattr(classifier, "feature_importances_"):
        raw_importance = classifier.feature_importances_
        column_name = "Coefficient"
    else:
        raw_importance = np.zeros(len(feature_names))
        column_name = "Coefficient"

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        column_name: raw_importance,
    })
    feature_importance["Importance"] = feature_importance[column_name].abs()
    feature_importance.sort_values("Importance", ascending=False).to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )


def train_and_save() -> dict[str, float]:
    """Train the production model and save all dashboard artifacts."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training dataset")
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    comparison_rows = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, model in candidate_models.items():
        logger.info("Training %s", model_name)
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor(X)),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        comparison_rows.append({"Model": model_name, **metrics})
        trained_pipelines[model_name] = pipeline

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "F1 Score",
        ascending=False,
    )
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    comparison_best_model = str(comparison.iloc[0]["Model"])
    production_model_name = "Logistic Regression"
    pipeline = trained_pipelines[production_model_name]
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(pipeline, X_test, y_test)
    metrics["ROC AUC"] = auc(*roc_curve(y_test, probabilities)[:2])
    metrics["Best Model"] = production_model_name
    metrics["Best Comparison Model"] = comparison_best_model

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)

    pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().to_csv(CLASSIFICATION_REPORT_PATH)

    np.save(CONFUSION_MATRIX_PATH, confusion_matrix(y_test, predictions))

    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    joblib.dump(
        {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc(fpr, tpr)},
        ROC_PATH,
    )
    save_feature_importance(pipeline)
    logger.info(
        "Training completed with production model: %s",
        production_model_name,
    )
    return metrics


if __name__ == "__main__":
    saved_metrics = train_and_save()
    print("Training completed successfully")
    for metric_name, metric_value in saved_metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: {metric_value:.4f}")
        else:
            print(f"{metric_name}: {metric_value}")
