"""Model performance dashboard loaded from saved reports."""

from __future__ import annotations

import streamlit as st

from components import configure_page, metric_card, render_footer, render_header
from src.reports import ReportLoadError, load_all_reports
from src.visualization import (
    create_confusion_matrix,
    create_feature_importance_chart,
    create_roc_curve,
)


configure_page("Model Performance")
render_header(
    "Model Performance",
    "Saved model evaluation artifacts for transparent, retraining-free monitoring.",
)


@st.cache_data(show_spinner=False)
def cached_reports():
    """Load model reports from disk."""

    return load_all_reports()


try:
    reports = cached_reports()
except ReportLoadError as exc:
    st.error(f"{exc}. Run `python -m src.train` to regenerate reports.")
    st.stop()

metrics = reports["metrics"]
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Accuracy", f'{metrics["Accuracy"]:.2%}', "Overall correctness")
with col2:
    metric_card("Precision", f'{metrics["Precision"]:.2%}', "Predicted churn quality")
with col3:
    metric_card("Recall", f'{metrics["Recall"]:.2%}', "Churn capture rate")
with col4:
    metric_card("F1 Score", f'{metrics["F1 Score"]:.2%}', "Precision-recall balance")

left, right = st.columns(2)
with left:
    st.plotly_chart(
        create_confusion_matrix(reports["confusion_matrix"]),
        use_container_width=True,
    )
with right:
    st.plotly_chart(create_roc_curve(reports["roc"]), use_container_width=True)

st.plotly_chart(
    create_feature_importance_chart(reports["feature_importance"]),
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Classification Report")
    st.dataframe(reports["classification_report"], use_container_width=True)
with right:
    st.subheader("Model Comparison")
    st.dataframe(reports["model_comparison"], use_container_width=True)

render_footer()
