"""Batch churn prediction page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import configure_page, render_footer, render_header
from src.config import REQUIRED_FEATURE_COLUMNS
from src.data_loader import validate_prediction_columns
from src.predict import predict_batch


configure_page("Batch Prediction")
render_header(
    "Batch Prediction",
    "Upload a CSV, validate model columns, score churn risk, and download results.",
)

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV containing the model input columns.")
    with st.expander("Required columns"):
        st.write(", ".join(REQUIRED_FEATURE_COLUMNS))
else:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read CSV file: {exc}")
        st.stop()

    if uploaded_df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    is_valid, missing_columns = validate_prediction_columns(uploaded_df)
    if not is_valid:
        st.error("The uploaded file is missing required model columns.")
        st.write(", ".join(missing_columns))
        st.stop()

    st.success(f"Validated {len(uploaded_df):,} rows.")
    progress = st.progress(0)
    try:
        progress.progress(35)
        scored_df = predict_batch(uploaded_df)
        progress.progress(100)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    st.dataframe(scored_df, use_container_width=True, height=420)
    st.download_button(
        "Download Predictions",
        data=scored_df.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

render_footer()
