"""Streamlit home dashboard."""

from __future__ import annotations

import streamlit as st

# pyrefly: ignore [missing-import]
from components import configure_page, metric_card, render_footer, render_header
from src.data_loader import load_data
from src.reports import ReportLoadError, load_metrics
from src.visualization import (
    create_churn_by_contract,
    create_churn_pie,
    create_monthly_charge_hist,
)


configure_page("Executive Overview")


@st.cache_data(show_spinner=False)
def cached_data():
    """Load and cache the cleaned dataset."""

    return load_data()


@st.cache_data(show_spinner=False)
def cached_metrics():
    """Load cached model metrics when available."""

    try:
        return load_metrics()
    except ReportLoadError:
        return {}


df = cached_data()
metrics = cached_metrics()

render_header(
    "Customer Churn Analytics",
    "A production-style retention dashboard for scoring churn risk and identifying revenue protection opportunities.",
)

total_customers = len(df)
churn_rate = df["Churn"].eq("Yes").mean() * 100
average_monthly = df["MonthlyCharges"].mean()
best_model = metrics.get("Best Model", "Run training")

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Customers", f"{total_customers:,}", "Clean customer records")
with col2:
    metric_card("Churn Rate", f"{churn_rate:.2f}%", "Observed churn share")
with col3:
    metric_card("Avg Monthly Charges", f"${average_monthly:.2f}", "Across dataset")
with col4:
    metric_card("Best Model", str(best_model), "Selected by F1 score")

st.divider()

left, right = st.columns([1, 1])
with left:
    st.plotly_chart(create_churn_pie(df), use_container_width=True)
with right:
    st.plotly_chart(create_churn_by_contract(df), use_container_width=True)

st.plotly_chart(create_monthly_charge_hist(df), use_container_width=True)

render_footer()
