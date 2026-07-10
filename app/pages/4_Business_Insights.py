"""Business insights dashboard."""

from __future__ import annotations

import streamlit as st

from components import configure_page, metric_card, render_footer, render_header
from src.data_loader import load_data
from src.reports import ReportLoadError, load_feature_importance
from src.visualization import (
    create_churn_by_contract,
    create_gender_churn,
    create_internet_churn,
    create_monthly_charges_box,
    create_payment_churn,
    create_tenure_hist,
)


configure_page("Business Insights")
render_header(
    "Business Insights",
    "Translate churn patterns into customer retention priorities.",
)


@st.cache_data(show_spinner=False)
def cached_data():
    """Load and cache the cleaned dataset."""

    return load_data()


@st.cache_data(show_spinner=False)
def cached_feature_importance():
    """Load feature importance when available."""

    try:
        return load_feature_importance()
    except ReportLoadError:
        return None


df = cached_data()
feature_importance = cached_feature_importance()

total_customers = len(df)
churn_percentage = df["Churn"].eq("Yes").mean() * 100
avg_monthly = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Customers", f"{total_customers:,}", "Analyzed accounts")
with c2:
    metric_card("Churn Percentage", f"{churn_percentage:.2f}%", "Dataset baseline")
with c3:
    metric_card("Avg Monthly Charges", f"${avg_monthly:.2f}", "Revenue exposure")
with c4:
    metric_card("Avg Tenure", f"{avg_tenure:.1f}", "Months")

st.subheader("Top Risk Factors")
if feature_importance is not None and not feature_importance.empty:
    st.dataframe(feature_importance.head(10), use_container_width=True)
else:
    st.info("Run training to generate feature importance.")

left, right = st.columns(2)
with left:
    st.plotly_chart(create_churn_by_contract(df), use_container_width=True)
    st.plotly_chart(create_internet_churn(df), use_container_width=True)
    st.plotly_chart(create_monthly_charges_box(df), use_container_width=True)
with right:
    st.plotly_chart(create_payment_churn(df), use_container_width=True)
    st.plotly_chart(create_gender_churn(df), use_container_width=True)
    st.plotly_chart(create_tenure_hist(df), use_container_width=True)

st.subheader("Business Recommendations")
recommendations = [
    "Prioritize month-to-month customers for contract migration offers.",
    "Create a save desk workflow for high monthly charge customers with short tenure.",
    "Audit fiber optic and electronic check segments where churn concentration is typically higher.",
    "Bundle technical support and online security in retention campaigns.",
    "Track recall and precision monthly before using churn scores for automated incentives.",
]
for recommendation in recommendations:
    st.write(f"- {recommendation}")

render_footer()
