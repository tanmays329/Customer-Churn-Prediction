"""Interactive data explorer page."""

from __future__ import annotations

import streamlit as st

from components import configure_page, metric_card, render_footer, render_header
from src.data_loader import load_data
from src.visualization import (
    create_churn_by_contract,
    create_churn_pie,
    create_contract_bar,
    create_correlation_heatmap,
    create_gender_churn,
    create_internet_service_chart,
    create_monthly_charge_hist,
    create_payment_chart,
    create_tenure_hist,
)


configure_page("Data Explorer")
render_header(
    "Customer Data Explorer",
    "Filter, search, inspect, and export the cleaned customer churn dataset.",
)


@st.cache_data(show_spinner=False)
def cached_data():
    """Load and cache the cleaned dataset."""

    return load_data()


df = cached_data()

st.sidebar.header("Filters")
search_text = st.sidebar.text_input("Search customers", "")
contract = st.sidebar.multiselect(
    "Contract",
    sorted(df["Contract"].dropna().unique()),
)
internet = st.sidebar.multiselect(
    "Internet Service",
    sorted(df["InternetService"].dropna().unique()),
)
gender = st.sidebar.multiselect("Gender", sorted(df["gender"].dropna().unique()))
churn = st.sidebar.multiselect("Churn", sorted(df["Churn"].dropna().unique()))
tenure_range = st.sidebar.slider(
    "Tenure Range",
    int(df["tenure"].min()),
    int(df["tenure"].max()),
    (int(df["tenure"].min()), int(df["tenure"].max())),
)
monthly_range = st.sidebar.slider(
    "Monthly Charges Range",
    float(df["MonthlyCharges"].min()),
    float(df["MonthlyCharges"].max()),
    (float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())),
)

filtered_df = df.copy()
if search_text:
    search_mask = filtered_df.astype(str).apply(
        lambda column: column.str.contains(search_text, case=False, na=False)
    )
    filtered_df = filtered_df[search_mask.any(axis=1)]
if contract:
    filtered_df = filtered_df[filtered_df["Contract"].isin(contract)]
if internet:
    filtered_df = filtered_df[filtered_df["InternetService"].isin(internet)]
if gender:
    filtered_df = filtered_df[filtered_df["gender"].isin(gender)]
if churn:
    filtered_df = filtered_df[filtered_df["Churn"].isin(churn)]

filtered_df = filtered_df[
    filtered_df["tenure"].between(*tenure_range)
    & filtered_df["MonthlyCharges"].between(*monthly_range)
]

customers = len(filtered_df)
churn_rate = filtered_df["Churn"].eq("Yes").mean() * 100 if customers else 0
avg_monthly = filtered_df["MonthlyCharges"].mean() if customers else 0
avg_tenure = filtered_df["tenure"].mean() if customers else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Customers", f"{customers:,}", "After filters")
with c2:
    metric_card("Churn Rate", f"{churn_rate:.2f}%", "Filtered segment")
with c3:
    metric_card("Avg Monthly Charges", f"${avg_monthly:.2f}", "Filtered segment")
with c4:
    metric_card("Avg Tenure", f"{avg_tenure:.1f}", "Months")

st.subheader("Dataset")
st.dataframe(filtered_df, use_container_width=True, height=390)
st.download_button(
    "Download Filtered Dataset",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_customers.csv",
    mime="text/csv",
    use_container_width=True,
)

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
else:
    tab1, tab2, tab3 = st.tabs(["Overview", "Segments", "Correlation"])
    with tab1:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(create_churn_pie(filtered_df), use_container_width=True)
            st.plotly_chart(create_monthly_charge_hist(filtered_df), use_container_width=True)
        with right:
            st.plotly_chart(create_contract_bar(filtered_df), use_container_width=True)
            st.plotly_chart(create_tenure_hist(filtered_df), use_container_width=True)
    with tab2:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(create_payment_chart(filtered_df), use_container_width=True)
            st.plotly_chart(create_churn_by_contract(filtered_df), use_container_width=True)
        with right:
            st.plotly_chart(create_internet_service_chart(filtered_df), use_container_width=True)
            st.plotly_chart(create_gender_churn(filtered_df), use_container_width=True)
    with tab3:
        st.plotly_chart(create_correlation_heatmap(filtered_df), use_container_width=True)

render_footer()
