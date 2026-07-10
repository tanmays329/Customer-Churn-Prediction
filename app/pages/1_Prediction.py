"""Single-customer churn prediction page."""

from __future__ import annotations

import streamlit as st

from components import (
    configure_page,
    metric_card,
    render_footer,
    render_header,
    risk_badge,
)
from src.predict import predict_customer
from src.explainability import create_shap_waterfall, explain_prediction
from src.utils import prepare_customer_data
from src.visualization import create_probability_gauge


configure_page("Prediction")
render_header(
    "Customer Risk Scoring",
    "Score an individual account and convert model output into retention actions.",
)

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Customer Profile")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure in Months", 0, 72, 12)
    with c2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    monthly = st.slider("Monthly Charges", 18.0, 120.0, 70.0)
    submitted = st.button("Score Customer", use_container_width=True)

with right:
    st.subheader("Prediction Output")
    if submitted:
        try:
            customer = prepare_customer_data(
                gender,
                senior,
                partner,
                dependents,
                internet,
                security,
                tech_support,
                contract,
                payment,
                tenure,
                monthly,
            )
            result = predict_customer(customer)
            st.markdown(risk_badge(result["risk"]), unsafe_allow_html=True)
            st.plotly_chart(
                create_probability_gauge(result["probability"]),
                use_container_width=True,
            )
            m1, m2 = st.columns(2)
            with m1:
                metric_card("Prediction", result["prediction"], "Model class")
            with m2:
                metric_card(
                    "Confidence",
                    f'{result["confidence"]}%',
                    "Highest class probability",
                )
            st.subheader("Recommended Actions")
            for action in result["recommendation"]:
                st.write(f"- {action}")
            with st.expander("Model Explanation"):
                explanation = explain_prediction(customer)
                if explanation.empty:
                    st.info("Local contribution explanation is unavailable for this model.")
                else:
                    st.dataframe(explanation, use_container_width=True)
                shap_fig = create_shap_waterfall(customer)
                if shap_fig is not None:
                    st.pyplot(shap_fig, clear_figure=True)
                else:
                    st.info("Install SHAP to render the waterfall explanation.")
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
    else:
        st.info("Enter account details and score the customer.")

render_footer()
