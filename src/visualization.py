"""Plotly visualization helpers used across Streamlit dashboards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------
# Color Theme
# -----------------------------------------------------

PRIMARY = "#2457c5"
SECONDARY = "#4f8f7b"
SUCCESS = "#16825d"
DANGER = "#c2413d"
WARNING = "#b7791f"

COLOR_SEQUENCE = [
    PRIMARY,
    SECONDARY,
    "#4f8f7b",
    "#d99a3e",
    "#8a6fb1",
]


# -----------------------------------------------------
# Common Layout
# -----------------------------------------------------

def apply_layout(fig, title: str, height: int = 430):
    """Apply the common chart layout."""

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        font=dict(
            family="Arial",
            size=14
        ),
        legend_title_text="",
    )

    return fig

# -----------------------------------------------------
# Churn Distribution
# -----------------------------------------------------

def create_churn_pie(df: pd.DataFrame) -> go.Figure:

    fig = px.pie(
        df,
        names="Churn",
        hole=0.45,
        color="Churn",
        color_discrete_map={
            "No": SUCCESS,
            "Yes": DANGER
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    return apply_layout(
        fig,
        "Customer Churn Distribution"
    )


# -----------------------------------------------------
# Contract Distribution
# -----------------------------------------------------

def create_contract_bar(df: pd.DataFrame) -> go.Figure:

    contract = (
        df["Contract"]
        .value_counts()
        .reset_index()
    )

    contract.columns = [
        "Contract",
        "Customers"
    ]

    fig = px.bar(
        contract,
        x="Contract",
        y="Customers",
        color="Contract",
        text="Customers",
        color_discrete_sequence=COLOR_SEQUENCE
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False
    )

    return apply_layout(
        fig,
        "Contract Distribution"
    )


# -----------------------------------------------------
# Monthly Charges Histogram
# -----------------------------------------------------

def create_monthly_charge_hist(df: pd.DataFrame) -> go.Figure:

    fig = px.histogram(
        df,
        x="MonthlyCharges",
        nbins=30,
        color_discrete_sequence=[PRIMARY]
    )

    return apply_layout(
        fig,
        "Monthly Charges Distribution"
    )


# -----------------------------------------------------
# Tenure Histogram
# -----------------------------------------------------

def create_tenure_hist(df: pd.DataFrame) -> go.Figure:

    fig = px.histogram(
        df,
        x="tenure",
        nbins=25,
        color_discrete_sequence=[SECONDARY]
    )

    return apply_layout(
        fig,
        "Customer Tenure Distribution"
    )


# -----------------------------------------------------
# Internet Service Distribution
# -----------------------------------------------------

def create_internet_service_chart(df: pd.DataFrame) -> go.Figure:

    internet = (
        df["InternetService"]
        .value_counts()
        .reset_index()
    )

    internet.columns = [
        "Internet Service",
        "Customers"
    ]

    fig = px.bar(
        internet,
        x="Internet Service",
        y="Customers",
        color="Internet Service",
        text="Customers",
        color_discrete_sequence=COLOR_SEQUENCE
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False
    )

    return apply_layout(
        fig,
        "Internet Service Distribution"
    )


# -----------------------------------------------------
# Payment Method Distribution
# -----------------------------------------------------

def create_payment_chart(df: pd.DataFrame) -> go.Figure:

    payment = (
        df["PaymentMethod"]
        .value_counts()
        .reset_index()
    )

    payment.columns = [
        "Payment Method",
        "Customers"
    ]

    fig = px.bar(
        payment,
        x="Payment Method",
        y="Customers",
        color="Payment Method",
        text="Customers",
        color_discrete_sequence=COLOR_SEQUENCE
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False
    )

    return apply_layout(
        fig,
        "Payment Method Distribution"
    )


# -----------------------------------------------------
# Churn by Contract
# -----------------------------------------------------

def create_churn_by_contract(df: pd.DataFrame) -> go.Figure:

    temp = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        temp,
        x="Contract",
        y="Customers",
        color="Churn",
        barmode="group",
        color_discrete_map={
            "No": SUCCESS,
            "Yes": DANGER
        }
    )

    return apply_layout(
        fig,
        "Churn by Contract Type"
    )


# -----------------------------------------------------
# Churn by Gender
# -----------------------------------------------------

def create_gender_churn(df: pd.DataFrame) -> go.Figure:

    temp = (
        df.groupby(["gender", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        temp,
        x="gender",
        y="Customers",
        color="Churn",
        barmode="group",
        color_discrete_map={
            "No": SUCCESS,
            "Yes": DANGER
        }
    )

    return apply_layout(
        fig,
        "Gender vs Churn"
    )


# -----------------------------------------------------
# Churn by Internet Service
# -----------------------------------------------------

def create_internet_churn(df: pd.DataFrame) -> go.Figure:

    temp = (
        df.groupby(["InternetService", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        temp,
        x="InternetService",
        y="Customers",
        color="Churn",
        barmode="group",
        color_discrete_map={
            "No": SUCCESS,
            "Yes": DANGER
        }
    )

    return apply_layout(
        fig,
        "Internet Service vs Churn"
    )


# -----------------------------------------------------
# Correlation Heatmap
# -----------------------------------------------------

def create_payment_churn(df: pd.DataFrame) -> go.Figure:
    """Create churn counts by payment method."""

    temp = (
        df.groupby(["PaymentMethod", "Churn"])
        .size()
        .reset_index(name="Customers")
    )

    fig = px.bar(
        temp,
        x="PaymentMethod",
        y="Customers",
        color="Churn",
        barmode="group",
        color_discrete_map={"No": SUCCESS, "Yes": DANGER},
    )
    fig.update_xaxes(tickangle=-25)
    return apply_layout(fig, "Churn by Payment Method")


def create_monthly_charges_box(df: pd.DataFrame) -> go.Figure:
    """Create monthly charges boxplot by churn label."""

    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        color_discrete_map={"No": SUCCESS, "Yes": DANGER},
    )
    return apply_layout(fig, "Monthly Charges by Churn")


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:

    numeric = df.select_dtypes(include=["int64", "float64"])

    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto"
    )

    return apply_layout(
        fig,
        "Correlation Heatmap"
    )


def create_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Create a labeled confusion matrix heatmap."""

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Customers"),
        x=["No Churn", "Churn"],
        y=["No Churn", "Churn"],
    )
    return apply_layout(fig, "Confusion Matrix", height=390)


def create_roc_curve(roc: dict) -> go.Figure:
    """Create an ROC curve chart from saved arrays."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name=f"AUC {roc['auc']:.3f}",
            line=dict(color=PRIMARY, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline",
            line=dict(color="#9aa6b2", dash="dash"),
        )
    )
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return apply_layout(fig, "ROC Curve", height=390)


def create_feature_importance_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create a horizontal feature importance chart."""

    top_features = df.head(top_n).sort_values("Importance", ascending=True)
    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Teal",
    )
    return apply_layout(fig, "Top Feature Importance")


def create_probability_gauge(probability: float) -> go.Figure:
    """Create a churn probability gauge."""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PRIMARY},
                "steps": [
                    {"range": [0, 50], "color": "#e8f7f0"},
                    {"range": [50, 80], "color": "#fff4df"},
                    {"range": [80, 100], "color": "#fdecec"},
                ],
            },
        )
    )
    return apply_layout(fig, "Churn Probability", height=300)
