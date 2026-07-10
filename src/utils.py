"""General business and input preparation utilities."""

from __future__ import annotations

import pandas as pd

from src.config import REQUIRED_FEATURE_COLUMNS


def prepare_customer_data(
    gender: str,
    senior: int,
    partner: str,
    dependents: str,
    internet: str,
    security: str,
    tech_support: str,
    contract: str,
    payment: str,
    tenure: int,
    monthly: float,
) -> pd.DataFrame:
    """Prepare a customer record in the format expected by the ML model."""

    total = tenure * monthly

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    return customer[REQUIRED_FEATURE_COLUMNS]


def get_risk(probability: float) -> str:
    """Return a risk label from a churn probability in the 0-1 range."""

    if probability >= 0.80:
        return "High"
    if probability >= 0.50:
        return "Medium"
    return "Low"


def get_recommendation(risk: str) -> list[str]:
    """Return business recommendations for a risk tier."""

    recommendations = {
        "High": [
            "Offer a targeted retention discount within 24 hours.",
            "Route the account to a customer success representative.",
            "Encourage migration to a longer-term contract.",
            "Review recent service issues before outreach.",
        ],
        "Medium": [
            "Send a personalized service bundle or loyalty offer.",
            "Monitor account activity over the next billing cycle.",
            "Promote support options and value-added services.",
        ],
        "Low": [
            "Maintain standard engagement and service quality.",
            "Consider low-touch loyalty messaging.",
        ],
    }
    return recommendations.get(risk, recommendations["Low"])


def confidence_from_probability(probability: float) -> float:
    """Convert class probability into prediction confidence."""

    return max(probability, 1 - probability)
