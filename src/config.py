"""Central configuration for the customer churn project."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

RAW_DATA_PATH = DATA_DIR / "customer_data.csv"
DATA_PATH = DATA_DIR / "clean_customer_data.csv"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"

METRICS_PATH = REPORTS_DIR / "metrics.pkl"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.npy"
ROC_PATH = REPORTS_DIR / "roc.pkl"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

STYLE_PATH = ASSETS_DIR / "style.css"
LOG_PATH = LOGS_DIR / "app.log"

APP_NAME = "Customer Churn Prediction System"
APP_TAGLINE = "Telecom retention analytics and churn risk scoring"
VERSION = "2.0.0"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

REQUIRED_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

RISK_THRESHOLDS = {
    "High": 0.80,
    "Medium": 0.50,
}

CHURN_LABELS = {
    0: NEGATIVE_LABEL,
    1: POSITIVE_LABEL,
}

for directory in (ASSETS_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
