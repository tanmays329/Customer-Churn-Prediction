# Customer Churn Prediction System

A production-style machine learning portfolio project for predicting telecom customer churn, exploring customer behavior, and translating model outputs into retention actions.

## Project Overview and introduction

This project uses a Scikit-learn pipeline to train a churn classifier and a Streamlit analytics app to serve business users. The application supports single-customer prediction, batch CSV scoring, model performance review, customer data exploration, and business insight dashboards.

## Architecture

- `src/` contains data loading, preprocessing, training, prediction, reporting, explainability, logging, and visualization utilities.
- `app/` contains the Streamlit user interface and reusable UI components.
- `data/` stores raw and cleaned customer datasets.
- `models/` stores the trained model artifact.
- `reports/` stores evaluation artifacts used by dashboards without retraining.
- `assets/` stores custom styling.
- `logs/` stores runtime and training logs.

## Folder Structure

```text
Customer Churn Prediction/
├── app/
│   ├── Home.py
│   ├── components.py
│   └── pages/
│       ├── 1_Prediction.py
│       ├── 2_Data_Explorer.py
│       ├── 3_Model_Performance.py
│       ├── 4_Business_Insights.py
│       └── 5_Batch_Prediction.py
├── assets/
│   └── style.css
├── data/
│   ├── customer_data.csv
│   └── clean_customer_data.csv
├── models/
│   └── churn_model.pkl
├── reports/
│   ├── metrics.pkl
│   ├── classification_report.csv
│   ├── feature_importance.csv
│   ├── confusion_matrix.npy
│   ├── roc.pkl
│   └── model_comparison.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── logging.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── reports.py
│   ├── train.py
│   ├── utils.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model and generate dashboard artifacts:

```bash
python -m src.train
```

Run the Streamlit app:

```bash
streamlit run app/Home.py
```

## Model Details

- Production model: Logistic Regression
- Preprocessing: one-hot encoding for categorical variables with passthrough numeric columns
- Evaluation artifacts: accuracy, precision, recall, F1 score, ROC AUC, classification report, confusion matrix, feature importance, and model comparison
- Explainability: coefficient-based local contributions with optional SHAP waterfall plots

## Features

- Executive KPI dashboard
- Individual churn prediction with probability, risk tier, confidence, and recommendations
- Batch CSV upload, validation, prediction, risk labeling, and download
- Model performance dashboard loaded from saved reports
- Data explorer with search, filters, sliders, downloads, and correlation heatmap
- Business insights dashboard with customer statistics, churn segments, and recommendations
- Centralized configuration, logging, error handling, and reusable Streamlit components


## Future Improvements

- Add automated unit tests for dashboard report loading and batch validation.
- Track model drift using scheduled evaluation data.
- Add experiment tracking with MLflow.
- Add authentication for internal business use.
- Add a FastAPI service layer for production API deployment.

## Author

Tanmay
