"""
Vercel Serverless Function: FastAPI backend for Telecom Customer Churn Prediction.
Loads the regularized Decision Tree Classifier (Zero Overfitting).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Serverless API powered by a regularized, non-overfitted Decision Tree Model",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [
    os.path.join(BASE_DIR, "decision_tree_pipeline.joblib"),
    os.path.join(BASE_DIR, "..", "models", "decision_tree_pipeline.joblib"),
]

model_pipeline = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            model_pipeline = joblib.load(path)
            print(f"Loaded model successfully from: {path}")
            break
        except Exception as e:
            print(f"Failed loading from {path}: {e}")

NUMERICAL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "charges_per_tenure", "is_new_customer", "is_long_tenure",
    "services_count", "has_partner_and_dependents", "is_month_to_month"
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "tenure_group"
]

def engineer_features_dict(raw: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    col_map = {}
    for col in df.columns:
        c_clean = str(col).strip().lower().replace("_", "").replace(" ", "")
        if c_clean == "gender": col_map[col] = "gender"
        elif c_clean in ["seniorcitizen", "senior"]: col_map[col] = "SeniorCitizen"
        elif c_clean == "partner": col_map[col] = "Partner"
        elif c_clean in ["dependents", "numdependents"]: col_map[col] = "Dependents"
        elif c_clean in ["tenure", "tenuremonths"]: col_map[col] = "tenure"
        elif c_clean in ["monthlycharges", "monthlybill", "bill"]: col_map[col] = "MonthlyCharges"
        elif c_clean in ["totalcharges", "totalbill"]: col_map[col] = "TotalCharges"
        elif c_clean == "phoneservice": col_map[col] = "PhoneService"
        elif c_clean == "multiplelines": col_map[col] = "MultipleLines"
        elif c_clean in ["internetservice", "internet"]: col_map[col] = "InternetService"
        elif c_clean == "onlinesecurity": col_map[col] = "OnlineSecurity"
        elif c_clean == "onlinebackup": col_map[col] = "OnlineBackup"
        elif c_clean == "deviceprotection": col_map[col] = "DeviceProtection"
        elif c_clean in ["techsupport", "support"]: col_map[col] = "TechSupport"
        elif c_clean == "streamingtv": col_map[col] = "StreamingTV"
        elif c_clean == "streamingmovies": col_map[col] = "StreamingMovies"
        elif c_clean in ["contract", "contracttype"]: col_map[col] = "Contract"
        elif c_clean == "paperlessbilling": col_map[col] = "PaperlessBilling"
        elif c_clean in ["paymentmethod", "payment"]: col_map[col] = "PaymentMethod"

    df = df.rename(columns=col_map)

    # Defaults
    if "gender" not in df: df["gender"] = "Male"
    if "SeniorCitizen" not in df: df["SeniorCitizen"] = 0
    if "Partner" not in df: df["Partner"] = "No"
    if "Dependents" not in df: df["Dependents"] = "No"
    if "tenure" not in df: df["tenure"] = 12
    if "MonthlyCharges" not in df: df["MonthlyCharges"] = 500.0
    if "TotalCharges" not in df: df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    if "PhoneService" not in df: df["PhoneService"] = "Yes"
    if "MultipleLines" not in df: df["MultipleLines"] = "No"
    if "InternetService" not in df: df["InternetService"] = "Fiber optic"
    if "OnlineSecurity" not in df: df["OnlineSecurity"] = "No"
    if "OnlineBackup" not in df: df["OnlineBackup"] = "No"
    if "DeviceProtection" not in df: df["DeviceProtection"] = "No"
    if "TechSupport" not in df: df["TechSupport"] = "No"
    if "StreamingTV" not in df: df["StreamingTV"] = "Yes"
    if "StreamingMovies" not in df: df["StreamingMovies"] = "Yes"
    if "Contract" not in df: df["Contract"] = "Month-to-month"
    if "PaperlessBilling" not in df: df["PaperlessBilling"] = "Yes"
    if "PaymentMethod" not in df: df["PaymentMethod"] = "Electronic check"

    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(12).clip(lower=1)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(500.0).clip(lower=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(df["MonthlyCharges"] * df["tenure"]).clip(lower=0)

    # Contract normalization
    c_str = str(df["Contract"].iloc[0]).lower()
    if "two" in c_str or "2" in c_str:
        df["Contract"] = "Two year"
    elif "one" in c_str or "1" in c_str:
        df["Contract"] = "One year"
    else:
        df["Contract"] = "Month-to-month"

    # Feature engineering
    df["charges_per_tenure"] = np.round(df["MonthlyCharges"] / (df["tenure"] + 1), 2)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["is_long_tenure"] = (df["tenure"] >= 24).astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["has_partner_and_dependents"] = ((df["Partner"] == "Yes") & (df["Dependents"] == "Yes")).astype(int)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72, 1000],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m", "72m+"]
    ).astype(str)

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    count_val = sum((df[sc].iloc[0] == "Yes") for sc in service_cols if sc in df)
    df["services_count"] = count_val

    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    return df[all_features]

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "model_architecture": "Regularized Decision Tree (Zero Overfitting)",
        "cross_val_auc": "0.5952",
        "features_count": len(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    }

@app.post("/api/predict")
def predict_churn(customer_data: Dict[str, Any]):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model artifact could not be loaded.")

    X = engineer_features_dict(customer_data)
    pred_class = int(model_pipeline.predict(X)[0])
    prob_churn = float(model_pipeline.predict_proba(X)[0][1])

    if prob_churn >= 0.65:
        tier = "Critical Risk"
        color = "#d32f2f"
    elif prob_churn >= 0.50:
        tier = "High Risk"
        color = "#f57c00"
    elif prob_churn >= 0.35:
        tier = "Moderate Risk"
        color = "#fbc02d"
    else:
        tier = "Low Risk"
        color = "#2e7d32"

    recommendations = []
    contract = str(customer_data.get("Contract") or customer_data.get("contract_type") or "Month-to-month")
    tenure = float(customer_data.get("tenure") or customer_data.get("tenure_months") or 12)
    monthly_charges = float(customer_data.get("MonthlyCharges") or customer_data.get("monthly_charges") or 500.0)

    if "month" in contract.lower():
        recommendations.append("Contract Commitment: Transition to 1-Year or 2-Year plan with bonus data.")
    if tenure < 6:
        recommendations.append("Onboarding Care: Trigger proactive welcome check-in call.")
    if monthly_charges > 600:
        recommendations.append("Tariff Optimization: Propose competitive family plan to lower monthly spend.")
    if not recommendations:
        recommendations.append("Healthy Account: High retention probability. Candidate for premium service bundle.")

    return {
        "prediction": "Likely to Churn" if pred_class == 1 else "Likely to Stay (Retained)",
        "churn_class": pred_class,
        "churn_probability": round(prob_churn * 100, 2),
        "risk_tier": tier,
        "badge_color": color,
        "recommendations": recommendations
    }

@app.post("/api/predict-batch")
def predict_batch(records: List[Dict[str, Any]]):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model artifact could not be loaded.")

    scored_records = []
    for r in records:
        res = predict_churn(r)
        merged = dict(r)
        merged["Churn_Prediction"] = res["prediction"]
        merged["Churn_Probability_Pct"] = res["churn_probability"]
        merged["Risk_Tier"] = res["risk_tier"]
        merged["Recommended_Action"] = res["recommendations"][0] if res["recommendations"] else "Maintain standard service"
        scored_records.append(merged)

    return {
        "total_scored": len(scored_records),
        "results": scored_records
    }
