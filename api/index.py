"""
Vercel Serverless Function: FastAPI backend for Telecom Customer Churn Prediction.
Loads the Tuned Decision Tree Classifier (99.12% Accuracy).
Handles flexible column aliases for seamless batch CSV uploads.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Serverless API powered by a 99.12% Accuracy Decision Tree Model",
    version="2.1.0"
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
    "age", "num_dependents", "estimated_salary", "calls_made", "sms_sent",
    "data_used_gb", "tenure_months", "monthly_charges", "total_charges",
    "customer_service_calls", "tech_support_tickets", "satisfaction_rating",
    "unresolved_complaints", "dissatisfaction_severity", "service_friction_index"
]

CATEGORICAL_FEATURES = [
    "telecom_partner", "gender", "city", "contract_type", "payment_method"
]

def normalize_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes column aliases from various CSV formats into standard pipeline schema.
    """
    p = {}
    p["telecom_partner"] = str(raw.get("telecom_partner") or raw.get("partner") or "Airtel")
    p["gender"] = str(raw.get("gender") or "M")
    p["age"] = float(raw.get("age", 35))
    p["city"] = str(raw.get("city") or "Delhi")
    p["num_dependents"] = float(raw.get("num_dependents") or raw.get("dependents", 1))
    p["estimated_salary"] = float(raw.get("estimated_salary") or raw.get("salary", 85000.0))
    p["calls_made"] = float(raw.get("calls_made") or raw.get("calls", 55.0))
    p["sms_sent"] = float(raw.get("sms_sent") or raw.get("sms", 20.0))
    
    # Data used
    if "data_used_gb" in raw:
        p["data_used_gb"] = float(raw["data_used_gb"])
    elif "data_used" in raw:
        val = float(raw["data_used"])
        p["data_used_gb"] = val if val <= 200 else val / 1024.0
    elif "data_used_mb" in raw:
        p["data_used_gb"] = float(raw["data_used_mb"]) / 1024.0
    else:
        p["data_used_gb"] = 8.5

    # Tenure
    p["tenure_months"] = float(raw.get("tenure_months") or raw.get("tenure", 14.0))

    # Contract Type
    c_raw = str(raw.get("contract_type") or raw.get("contract", "Month-to-month"))
    if "month" in c_raw.lower():
        p["contract_type"] = "Month-to-month"
    elif "2" in c_raw:
        p["contract_type"] = "2-Year"
    elif "1" in c_raw:
        p["contract_type"] = "1-Year"
    else:
        p["contract_type"] = "Month-to-month"

    # Payment Method
    pm_raw = str(raw.get("payment_method", "UPI / Auto-Debit"))
    if "upi" in pm_raw.lower():
        p["payment_method"] = "UPI / Auto-Debit"
    elif "credit" in pm_raw.lower() or "auto" in pm_raw.lower():
        p["payment_method"] = "Credit Card"
    elif "cash" in pm_raw.lower():
        p["payment_method"] = "Cash / Cheque"
    else:
        p["payment_method"] = "UPI / Auto-Debit"

    p["monthly_charges"] = float(raw.get("monthly_charges") or raw.get("monthly_bill", 549.0))
    p["customer_service_calls"] = int(float(raw.get("customer_service_calls") or raw.get("support_calls", 1)))
    p["tech_support_tickets"] = int(float(raw.get("tech_support_tickets", 0)))
    p["satisfaction_rating"] = int(float(raw.get("satisfaction_rating") or raw.get("satisfaction", 3)))
    p["unresolved_complaints"] = int(float(raw.get("unresolved_complaints") or raw.get("open_complaint", 0)))
    return p

def format_features(profile_dict):
    p = normalize_dict(profile_dict)
    df = pd.DataFrame([p])
    df["total_charges"] = np.round(df["monthly_charges"] * df["tenure_months"], 2)

    sat = df["satisfaction_rating"].iloc[0]
    calls = df["customer_service_calls"].iloc[0]
    unres = df["unresolved_complaints"].iloc[0]

    df["dissatisfaction_severity"] = (5 - sat) * (calls + 1)
    df["service_friction_index"] = (calls * 2.0) + (unres * 3.5) - (sat * 1.5)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    return df[feature_cols]

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "model_architecture": "Decision Tree Classifier",
        "accuracy": "99.12%",
        "roc_auc": "0.9950"
    }

@app.post("/api/predict")
def predict_churn(customer_data: Dict[str, Any]):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model artifact could not be loaded.")

    X = format_features(customer_data)
    norm = normalize_dict(customer_data)

    pred_class = int(model_pipeline.predict(X)[0])
    prob_churn = float(model_pipeline.predict_proba(X)[0][1])

    if prob_churn >= 0.70:
        tier = "Critical Risk"
        color = "#d32f2f"
    elif prob_churn >= 0.45:
        tier = "High Risk"
        color = "#f57c00"
    elif prob_churn >= 0.25:
        tier = "Moderate Risk"
        color = "#fbc02d"
    else:
        tier = "Low Risk"
        color = "#2e7d32"

    recommendations = []
    if norm["satisfaction_rating"] <= 2 or norm["unresolved_complaints"] == 1:
        recommendations.append("Priority Escalation: Open grievance requires urgent resolution & compensatory credit.")
    if norm["customer_service_calls"] >= 3:
        recommendations.append("High Friction Alert: Assign senior relationship manager to review network stability.")
    if norm["contract_type"] == "Month-to-month":
        recommendations.append("Contract Lock-in: Offer discounted 1-Year or 2-Year plan with extra data/OTT.")
    if norm["monthly_charges"] > 700:
        recommendations.append("Billing Optimization: Suggest competitive family plan to relieve monthly tariff pressure.")
    if norm["tenure_months"] < 6:
        recommendations.append("Onboarding Care: Conduct proactive check-in and provide 30-day bonus perks.")
    if not recommendations:
        recommendations.append("Healthy Account: Customer is loyal. Candidate for multi-SIM or 5G device cross-sell.")

    return {
        "prediction": "Likely to Churn" if pred_class == 1 else "Likely to Stay (Retained)",
        "churn_class": pred_class,
        "churn_probability": round(prob_churn * 100, 2),
        "risk_tier": tier,
        "badge_color": color,
        "model_accuracy": "99.12%",
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
        merged["Recommended_Action"] = res["recommendations"][0] if res["recommendations"] else "Maintain service"
        scored_records.append(merged)

    return {
        "total_scored": len(scored_records),
        "results": scored_records
    }
