"""
Vercel Serverless Function: FastAPI backend for Telecom Customer Churn Prediction.
Loads the Tuned Decision Tree Classifier (99.12% Accuracy).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Serverless API powered by a 99.12% Accuracy Decision Tree Model",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading logic (checks api/ first, then models/)
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

class CustomerProfile(BaseModel):
    telecom_partner: str = Field(default="Airtel", description="Reliance Jio, Airtel, Vodafone, BSNL")
    gender: str = Field(default="M", description="M or F")
    age: int = Field(default=35, ge=18, le=90)
    city: str = Field(default="Delhi", description="Delhi, Mumbai, Bangalore, Chennai, Hyderabad, Kolkata")
    num_dependents: int = Field(default=1, ge=0, le=10)
    estimated_salary: float = Field(default=85000.0, ge=10000.0)
    calls_made: float = Field(default=55.0, ge=0.0)
    sms_sent: float = Field(default=20.0, ge=0.0)
    data_used_gb: float = Field(default=8.5, ge=0.0)
    tenure_months: float = Field(default=14.0, ge=0.1)
    contract_type: str = Field(default="Month-to-month", description="Month-to-month, 1-Year, 2-Year")
    payment_method: str = Field(default="UPI / Auto-Debit")
    monthly_charges: float = Field(default=549.0, ge=100.0)
    customer_service_calls: int = Field(default=1, ge=0)
    tech_support_tickets: int = Field(default=0, ge=0)
    satisfaction_rating: int = Field(default=3, ge=1, le=5)
    unresolved_complaints: int = Field(default=0, ge=0, le=1)

def format_features(profile_dict):
    df = pd.DataFrame([profile_dict])
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

@app.get("/api/metrics")
def get_metrics():
    return {
        "model": "Decision Tree Classifier",
        "accuracy": 0.9912,
        "roc_auc": 0.9950,
        "precision": 0.9910,
        "recall": 0.9809,
        "f1_score": 0.9859,
        "cross_validation_accuracy": "98.76% +/- 0.09%",
        "top_features": [
            {"feature": "service_friction_index", "importance": 0.5929},
            {"feature": "contract_type_Month-to-month", "importance": 0.1936},
            {"feature": "tenure_months", "importance": 0.1182},
            {"feature": "dissatisfaction_severity", "importance": 0.0432},
            {"feature": "monthly_charges", "importance": 0.0192}
        ]
    }

@app.post("/api/predict")
def predict_churn(customer: CustomerProfile):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model artifact could not be loaded.")

    data = customer.model_dump()
    X = format_features(data)

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
    if customer.satisfaction_rating <= 2 or customer.unresolved_complaints == 1:
        recommendations.append("Priority Escalation: Open grievance requires urgent resolution & compensatory credit.")
    if customer.customer_service_calls >= 3:
        recommendations.append("High Friction Alert: Assign senior relationship manager to review network stability.")
    if customer.contract_type == "Month-to-month":
        recommendations.append("Contract Lock-in: Offer discounted 1-Year or 2-Year plan with extra data/OTT.")
    if customer.monthly_charges > 700:
        recommendations.append("Billing Optimization: Suggest competitive family plan to relieve monthly tariff pressure.")
    if customer.tenure_months < 6:
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
def predict_batch(customers: List[CustomerProfile]):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model artifact could not be loaded.")

    results = []
    for c in customers:
        res = predict_churn(c)
        results.append(res)
    return {"total_scored": len(results), "results": results}
