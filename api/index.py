"""
Vercel Serverless Function: FastAPI backend for Telecom Customer Churn Prediction.
Supports:
1. Account & Billing Churn Predictor (10k dataset, >= 88.9% Accuracy)
2. Indian Telecom Experience & Survey Predictor (Jio/Airtel/Vi specific, >= 86.2% Accuracy)
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
    description="Serverless API for Account Churn and Indian Telecom Experience Surveys (Jio, Airtel, Vi)",
    version="4.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load Primary Account Churn Model
MODEL_PATHS = [
    os.path.join(BASE_DIR, "decision_tree_pipeline.joblib"),
    os.path.join(BASE_DIR, "..", "models", "decision_tree_pipeline.joblib"),
]
model_pipeline = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            model_pipeline = joblib.load(path)
            print(f"Loaded primary model successfully from: {path}")
            break
        except Exception as e:
            print(f"Failed loading from {path}: {e}")

# 2. Load Survey Churn Model
SURVEY_PATHS = [
    os.path.join(BASE_DIR, "survey_churn_pipeline.joblib"),
    os.path.join(BASE_DIR, "..", "models", "survey_churn_pipeline.joblib"),
]
survey_pipeline = None
for path in SURVEY_PATHS:
    if os.path.exists(path):
        try:
            survey_pipeline = joblib.load(path)
            print(f"Loaded survey model successfully from: {path}")
            break
        except Exception as e:
            print(f"Failed loading from {path}: {e}")

NUMERICAL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "satisfaction_score", "customer_service_calls", "unresolved_complaints",
    "avg_network_speed_pct", "dissatisfaction_severity", "service_friction_index",
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
        elif c_clean in ["satisfactionscore", "satisfaction", "rating"]: col_map[col] = "satisfaction_score"
        elif c_clean in ["customerservicecalls", "servicecalls", "cscalls"]: col_map[col] = "customer_service_calls"
        elif c_clean in ["unresolvedcomplaints", "complaints", "unresolved"]: col_map[col] = "unresolved_complaints"
        elif c_clean in ["avgnetworkspeedpct", "networkspeed", "speedpct"]: col_map[col] = "avg_network_speed_pct"

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
    if "satisfaction_score" not in df: df["satisfaction_score"] = 3
    if "customer_service_calls" not in df: df["customer_service_calls"] = 1
    if "unresolved_complaints" not in df: df["unresolved_complaints"] = 0
    if "avg_network_speed_pct" not in df: df["avg_network_speed_pct"] = 90.0

    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(12).clip(lower=1)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(500.0).clip(lower=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(df["MonthlyCharges"] * df["tenure"]).clip(lower=0)
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce").fillna(3).clip(1, 5).astype(int)
    df["customer_service_calls"] = pd.to_numeric(df["customer_service_calls"], errors="coerce").fillna(1).clip(0, 10).astype(int)
    df["unresolved_complaints"] = pd.to_numeric(df["unresolved_complaints"], errors="coerce").fillna(0).clip(0, 5).astype(int)
    df["avg_network_speed_pct"] = pd.to_numeric(df["avg_network_speed_pct"], errors="coerce").fillna(90.0).clip(50.0, 100.0)

    # Contract normalization
    c_str = str(df["Contract"].iloc[0]).lower()
    if "two" in c_str or "2" in c_str:
        df["Contract"] = "Two year"
    elif "one" in c_str or "1" in c_str:
        df["Contract"] = "One year"
    else:
        df["Contract"] = "Month-to-month"

    # Feature engineering
    sat = df["satisfaction_score"].iloc[0]
    calls = df["customer_service_calls"].iloc[0]
    unres = df["unresolved_complaints"].iloc[0]

    df["dissatisfaction_severity"] = (5 - sat) * (calls + 1)
    df["service_friction_index"] = (calls * 1.5) + (unres * 2.5) - (sat * 1.2)
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
        "primary_model_loaded": model_pipeline is not None,
        "survey_model_loaded": survey_pipeline is not None,
        "primary_model_accuracy": "88.90%",
        "survey_model_accuracy": "86.20%",
        "features_count": len(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    }

@app.post("/api/predict")
def predict_churn(customer_data: Dict[str, Any]):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Primary model artifact could not be loaded.")

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
    sat = float(customer_data.get("satisfaction_score") or 3)
    calls = float(customer_data.get("customer_service_calls") or 1)
    unres = float(customer_data.get("unresolved_complaints") or 0)

    if sat <= 2:
        recommendations.append("Service Recovery: Customer reported low satisfaction. Assign dedicated VIP retention agent.")
    if unres > 0:
        recommendations.append(f"Complaint Resolution: Fast-track resolution for {int(unres)} unresolved ticket(s).")
    if calls >= 3:
        recommendations.append(f"Support Follow-up: Follow up proactively on {int(calls)} recent customer support calls.")
    if "month" in contract.lower():
        recommendations.append("Contract Commitment: Transition to 1-Year or 2-Year plan with guaranteed price-lock.")
    if tenure < 6:
        recommendations.append("Onboarding Care: Trigger proactive welcome check-in call.")
    if monthly_charges > 700:
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
        raise HTTPException(status_code=500, detail="Primary model artifact could not be loaded.")

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

# -------------------------------------------------------------
# INDIAN TELECOM SURVEY SPECIFIC INFERENCE
# -------------------------------------------------------------
RELIABILITY_MAP = {
    "Very Reliable": 5, "Reliable": 4, "Neutral": 3, "Unreliable": 2, "Very Unreliable": 1,
    5: 5, 4: 4, 3: 3, 2: 2, 1: 1
}
DROP_MAP = {
    "None": 0, "1-3": 2, "4-7": 5, "8-15": 11, "More than 15": 18, "Not Stated": 2
}

SURVEY_NUM = ['reliability_streaming', 'reliability_video_calls', 'reliability_browsing', 'reliability_gaming', 'call_drops_count', 'quality_friction_index', 'value_friction_index']
SURVEY_CAT = ['provider', 'network_type', 'plan_type', 'age_group', 'city', 'monthly_bill_range', 'monthly_data_range', 'call_drops_weekly']

def engineer_survey_dict(raw: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    if "provider" not in df: df["provider"] = "Jio"
    if "network_type" not in df: df["network_type"] = "5G"
    if "plan_type" not in df: df["plan_type"] = "Prepaid (monthly recharge)"
    if "age_group" not in df: df["age_group"] = "18–24"
    if "city" not in df: df["city"] = "Ahmedabad"
    if "monthly_bill_range" not in df: df["monthly_bill_range"] = "200–499"
    if "monthly_data_range" not in df: df["monthly_data_range"] = "21–50 GB"
    if "call_drops_weekly" not in df: df["call_drops_weekly"] = "1-3"

    for col in ["reliability_streaming", "reliability_video_calls", "reliability_browsing", "reliability_gaming"]:
        val = df[col].iloc[0] if col in df else 4
        df[col] = RELIABILITY_MAP.get(val, 4)

    drops_str = str(df["call_drops_weekly"].iloc[0]).strip()
    drops_count = DROP_MAP.get(drops_str, 2)
    df["call_drops_count"] = drops_count

    streaming = df["reliability_streaming"].iloc[0]
    calls = df["reliability_video_calls"].iloc[0]
    browsing = df["reliability_browsing"].iloc[0]
    bill = str(df["monthly_bill_range"].iloc[0])

    bill_burden = 1.5 if "1,500" in bill else (0.8 if "800" in bill else 0.0)
    df["quality_friction_index"] = (drops_count * 0.4) + (5 - calls) * 1.2 + (5 - streaming) * 0.8
    df["value_friction_index"] = bill_burden * 1.5 + (5 - browsing) * 0.6

    return df[SURVEY_NUM + SURVEY_CAT]

@app.post("/api/predict-survey")
def predict_survey(survey_data: Dict[str, Any]):
    if survey_pipeline is None:
        raise HTTPException(status_code=500, detail="Survey model artifact could not be loaded.")

    X = engineer_survey_dict(survey_data)
    pred_class = int(survey_pipeline.predict(X)[0])
    prob_churn = float(survey_pipeline.predict_proba(X)[0][1])

    provider = str(survey_data.get("provider", "Jio"))
    network = str(survey_data.get("network_type", "5G"))
    drops = str(survey_data.get("call_drops_weekly", "1-3"))
    plan = str(survey_data.get("plan_type", "Prepaid (monthly recharge)"))

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

    recs = []
    if "more than" in drops.lower() or "8-15" in drops:
        recs.append(f"Network Priority: Frequent call drops ({drops} weekly). Trigger automated cell tower RF optimization ticket.")
    if "month" in plan.lower():
        recs.append("Plan Commitment: Customer on monthly recharge. Offer Rs. 50 cashback on 3-Month or 1-Year renewal.")
    if network == "4G":
        recs.append("5G Upgrade: Offer free 5G SIM upgrade with 50GB high-speed bonus data.")
    if not recs:
        recs.append(f"Healthy Subscriber: Loyal {provider} user. Target for 5G broadband / family plan.")

    return {
        "provider": provider,
        "network_type": network,
        "prediction": "Likely to Switch in 6 Months" if pred_class == 1 else "Likely to Stay (Loyal)",
        "switch_probability": round(prob_churn * 100, 2),
        "risk_tier": tier,
        "badge_color": color,
        "recommendations": recs
    }

@app.post("/api/predict-survey-batch")
def predict_survey_batch(records: List[Dict[str, Any]]):
    if survey_pipeline is None:
        raise HTTPException(status_code=500, detail="Survey model artifact could not be loaded.")

    results = []
    for r in records:
        res = predict_survey(r)
        merged = dict(r)
        merged["Switch_Prediction_6Mo"] = res["prediction"]
        merged["Switch_Probability_Pct"] = res["switch_probability"]
        merged["Risk_Tier"] = res["risk_tier"]
        merged["Action_Plan"] = res["recommendations"][0] if res["recommendations"] else "Maintain standard service"
        results.append(merged)

    return {
        "total_scored": len(results),
        "results": results
    }
