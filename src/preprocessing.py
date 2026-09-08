"""
Preprocessing and feature engineering pipeline for Telecom Customer Churn Prediction.
Optimized for robust generalization across real-world customer account distributions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "charges_per_tenure",
    "is_new_customer",
    "is_long_tenure",
    "services_count",
    "has_partner_and_dependents",
    "is_month_to_month",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
    "tenure_group",
]

def engineer_features(df_raw):
    """
    Applies domain feature engineering while remaining resilient to missing columns.
    """
    df = df_raw.copy()

    # Column normalization (case-insensitive & alias-friendly)
    col_map = {}
    for col in df.columns:
        c_clean = col.strip().lower().replace("_", "").replace(" ", "")
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
        elif c_clean in ["churn", "target"]: col_map[col] = "Churn"

    df = df.rename(columns=col_map)

    # Defaults for missing columns
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

    # Coerce numeric types
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(12).clip(lower=1)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(500.0).clip(lower=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(df["MonthlyCharges"] * df["tenure"]).clip(lower=0)

    # Clean Contract
    def normalize_contract(c):
        c_str = str(c).lower()
        if "two" in c_str or "2" in c_str:
            return "Two year"
        elif "one" in c_str or "1" in c_str:
            return "One year"
        return "Month-to-month"
    df["Contract"] = df["Contract"].apply(normalize_contract)

    # Feature engineering
    df["charges_per_tenure"] = np.round(df["MonthlyCharges"] / (df["tenure"] + 1), 2)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["is_long_tenure"] = (df["tenure"] >= 24).astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["has_partner_and_dependents"] = ((df["Partner"] == "Yes") & (df["Dependents"] == "Yes")).astype(int)

    # Tenure group binning
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72, 1000],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m", "72m+"]
    ).astype(str)

    # Count of active digital services
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    count_series = pd.Series(0, index=df.index)
    for sc in service_cols:
        count_series += (df[sc] == "Yes").astype(int)
    df["services_count"] = count_series

    return df

def load_and_clean_data(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "telecom_churn.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading dataset from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df = engineer_features(df_raw)

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].apply(lambda v: 1 if str(v).strip().lower() in ["yes", "1", "true"] else 0).astype(int)

    return df

def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    return preprocessor

def get_train_test_data(csv_path=None, test_size=0.2, random_state=42):
    df = load_and_clean_data(csv_path=csv_path)

    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[all_features].copy()
    y = df["Churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Dataset split: Train = {X_train.shape[0]:,} samples, Test = {X_test.shape[0]:,} samples")
    print(f"Train Churn Rate: {y_train.mean():.2%}, Test Churn Rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test
