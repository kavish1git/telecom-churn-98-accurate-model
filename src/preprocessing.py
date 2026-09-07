"""
Preprocessing and feature engineering pipeline for the Maximized Accuracy Churn System (>= 98%).
Covers all 4 core pillars plus domain synergy interaction metrics:
1. Demographics: age, gender, num_dependents, city
2. Usage Patterns: calls_made, sms_sent, data_used_gb, tenure_months
3. Billing Info: contract_type, payment_method, monthly_charges, total_charges, estimated_salary
4. Service Feedback: customer_service_calls, tech_support_tickets, satisfaction_rating, unresolved_complaints
5. Domain Interactions: dissatisfaction_severity, service_friction_index
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = [
    "age",
    "num_dependents",
    "estimated_salary",
    "calls_made",
    "sms_sent",
    "data_used_gb",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "customer_service_calls",
    "tech_support_tickets",
    "satisfaction_rating",
    "unresolved_complaints",
    "dissatisfaction_severity",
    "service_friction_index",
]

CATEGORICAL_FEATURES = [
    "telecom_partner",
    "gender",
    "city",
    "contract_type",
    "payment_method",
]

def load_and_clean_data(csv_path=None, sample_size=None, random_state=42):
    if csv_path is None:
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "telecom_churn_refined.csv"
        )
        if not os.path.exists(csv_path):
            csv_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "telecom_churn.csv"
            )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size:,} records from {len(df):,} total for optimized training speed...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # Ensure non-negative numbers
    for col in ["calls_made", "sms_sent", "customer_service_calls", "tech_support_tickets", "monthly_charges"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    if "data_used_gb" not in df.columns and "data_used_mb" in df.columns:
        df["data_used_gb"] = np.round(df["data_used_mb"].clip(lower=0) / 1024.0, 2)
    elif "data_used_gb" not in df.columns and "data_used" in df.columns:
        df["data_used_gb"] = np.round(df["data_used"].clip(lower=0) / 1024.0, 2)

    if "total_charges" not in df.columns and "monthly_charges" in df.columns and "tenure_months" in df.columns:
        df["total_charges"] = np.round(df["monthly_charges"] * df["tenure_months"], 2)

    # Engineer high-leverage synergy features
    sat = df.get("satisfaction_rating", 3)
    calls = df.get("customer_service_calls", 1)
    unres = df.get("unresolved_complaints", 0)

    df["dissatisfaction_severity"] = (5 - sat) * (calls + 1)
    df["service_friction_index"] = (calls * 2.0) + (unres * 3.5) - (sat * 1.5)

    if "churn" in df.columns:
        df["churn"] = df["churn"].astype(int)

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

def get_train_test_data(csv_path=None, test_size=0.2, random_state=42, sample_size=None):
    df = load_and_clean_data(csv_path=csv_path, sample_size=sample_size, random_state=random_state)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df["churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Dataset split: Train = {X_train.shape[0]:,} samples, Test = {X_test.shape[0]:,} samples")
    print(f"Train Churn Rate: {y_train.mean():.2%}, Test Churn Rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test
