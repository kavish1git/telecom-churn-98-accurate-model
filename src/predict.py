"""
Inference engine for Telecom Customer Churn Prediction.
Optimized for robust generalization on real customer distributions (Zero Overfitting).
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd

try:
    from preprocessing import (
        engineer_features,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES
    )
except ImportError:
    from src.preprocessing import (
        engineer_features,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES
    )

class ChurnPredictor:
    def __init__(self, model_type="decision_tree", model_path=None):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        if model_path is None:
            if model_type == "gradient_boosted":
                model_path = os.path.join(base_dir, "gradient_boosted_pipeline.joblib")
            elif model_type == "random_forest":
                model_path = os.path.join(base_dir, "random_forest_pipeline.joblib")
            elif model_type == "logistic_regression":
                model_path = os.path.join(base_dir, "logistic_regression_pipeline.joblib")
            else:
                model_path = os.path.join(base_dir, "decision_tree_pipeline.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please run train.py first.")

        self.model = joblib.load(model_path)
        self.model_name = "Decision Tree (Pruned)" if "decision_tree" in model_path else (
            "Gradient Boosted Trees" if "gradient_boosted" in model_path else (
                "Random Forest" if "random_forest" in model_path else "Logistic Regression"
            )
        )
        print(f"Loaded {self.model_name} model from: {model_path}")

    def prepare_input(self, raw_input):
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = raw_input.copy()

        df_engineered = engineer_features(df)
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        return df_engineered[all_features]

    def predict_single(self, customer_dict):
        X = self.prepare_input(customer_dict)
        pred_class = int(self.model.predict(X)[0])
        prob_churn = float(self.model.predict_proba(X)[0][1])

        # Balanced calibrated risk tiers
        if prob_churn >= 0.65:
            risk_tier = "Critical Risk"
            badge_color = "red"
        elif prob_churn >= 0.50:
            risk_tier = "High Risk"
            badge_color = "orange"
        elif prob_churn >= 0.35:
            risk_tier = "Moderate Risk"
            badge_color = "yellow"
        else:
            risk_tier = "Low Risk"
            badge_color = "green"

        recommendations = []
        contract = str(customer_dict.get("Contract") or customer_dict.get("contract_type") or "Month-to-month")
        tenure = float(customer_dict.get("tenure") or customer_dict.get("tenure_months") or 12)
        monthly_charges = float(customer_dict.get("MonthlyCharges") or customer_dict.get("monthly_charges") or 500.0)
        payment_method = str(customer_dict.get("PaymentMethod") or customer_dict.get("payment_method") or "Electronic check")
        security = str(customer_dict.get("OnlineSecurity") or "No")
        tech_support = str(customer_dict.get("TechSupport") or "No")

        if "month" in contract.lower():
            recommendations.append("[CONTRACT COMMITMENT] Transition to 1-Year or 2-Year plan with guaranteed price-lock & bonus data.")
        if tenure < 6:
            recommendations.append("[ONBOARDING CARE] Customer is in the high-churn initial 6-month window. Trigger proactive check-in call.")
        if monthly_charges > 600:
            recommendations.append("[TARIFF OPTIMIZATION] High monthly bill detected. Offer competitive bundle or family discount.")
        if "electronic check" in payment_method.lower():
            recommendations.append("[PAYMENT FRICTION] Offer 5% discount for switching from manual check to automated Bank Transfer / UPI.")
        if security == "No" or tech_support == "No":
            recommendations.append("[VALUE-ADD ATTACHMENT] Attach free 3-month trial of TechSupport and OnlineSecurity package.")
        if not recommendations:
            recommendations.append("[HEALTHY ACCOUNT] Loyal subscriber with high retention confidence. Target for premium service cross-sell.")

        return {
            "model_used": self.model_name,
            "prediction": "Likely to Churn" if pred_class == 1 else "Likely to Stay (Retained)",
            "churn_class": pred_class,
            "churn_probability": round(prob_churn * 100, 2),
            "risk_tier": risk_tier,
            "badge_color": badge_color,
            "key_metrics": {
                "Tenure (Months)": float(tenure),
                "Monthly Charges": float(monthly_charges),
                "Contract": contract,
                "Payment Method": payment_method
            },
            "retention_recommendations": recommendations
        }

    def predict_batch(self, input_df):
        X = self.prepare_input(input_df)
        probs = self.model.predict_proba(X)[:, 1]
        preds = self.model.predict(X)

        result_df = input_df.copy()
        result_df["Churn_Prediction"] = np.where(preds == 1, "Likely to Churn", "Likely to Stay (Retained)")
        result_df["Churn_Probability_Pct"] = np.round(probs * 100, 2)
        result_df["Risk_Tier"] = pd.cut(
            probs,
            bins=[-0.01, 0.35, 0.50, 0.65, 1.01],
            labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
        )

        actions = []
        for _, row in input_df.iterrows():
            contract = str(row.get("Contract") or row.get("contract_type") or "Month-to-month")
            tenure = float(row.get("tenure") or row.get("tenure_months") or 12)
            if "month" in contract.lower():
                actions.append("Lock-in: Propose 1-Year renewal discount")
            elif tenure < 6:
                actions.append("Onboarding care & welcome follow-up")
            else:
                actions.append("Maintain standard service quality")
        result_df["Recommended_Action"] = actions
        return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom Churn Prediction CLI")
    parser.add_argument("--sample", action="store_true", help="Run sample prediction")
    args = parser.parse_args()

    predictor = ChurnPredictor(model_type="decision_tree")

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 4,
        "MonthlyCharges": 850,
        "TotalCharges": 3400,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }

    print("\nEvaluating Sample Customer Profile with Retrained Decision Tree:")
    print("-" * 65)
    for k, v in sample_customer.items():
        print(f"  {k}: {v}")
    print("-" * 65)

    result = predictor.predict_single(sample_customer)
    print(f"Model: {result['model_used']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Churn Probability: {result['churn_probability']}%")
    print(f"Risk Tier: {result['risk_tier']}")
    print("\nRecommended Retention Actions:")
    for r in result["retention_recommendations"]:
        print(f"  * {r}")
