"""
Inference engine for Indian Telecom Experience & Survey Churn Prediction (Jio, Airtel, Vi).
"""

import os
import joblib
import pandas as pd
import numpy as np

RELIABILITY_SCALE = {
    "Very Reliable": 5, "Reliable": 4, "Neutral": 3, "Unreliable": 2, "Very Unreliable": 1,
    5: 5, 4: 4, 3: 3, 2: 2, 1: 1
}

DROP_SCALE = {
    "None": 0, "1-3": 2, "4-7": 5, "8-15": 11, "More than 15": 18, "Not Stated": 2
}

NUM_COLS = [
    'reliability_streaming', 'reliability_video_calls', 'reliability_browsing',
    'reliability_gaming', 'call_drops_count', 'quality_friction_index', 'value_friction_index'
]
CAT_COLS = [
    'provider', 'network_type', 'plan_type', 'age_group', 'city',
    'monthly_bill_range', 'monthly_data_range', 'call_drops_weekly'
]

class SurveyChurnPredictor:
    def __init__(self, model_path=None):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        if model_path is None:
            model_path = os.path.join(base_dir, "survey_churn_pipeline.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Survey model pipeline not found at {model_path}")
        self.model = joblib.load(model_path)
        print(f"Loaded Indian Telecom Survey Model from: {model_path}")

    def prepare_input(self, raw_input):
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        else:
            df = raw_input.copy()

        # Fill defaults
        if "provider" not in df: df["provider"] = "Jio"
        if "network_type" not in df: df["network_type"] = "5G"
        if "plan_type" not in df: df["plan_type"] = "Prepaid (monthly recharge)"
        if "age_group" not in df: df["age_group"] = "18–24"
        if "city" not in df: df["city"] = "Ahmedabad"
        if "monthly_bill_range" not in df: df["monthly_bill_range"] = "200–499"
        if "monthly_data_range" not in df: df["monthly_data_range"] = "21–50 GB"
        if "call_drops_weekly" not in df: df["call_drops_weekly"] = "1-3"

        # Map reliabilities to numeric
        for col in ["reliability_streaming", "reliability_video_calls", "reliability_browsing", "reliability_gaming"]:
            val = df.get(col, 4)
            if hasattr(val, "iloc"):
                df[col] = val.map(lambda v: RELIABILITY_SCALE.get(v, 4) if not isinstance(v, (int, float)) else int(v)).fillna(4)
            else:
                df[col] = RELIABILITY_SCALE.get(val, 4)

        # Call drops count
        df["call_drops_count"] = df["call_drops_weekly"].map(lambda d: DROP_SCALE.get(str(d).strip(), 2)).fillna(2)

        # Engineered features
        drops = df["call_drops_count"]
        calls = df["reliability_video_calls"]
        streaming = df["reliability_streaming"]
        browsing = df["reliability_browsing"]
        bill = df["monthly_bill_range"]

        bill_burden = np.where(bill == '1,500+', 1.5, np.where(bill == '800–1,499', 0.8, 0.0))
        df["quality_friction_index"] = (drops * 0.4) + (5 - calls) * 1.2 + (5 - streaming) * 0.8
        df["value_friction_index"] = bill_burden * 1.5 + (5 - browsing) * 0.6

        return df[NUM_COLS + CAT_COLS]

    def predict_single(self, survey_dict):
        X = self.prepare_input(survey_dict)
        pred_class = int(self.model.predict(X)[0])
        prob_churn = float(self.model.predict_proba(X)[0][1])

        provider = str(survey_dict.get("provider", "Jio"))
        network = str(survey_dict.get("network_type", "5G"))
        drops = str(survey_dict.get("call_drops_weekly", "1-3"))
        plan = str(survey_dict.get("plan_type", "Prepaid (monthly recharge)"))
        bill = str(survey_dict.get("monthly_bill_range", "200–499"))

        if prob_churn >= 0.65:
            tier = "Critical Risk"
            badge_color = "red"
        elif prob_churn >= 0.50:
            tier = "High Risk"
            badge_color = "orange"
        elif prob_churn >= 0.35:
            tier = "Moderate Risk"
            badge_color = "yellow"
        else:
            tier = "Low Risk"
            badge_color = "green"

        recommendations = []
        if "more than" in drops.lower() or "8-15" in drops:
            recommendations.append(f"[NETWORK STABILITY] Frequent call drops reported ({drops} weekly). Trigger automated cell tower RF optimization ticket.")
        if "month" in plan.lower():
            recommendations.append(f"[PLAN LOCK-IN] Customer on prepaid monthly recharge. Offer Rs. 50 cashback on 3-Month or 1-Year long-term plan.")
        if "1,500" in bill or "800" in bill:
            recommendations.append(f"[PRICE PROTECTION] High monthly bill ({bill}). Offer bundled OTT (Hotstar/Prime) or family plan tariff relief.")
        if network == "4G":
            recommendations.append(f"[5G UPGRADE] Offer free 5G SIM upgrade with complimentary 50GB high-speed booster data.")
        if not recommendations:
            recommendations.append(f"[HEALTHY SUBSCRIBER] Loyal {provider} user with satisfactory network experience. Target for 5G broadband / device finance offer.")

        return {
            "provider": provider,
            "network_type": network,
            "prediction": "Likely to Switch in 6 Months (At-Risk)" if pred_class == 1 else "Likely to Stay (Loyal Subscriber)",
            "churn_class": pred_class,
            "churn_probability": round(prob_churn * 100, 2),
            "risk_tier": tier,
            "badge_color": badge_color,
            "recommendations": recommendations
        }

    def predict_batch(self, input_df):
        X = self.prepare_input(input_df)
        probs = self.model.predict_proba(X)[:, 1]
        preds = self.model.predict(X)

        result_df = input_df.copy()
        result_df["Switch_Prediction_6Mo"] = np.where(preds == 1, "Likely to Switch", "Likely to Stay")
        result_df["Switch_Probability_Pct"] = np.round(probs * 100, 2)
        result_df["Risk_Tier"] = pd.cut(
            probs,
            bins=[-0.01, 0.35, 0.50, 0.65, 1.01],
            labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
        )

        actions = []
        for _, row in input_df.iterrows():
            drops = str(row.get("call_drops_weekly", "1-3"))
            plan = str(row.get("plan_type", "Prepaid"))
            if "more than" in drops.lower() or "8-15" in drops:
                actions.append("Network Priority: Fix frequent call drops & tower handover")
            elif "month" in plan.lower():
                actions.append("Tariff Lock-in: Propose 3-Month / 1-Year renewal discount")
            else:
                actions.append("Standard Engagement: Maintain quality of service")
        result_df["Retention_Action"] = actions
        return result_df
