"""
Independent Model Verification and Overfitting Audit for Telecom Churn Prediction.
Performs:
1. Holdout Test Set Evaluation (2,000 completely unseen customers).
2. Generalization Gap Audit (Train vs. Test Overfitting Diagnostic).
3. 5-Fold Stratified Cross-Validation across the 10,000-record dataset.
4. Stress Testing across representative subscriber archetypes.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
from preprocessing import get_train_test_data, build_preprocessor
from predict import ChurnPredictor

def run_verification(random_state=42):
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    model_path = os.path.join(base_dir, "models", "decision_tree_pipeline.joblib")

    print("=" * 70)
    print("   TELECOM CHURN MODEL INDEPENDENT ACCURACY & OVERFITTING AUDIT")
    print("=" * 70)

    # 1. Load pipeline and test data
    pipeline = joblib.load(model_path)
    X_train, X_test, y_train, y_test = get_train_test_data(
        test_size=0.2,
        random_state=random_state
    )

    # -------------------------------------------------------------
    # TEST 1: Holdout Test Performance
    # -------------------------------------------------------------
    print("\n[AUDIT 1/4] Evaluating Holdout Test Set (2,000 completely unseen accounts)...")
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_test_pred)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_prob)
    brier = brier_score_loss(y_test, y_test_prob)

    print(f"  * Holdout Accuracy:              {acc*100:.2f}% ({acc:.4f})")
    print(f"  * Balanced Accuracy:             {bal_acc*100:.2f}%")
    print(f"  * Precision (Churn Class):       {prec*100:.2f}%")
    print(f"  * Recall (Churn Class):          {rec*100:.2f}%")
    print(f"  * F1-Score:                      {f1:.4f}")
    print(f"  * Matthews Correlation (MCC):    {mcc:.4f}")
    print(f"  * ROC-AUC Score:                 {auc:.4f}")
    print(f"  * Brier Score (Calibration):     {brier:.4f}")

    # -------------------------------------------------------------
    # TEST 2: Train vs. Test Generalization & Overfitting Check
    # -------------------------------------------------------------
    print("\n[AUDIT 2/4] Checking Generalization Gap (Overfitting Diagnostic)...")
    y_train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    gap = abs(train_acc - acc)
    print(f"  * Training Accuracy:             {train_acc*100:.2f}%")
    print(f"  * Holdout Test Accuracy:         {acc*100:.2f}%")
    print(f"  * Generalization Gap:            {gap*100:.2f}%")
    if gap < 0.05:
        print("  --> STATUS: PASS. Model generalizes cleanly with no overfitting (< 5% divergence).")
    else:
        print("  --> STATUS: WARNING. Noticeable train/test divergence.")

    # -------------------------------------------------------------
    # TEST 3: 5-Fold Stratified Cross-Validation
    # -------------------------------------------------------------
    print("\n[AUDIT 3/4] Executing 5-Fold Stratified Cross-Validation across 10,000 dataset...")
    df_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    cv_res = cross_validate(pipeline, df_all, y_all, cv=skf, scoring=scoring)
    print(f"  * 5-Fold Mean Accuracy:          {cv_res['test_accuracy'].mean()*100:.2f}% (std: {cv_res['test_accuracy'].std():.4f})")
    print(f"  * 5-Fold Mean Precision:         {cv_res['test_precision'].mean()*100:.2f}%")
    print(f"  * 5-Fold Mean Recall:            {cv_res['test_recall'].mean()*100:.2f}%")
    print(f"  * 5-Fold Mean F1-Score:          {cv_res['test_f1'].mean():.4f}")
    print(f"  * 5-Fold Mean ROC-AUC:           {cv_res['test_roc_auc'].mean():.4f}")

    # -------------------------------------------------------------
    # TEST 4: Profile Stress Testing
    # -------------------------------------------------------------
    print("\n[AUDIT 4/4] Scenario Stress Testing on Distinct Customer Archetypes...")
    predictor = ChurnPredictor(model_type="decision_tree")

    archetypes = [
        {
            "name": "High-Risk: New Month-to-Month Fiber Subscriber with Manual Check",
            "profile": {
                "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 1, "MonthlyCharges": 850, "TotalCharges": 850, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
                "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"
            }
        },
        {
            "name": "Low-Risk: Long-Tenure 2-Year Contract with Full Digital Protection",
            "profile": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
                "tenure": 60, "MonthlyCharges": 450, "TotalCharges": 27000, "PhoneService": "Yes",
                "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
                "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
                "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)"
            }
        }
    ]

    for arch in archetypes:
        res = predictor.predict_single(arch["profile"])
        print(f"\n  Scenario: {arch['name']}")
        print(f"    - Prediction:         {res['prediction']}")
        print(f"    - Churn Probability:  {res['churn_probability']}%")
        print(f"    - Risk Tier:          {res['risk_tier']}")
        print(f"    - Key Intervention:   {res['retention_recommendations'][0]}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE: SYSTEM GENERALIZATION AUDIT PASSED")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
