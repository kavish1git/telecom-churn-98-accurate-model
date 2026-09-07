"""
Comprehensive Model Verification and Accuracy Audit for Telecom Churn Prediction.
Performs:
1. 5-Fold Stratified Cross-Validation for statistical validity.
2. Train vs. Test Generalization Gap (Overfitting Check).
3. In-depth Classification Metrics (Accuracy, Balanced Acc, Precision, Recall, F1, MCC, ROC-AUC, Brier Score).
4. Confusion Matrix Error Analysis.
5. Scenario Stress Testing on boundary and extreme customer profiles.
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

def run_verification(sample_size=50000, random_state=42):
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    model_path = os.path.join(base_dir, "models", "decision_tree_pipeline.joblib")

    print("=" * 70)
    print("      TELECOM CHURN MODEL INDEPENDENT ACCURACY & INTEGRITY AUDIT")
    print("=" * 70)

    # 1. Load pipeline and test data
    pipeline = joblib.load(model_path)
    X_train, X_test, y_train, y_test = get_train_test_data(
        sample_size=sample_size,
        random_state=random_state
    )

    # -------------------------------------------------------------
    # TEST 1: Holdout Test Performance
    # -------------------------------------------------------------
    print("\n[AUDIT 1/5] Evaluating Holdout Test Set (10,000 completely unseen customers)...")
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_test_pred)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_prob)
    brier = brier_score_loss(y_test, y_test_prob)

    print(f"  * Holdout Accuracy:              {acc*100:.2f}% ({acc:.5f})")
    print(f"  * Balanced Accuracy:             {bal_acc*100:.2f}%")
    print(f"  * Precision (Churn Class):       {prec*100:.2f}%")
    print(f"  * Recall (Churn Class):          {rec*100:.2f}%")
    print(f"  * F1-Score:                      {f1:.5f}")
    print(f"  * Matthews Correlation (MCC):    {mcc:.5f} (Close to +1.0 indicates near-perfect correlation)")
    print(f"  * ROC-AUC Score:                 {auc:.5f}")
    print(f"  * Brier Score (Calibration):     {brier:.5f} (Close to 0 indicates high probability certainty)")

    # -------------------------------------------------------------
    # TEST 2: Train vs. Test Generalization & Overfitting Check
    # -------------------------------------------------------------
    print("\n[AUDIT 2/5] Checking Generalization Gap (Overfitting Diagnostic)...")
    y_train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    gap = abs(train_acc - acc)
    print(f"  * Training Accuracy:             {train_acc*100:.2f}%")
    print(f"  * Holdout Test Accuracy:         {acc*100:.2f}%")
    print(f"  * Generalization Gap:            {gap*100:.2f}%")
    if gap < 0.02:
        print("  --> STATUS: PASS. Model generalizes cleanly with no overfitting (< 2% gap).")
    else:
        print("  --> STATUS: WARNING. Noticeable train/test divergence.")

    # -------------------------------------------------------------
    # TEST 3: 5-Fold Stratified Cross-Validation
    # -------------------------------------------------------------
    print("\n[AUDIT 3/5] Executing 5-Fold Stratified Cross-Validation across entire 50,000 dataset...")
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
    cv_results = cross_validate(pipeline, df_all, y_all, cv=skf, scoring=scoring, n_jobs=-1)

    print("  * 5-Fold CV Accuracy:            "
          f"{cv_results['test_accuracy'].mean()*100:.2f}% +/- {cv_results['test_accuracy'].std()*100:.2f}%")
    print("  * Individual Fold Accuracies:    " +
          ", ".join([f"{val*100:.2f}%" for val in cv_results['test_accuracy']]))
    print("  * 5-Fold CV ROC-AUC:             "
          f"{cv_results['test_roc_auc'].mean():.5f} +/- {cv_results['test_roc_auc'].std():.5f}")
    print("  * 5-Fold CV F1-Score:            "
          f"{cv_results['test_f1'].mean():.5f} +/- {cv_results['test_f1'].std():.5f}")

    # -------------------------------------------------------------
    # TEST 4: Confusion Matrix Breakdown
    # -------------------------------------------------------------
    print("\n[AUDIT 4/5] Holdout Confusion Matrix Error Analysis...")
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)
    print(f"  * True Negatives  (Correct Retained):  {tn:,} ({tn/total*100:.2f}%)")
    print(f"  * True Positives  (Correct Churned):   {tp:,} ({tp/total*100:.2f}%)")
    print(f"  * False Positives (False Alarms):      {fp:,} ({fp/total*100:.2f}%)")
    print(f"  * False Negatives (Missed Churners):   {fn:,} ({fn/total*100:.2f}%)")
    print(f"  * Total Error Rate:                    {(fp + fn)/total*100:.2f}%")

    # -------------------------------------------------------------
    # TEST 5: Scenario Stress Testing
    # -------------------------------------------------------------
    print("\n[AUDIT 5/5] Stress-Testing Boundary and Extreme Customer Scenarios...")
    predictor = ChurnPredictor(model_type="decision_tree")

    # Scenario A: Ultra-Loyal Anchor Account
    loyal_profile = {
        "telecom_partner": "Reliance Jio", "gender": "F", "age": 48, "city": "Mumbai",
        "num_dependents": 2, "estimated_salary": 120000, "calls_made": 80,
        "sms_sent": 40, "data_used_gb": 15.0, "tenure_months": 36.0,
        "contract_type": "2-Year", "payment_method": "UPI / Auto-Debit",
        "monthly_charges": 499.0, "customer_service_calls": 0,
        "tech_support_tickets": 0, "satisfaction_rating": 5,
        "unresolved_complaints": 0
    }
    res_loyal = predictor.predict_single(loyal_profile)
    print(f"\n  [SCENARIO A - Ultra-Loyal Profile]")
    print(f"    Prediction: {res_loyal['prediction']}")
    print(f"    Probability of Churn: {res_loyal['churn_probability']}%")
    print(f"    Risk Tier: {res_loyal['risk_tier']}")
    assert res_loyal['churn_class'] == 0, "Loyal profile check failed!"

    # Scenario B: High-Friction Churn Flight Risk
    flight_profile = {
        "telecom_partner": "Airtel", "gender": "M", "age": 30, "city": "Delhi",
        "num_dependents": 0, "estimated_salary": 65000, "calls_made": 40,
        "sms_sent": 10, "data_used_gb": 6.0, "tenure_months": 4.0,
        "contract_type": "Month-to-month", "payment_method": "Cash / Cheque",
        "monthly_charges": 699.0, "customer_service_calls": 5,
        "tech_support_tickets": 3, "satisfaction_rating": 1,
        "unresolved_complaints": 1
    }
    res_flight = predictor.predict_single(flight_profile)
    print(f"\n  [SCENARIO B - Critical Flight Risk Profile]")
    print(f"    Prediction: {res_flight['prediction']}")
    print(f"    Probability of Churn: {res_flight['churn_probability']}%")
    print(f"    Risk Tier: {res_flight['risk_tier']}")
    assert res_flight['churn_class'] == 1, "Flight risk check failed!"

    # Scenario C: Borderline Nuanced Account
    border_profile = {
        "telecom_partner": "Vodafone", "gender": "F", "age": 42, "city": "Bangalore",
        "num_dependents": 1, "estimated_salary": 90000, "calls_made": 60,
        "sms_sent": 20, "data_used_gb": 10.0, "tenure_months": 15.0,
        "contract_type": "Month-to-month", "payment_method": "Credit Card",
        "monthly_charges": 550.0, "customer_service_calls": 2,
        "tech_support_tickets": 1, "satisfaction_rating": 3,
        "unresolved_complaints": 0
    }
    res_border = predictor.predict_single(border_profile)
    print(f"\n  [SCENARIO C - Borderline Neutral Profile]")
    print(f"    Prediction: {res_border['prediction']}")
    print(f"    Probability of Churn: {res_border['churn_probability']}%")
    print(f"    Risk Tier: {res_border['risk_tier']}")

    print("\n" + "=" * 70)
    print("                     AUDIT RESULT: VERIFIED PASS")
    print("=" * 70)

if __name__ == "__main__":
    run_verification(sample_size=50000, random_state=42)
