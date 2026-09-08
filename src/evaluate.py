"""
Comprehensive Model Evaluation Script for Telecom Customer Churn Prediction.
Evaluates:
1. Decision Tree Classifier (Pruned, interpretable baseline)
2. Gradient Boosted Decision Trees (Ensemble boosting)
3. Random Forest (Ensemble bagging)
4. Logistic Regression (Linear regularized baseline)

Generates:
- reports/model_comparison_metrics.csv
- reports/confusion_matrices.png & public/confusion_matrices.png
- reports/roc_curves.png & public/roc_curves.png
- reports/feature_importance.png & public/feature_importance.png
- reports/decision_tree_plot.png & public/decision_tree_plot.png
"""

import os
import shutil
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)
from sklearn.tree import plot_tree

from preprocessing import get_train_test_data, CATEGORICAL_FEATURES
from train import get_feature_names

def evaluate_all_models(random_state=42):
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    public_dir = os.path.join(base_dir, "public")
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(public_dir, exist_ok=True)

    print("Loading holdout test data (2,000 unseen customer accounts)...")
    X_train, X_test, y_train, y_test = get_train_test_data(
        test_size=0.2,
        random_state=random_state
    )

    models = {
        "Decision Tree (Pruned)": joblib.load(os.path.join(models_dir, "decision_tree_pipeline.joblib")),
        "Gradient Boosted Trees": joblib.load(os.path.join(models_dir, "gradient_boosted_pipeline.joblib")),
        "Random Forest": joblib.load(os.path.join(models_dir, "random_forest_pipeline.joblib")),
        "Logistic Regression": joblib.load(os.path.join(models_dir, "logistic_regression_pipeline.joblib")),
    }

    metrics_list = []
    preds_dict = {}
    probs_dict = {}

    print("\n" + "=" * 70)
    print("EVALUATION METRICS SUMMARY (TEST SET: 2,000 SAMPLES)")
    print("=" * 70)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        preds_dict[name] = y_pred
        probs_dict[name] = y_prob

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        metrics_list.append({
            "Model": name,
            "Accuracy": f"{acc*100:.2f}%",
            "Accuracy_Raw": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].to_string(index=False))

    csv_path = os.path.join(reports_dir, "model_comparison_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    # 1. Confusion Matrices
    print("\nGenerating Confusion Matrices plot...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    model_order = ["Decision Tree (Pruned)", "Gradient Boosted Trees", "Random Forest", "Logistic Regression"]
    for i, name in enumerate(model_order):
        cm = confusion_matrix(y_test, preds_dict[name])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if "Decision" in name else ("Greens" if "Gradient" in name else "Purples"),
            cbar=False,
            ax=axes[i],
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"]
        )
        axes[i].set_title(f"{name}\nAcc: {metrics_df.loc[metrics_df['Model']==name, 'Accuracy'].values[0]} | AUC: {metrics_df.loc[metrics_df['Model']==name, 'ROC-AUC'].values[0]}")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200)
    shutil.copyfile(cm_path, os.path.join(public_dir, "confusion_matrices.png"))
    plt.close()
    print(f"Saved confusion matrices -> {cm_path}")

    # 2. ROC Curves
    print("Generating ROC Curves plot...")
    plt.figure(figsize=(9, 6))
    colors = {"Decision Tree (Pruned)": "#1E88E5", "Gradient Boosted Trees": "#43A047", "Random Forest": "#FB8C00", "Logistic Regression": "#8E24AA"}
    for name in model_order:
        fpr, tpr, _ = roc_curve(y_test, probs_dict[name])
        auc_val = roc_auc_score(y_test, probs_dict[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=colors[name], lw=2.2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.5000)")
    plt.title("Receiver Operating Characteristic (ROC Curves) — 2,000 Unseen Accounts", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(reports_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=200)
    shutil.copyfile(roc_path, os.path.join(public_dir, "roc_curves.png"))
    plt.close()
    print(f"Saved ROC curves -> {roc_path}")

    # 3. Feature Importance Plot
    print("Generating Feature Importance plot...")
    dt_pipeline = models["Decision Tree (Pruned)"]
    gb_pipeline = models["Gradient Boosted Trees"]
    fitted_prep = dt_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_prep, CATEGORICAL_FEATURES)
    importances = gb_pipeline.named_steps["classifier"].feature_importances_

    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False).head(12)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_df, palette="Blues_r")
    plt.title("Top Telecom Churn Drivers (Gradient Boosted Tree Importance)", fontsize=13, fontweight="bold")
    plt.xlabel("Feature Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    feat_path = os.path.join(reports_dir, "feature_importance.png")
    plt.savefig(feat_path, dpi=200)
    shutil.copyfile(feat_path, os.path.join(public_dir, "feature_importance.png"))
    plt.close()
    print(f"Saved feature importance plot -> {feat_path}")

    # 4. Decision Tree Structural Plot
    print("Generating Decision Tree Structural plot...")
    dt_clf = dt_pipeline.named_steps["classifier"]
    plt.figure(figsize=(24, 10))
    plot_tree(
        dt_clf,
        max_depth=3,
        feature_names=feature_names,
        class_names=["Retained", "Churned"],
        filled=True,
        rounded=True,
        fontsize=9,
        precision=2
    )
    plt.title("Regularized Auditable Decision Tree Architecture (First 3 Depths)", fontsize=15, fontweight="bold")
    plt.tight_layout()

    tree_path = os.path.join(reports_dir, "decision_tree_plot.png")
    plt.savefig(tree_path, dpi=200)
    shutil.copyfile(tree_path, os.path.join(public_dir, "decision_tree_plot.png"))
    plt.close()
    print(f"Saved tree diagram -> {tree_path}")

    print("\n" + "=" * 70)
    print("ALL EVALUATION REPORTS & VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_all_models()
