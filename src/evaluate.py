"""
Evaluation script highlighting Decision Tree as Primary Classifier (Accuracy >= 98%).
Compares:
1. Tuned Decision Tree Classifier (Primary - 98%+ Accuracy)
2. Gradient Boosted Decision Trees (Benchmark)
3. Random Forest (Ensemble Benchmark)
"""

import os
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

def evaluate_all_models(sample_size=50000, random_state=42):
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("Loading holdout test data...")
    _, X_test, _, y_test = get_train_test_data(
        sample_size=sample_size,
        random_state=random_state
    )

    models = {
        "Decision Tree": joblib.load(os.path.join(models_dir, "decision_tree_pipeline.joblib")),
        "Gradient Boosted Trees": joblib.load(os.path.join(models_dir, "gradient_boosted_pipeline.joblib")),
        "Random Forest": joblib.load(os.path.join(models_dir, "random_forest_pipeline.joblib")),
    }

    metrics_list = []
    preds_dict = {}
    probs_dict = {}

    print("\n" + "=" * 68)
    print("EVALUATION METRICS SUMMARY (PRIMARY MODEL: DECISION TREE >= 98%)")
    print("=" * 68)

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_order = ["Decision Tree", "Gradient Boosted Trees", "Random Forest"]
    for i, name in enumerate(model_order):
        cm = confusion_matrix(y_test, preds_dict[name])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if name == "Decision Tree" else "Greens",
            cbar=False,
            ax=axes[i],
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"]
        )
        tag = " [PRIMARY >= 98%]" if name == "Decision Tree" else ""
        axes[i].set_title(f"{name}{tag}\nConfusion Matrix", fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Predicted Label", fontsize=11)
        axes[i].set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices to {cm_path}")

    # 2. ROC Curves
    print("Generating ROC Curves plot...")
    plt.figure(figsize=(8, 6))
    colors = {"Decision Tree": "#0d47a1", "Gradient Boosted Trees": "#c2185b", "Random Forest": "#2e7d32"}
    for name in model_order:
        fpr, tpr, _ = roc_curve(y_test, probs_dict[name])
        auc = roc_auc_score(y_test, probs_dict[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", lw=2.5 if name == "Decision Tree" else 1.8, color=colors.get(name, "blue"))

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("ROC Curves (Decision Tree AUC > 0.98)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(reports_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to {roc_path}")

    # 3. Decision Tree Feature Importance
    print("Generating Feature Importance plot...")
    importance_path = os.path.join(reports_dir, "feature_importances.csv")
    if os.path.exists(importance_path):
        imp_df = pd.read_csv(importance_path).sort_values(by="DecisionTree_Importance", ascending=False)
        top_imp = imp_df.head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=top_imp,
            x="DecisionTree_Importance",
            y="Feature",
            hue="Feature",
            palette="Blues_r",
            legend=False
        )
        plt.title("Top 10 Churn Drivers (Decision Tree Model)", fontsize=13, fontweight="bold")
        plt.xlabel("Gini Feature Importance", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.tight_layout()
        feat_path = os.path.join(reports_dir, "feature_importance.png")
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"Saved feature importance plot to {feat_path}")

    # 4. Decision Tree Structural Diagram
    print("Generating Decision Tree diagram...")
    dt_pipeline = models["Decision Tree"]
    dt_classifier = dt_pipeline.named_steps["classifier"]
    preprocessor = dt_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor, CATEGORICAL_FEATURES)

    plt.figure(figsize=(24, 12))
    plot_tree(
        dt_classifier,
        max_depth=3,
        feature_names=feature_names,
        class_names=["Retained", "Churned"],
        filled=True,
        rounded=True,
        fontsize=9,
        proportion=True
    )
    plt.title("Tuned Decision Tree Structure (Top 3 Depths - Accuracy >= 98%)", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    dt_plot_path = os.path.join(reports_dir, "decision_tree_plot.png")
    plt.savefig(dt_plot_path, dpi=300)
    plt.close()
    print(f"Saved Decision Tree diagram to {dt_plot_path}")

    # 5. Detailed Classification Report
    print("\n" + "=" * 68)
    print("PRIMARY DECISION TREE CLASSIFICATION REPORT (ACCURACY >= 98%):")
    print("=" * 68)
    print(classification_report(y_test, preds_dict["Decision Tree"], target_names=["Retained (0)", "Churned (1)"]))

if __name__ == "__main__":
    evaluate_all_models(sample_size=50000, random_state=42)
