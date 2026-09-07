# Telecom Customer Churn Prediction System

An end-to-end Machine Learning solution designed for the telecom industry to predict customer churn risk, analyze key behavioral drivers, and provide actionable retention interventions using **Decision Tree Classifiers** and ensemble benchmarks.

---

## 📌 Project Overview

Customer churn is a multi-million dollar challenge in telecommunications. Acquiring a replacement customer costs 5 to 25 times more than retaining an existing one. This system equips customer retention teams with predictive analytics and interpretable AI to intervene before subscribers migrate to competitors.

### 4 Pillars of Analysis
1. **Demographic Details**: Age, Gender, Number of Dependents, State, City.
2. **Usage Patterns**: Calls Made, SMS Sent, Monthly Data Used (MB / GB), Calls per Month.
3. **Billing & Account Details**: Estimated Salary, Salary per Dependent, Telecom Partner (Reliance Jio, Airtel, Vodafone, BSNL).
4. **Lifecycle & Tenure**: Registration Date, Tenure in Months, Account Maturity.

---

## 🛠 Project Structure

```
telecom_churn_prediction/
│
├── data/
│   ├── telecom_churn.csv        # 243,553 customer records
│   └── generate_data.py         # Synthetic generator utility
│
├── src/
│   ├── explore_data.py          # Data exploration & anomaly diagnosis
│   ├── preprocessing.py         # Pipeline transforms, scaling & encoding
│   ├── train.py                 # Trains & tunes Decision Tree, RF & Logistic Regression
│   ├── evaluate.py              # Evaluates metrics, confusion matrices, ROC curves
│   └── predict.py               # CLI & API inference engine
│
├── models/
│   ├── decision_tree_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   ├── logistic_regression_pipeline.joblib
│   └── model_metadata.json
│
├── reports/
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── decision_tree_plot.png
│   ├── feature_importance.png
│   ├── decision_tree_rules.txt
│   └── model_comparison_metrics.csv
│
├── app.py                       # Interactive Streamlit Web Application
├── requirements.txt             # Project dependencies
└── README.md                    # Documentation
```

---

## 🚀 Getting Started

### 1. Run Data Preprocessing and Training
```bash
python src/train.py
```
This trains the Decision Tree (with hyperparameter tuning via `GridSearchCV`), Random Forest, and Logistic Regression models, extracting decision rules and saving the trained pipelines.

### 2. Run Comprehensive Evaluation & Plot Generation
```bash
python src/evaluate.py
```
This produces:
- Comparison metrics table (`reports/model_comparison_metrics.csv`)
- Confusion matrices (`reports/confusion_matrices.png`)
- ROC Curves (`reports/roc_curves.png`)
- Decision tree visualization (`reports/decision_tree_plot.png`)
- Feature importance analysis (`reports/feature_importance.png`)

### 3. Run Inference via CLI
```bash
python src/predict.py --sample
```

### 4. Launch the Interactive Streamlit Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to test single customer risk profiles, explore the Decision Tree rules, or score full CSV files in bulk!
