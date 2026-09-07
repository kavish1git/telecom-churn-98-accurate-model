"""
Exploratory Data Analysis and Cleaning for the User's Telecom Churn Dataset.
Source: C:/Users/Kavish/Downloadsc/telecom_churn.csv/telecom_churn.csv
"""

import os
import shutil
import pandas as pd
import numpy as np

def explore_and_copy_dataset():
    src_path = r"C:\Users\Kavish\Downloadsc\telecom_churn.csv\telecom_churn.csv"
    dest_dir = r"C:\Users\Kavish\.gemini\antigravity\scratch\telecom_churn_prediction\data"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "telecom_churn.csv")

    print(f"Copying dataset from:\n  {src_path}\nto:\n  {dest_path}")
    shutil.copyfile(src_path, dest_path)
    print("Copy completed successfully.")

    print("\nLoading dataset into pandas...")
    df = pd.read_csv(dest_path)
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Churn Class Distribution ---")
    print(df['churn'].value_counts(dropna=False))
    print(df['churn'].value_counts(normalize=True).map("{:.2%}".format))

    print("\n--- Data Types & Summary ---")
    print(df.info())

    print("\n--- Numerical Columns Description ---")
    num_cols = ['age', 'num_dependents', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used']
    print(df[num_cols].describe().T)

    print("\n--- Check for Anomalies (e.g. Negative values) ---")
    for col in ['calls_made', 'sms_sent', 'data_used']:
        neg_count = (df[col] < 0).sum()
        print(f"{col}: {neg_count} negative values ({neg_count/len(df):.2%})")

    print("\n--- Categorical Columns Unique Values ---")
    for col in ['telecom_partner', 'gender', 'state', 'city']:
        n_uniq = df[col].nunique()
        top_vals = df[col].value_counts().head(5).to_dict()
        print(f"{col}: {n_uniq} unique values, Top 5: {top_vals}")

    print("\n--- Date of Registration ---")
    print("Min date:", df['date_of_registration'].min(), "Max date:", df['date_of_registration'].max())

if __name__ == "__main__":
    explore_and_copy_dataset()
