"""
Data generation script for Telecom Customer Churn dataset.
Synthesizes a realistic dataset of 7,043 customer accounts incorporating:
- Demographic details (Age, Gender, SeniorCitizen, Partner, Dependents)
- Usage patterns (Tenure, InternetService, StreamingTV, StreamingMovies, MonthlyMinutes, DataUsageGB)
- Billing info (Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, OverduePayments)
- Service feedback (CustomerServiceCalls, TechSupportTickets, SatisfactionRating, UnresolvedComplaints)
"""

import os
import numpy as np
import pandas as pd

def generate_telecom_churn_data(n_samples=7043, random_state=42):
    np.random.seed(random_state)

    # 1. Demographic details
    customer_ids = [f"TELCO-{10000 + i}" for i in range(n_samples)]
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.50, 0.50])
    age = np.clip(np.random.normal(loc=46, scale=17, size=n_samples).astype(int), 18, 85)
    senior_citizen = (age >= 65).astype(int)
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes",
                          np.random.choice(["Yes", "No"], size=n_samples, p=[0.50, 0.50]),
                          np.random.choice(["Yes", "No"], size=n_samples, p=[0.15, 0.85]))

    # 2. Usage Patterns
    # Tenure in months (1 to 72)
    tenure_months = np.random.geometric(p=0.03, size=n_samples)
    tenure_months = np.clip(tenure_months, 1, 72)

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])
    
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], size=n_samples, p=[0.42, 0.48, 0.10])
    
    online_security = np.where(internet_service == "No", "No internet service",
                               np.random.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62]))
    tech_support = np.where(internet_service == "No", "No internet service",
                            np.random.choice(["Yes", "No"], size=n_samples, p=[0.36, 0.64]))
    streaming_tv = np.where(internet_service == "No", "No internet service",
                            np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55]))
    streaming_movies = np.where(internet_service == "No", "No internet service",
                                np.random.choice(["Yes", "No"], size=n_samples, p=[0.46, 0.54]))

    # Monthly minutes call usage
    monthly_minutes = np.where(multiple_lines == "No phone service", 0,
                               np.clip(np.random.normal(loc=520, scale=210, size=n_samples).astype(int), 30, 1800))
    
    # Monthly data usage in GB
    data_usage_gb = np.where(internet_service == "No", 0.0,
                             np.where(internet_service == "DSL",
                                      np.clip(np.random.normal(loc=45, scale=20, size=n_samples), 2.0, 120.0),
                                      np.clip(np.random.normal(loc=135, scale=55, size=n_samples), 10.0, 450.0)))
    data_usage_gb = np.round(data_usage_gb, 1)

    # 3. Billing Info
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.22, 0.22, 0.22]
    )

    # Base monthly charge calculation based on plan features
    base_charge = 20.0
    base_charge += np.where(internet_service == "DSL", 25.0, np.where(internet_service == "Fiber optic", 50.0, 0.0))
    base_charge += np.where(multiple_lines == "Yes", 10.0, 0.0)
    base_charge += np.where(online_security == "Yes", 8.0, 0.0)
    base_charge += np.where(tech_support == "Yes", 8.0, 0.0)
    base_charge += np.where(streaming_tv == "Yes", 12.0, 0.0)
    base_charge += np.where(streaming_movies == "Yes", 12.0, 0.0)
    
    # Add minor noise
    monthly_charges = np.round(base_charge + np.random.uniform(-3.5, 3.5, size=n_samples), 2)
    monthly_charges = np.clip(monthly_charges, 18.5, 120.0)
    
    # Total charges is approximately tenure * monthly charges
    total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 15, size=n_samples), 2)
    total_charges = np.clip(total_charges, 18.5, 9000.0)

    # Overdue payment counts in last 12 months (0 to 4)
    overdue_payments = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.70, 0.16, 0.08, 0.04, 0.02])

    # 4. Service Feedback
    customer_service_calls = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7],
        size=n_samples,
        p=[0.40, 0.28, 0.14, 0.08, 0.05, 0.03, 0.015, 0.005]
    )
    
    tech_support_tickets = np.random.choice(
        [0, 1, 2, 3, 4, 5],
        size=n_samples,
        p=[0.55, 0.25, 0.11, 0.05, 0.03, 0.01]
    )

    # Satisfaction rating (1 to 5)
    satisfaction_latent = 4.2 - (customer_service_calls * 0.45) - (tech_support_tickets * 0.35)
    satisfaction_latent += np.where(tech_support == "Yes", 0.4, 0.0)
    satisfaction_latent += np.where(internet_service == "Fiber optic", -0.2, 0.1)
    satisfaction_latent += np.random.normal(0, 0.5, size=n_samples)
    satisfaction_rating = np.clip(np.round(satisfaction_latent).astype(int), 1, 5)

    # Unresolved complaints (0 = No, 1 = Yes)
    complaint_prob = np.clip(0.04 + 0.12 * (customer_service_calls >= 3) + 0.15 * (satisfaction_rating <= 2), 0, 0.85)
    unresolved_complaints = (np.random.uniform(0, 1, size=n_samples) < complaint_prob).astype(int)

    # 5. Churn Probability Modeling
    log_odds = -2.1
    log_odds += np.where(contract == "Month-to-month", 1.35, 0.0)
    log_odds += np.where(contract == "One year", -0.45, 0.0)
    log_odds += np.where(contract == "Two year", -1.40, 0.0)
    log_odds += -0.045 * (tenure_months - 24)
    log_odds += np.where(internet_service == "Fiber optic", 0.55, 0.0)
    log_odds += np.where(tech_support == "No", 0.40, 0.0)
    log_odds += np.where(online_security == "No", 0.30, 0.0)
    log_odds += 0.018 * (monthly_charges - 65.0)
    log_odds += np.where(payment_method == "Electronic check", 0.50, 0.0)
    log_odds += np.where(paperless_billing == "Yes", 0.25, 0.0)
    log_odds += 0.45 * overdue_payments
    log_odds += 0.42 * customer_service_calls
    log_odds += 0.30 * tech_support_tickets
    log_odds += -0.65 * (satisfaction_rating - 3)
    log_odds += 0.90 * unresolved_complaints
    log_odds += 0.25 * senior_citizen
    log_odds += np.where(partner == "No", 0.20, -0.15)
    log_odds += np.where(dependents == "No", 0.20, -0.15)

    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": gender,
        "Age": age,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "TenureMonths": tenure_months,
        "InternetService": internet_service,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "MonthlyMinutes": monthly_minutes,
        "DataUsageGB": data_usage_gb,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "OverduePayments": overdue_payments,
        "CustomerServiceCalls": customer_service_calls,
        "TechSupportTickets": tech_support_tickets,
        "SatisfactionRating": satisfaction_rating,
        "UnresolvedComplaints": unresolved_complaints,
        "Churn": churn
    })

    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "telecom_churn.csv")
    
    print("Generating telecom customer churn dataset...")
    df = generate_telecom_churn_data(n_samples=7043, random_state=42)
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset successfully created at: {csv_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    print("\nFeature preview:")
    print(df.head(3))
