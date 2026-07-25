"""
dataset_generator.py
Generates a realistic synthetic telecom customer churn dataset with 5,500 rows.
Includes logical churn patterns based on tenure, contract type, support tickets, 
internet service, monthly charges, and tech support availability.
"""

import numpy as np
import pandas as pd

def generate_telecom_churn_data(num_samples=5500, random_state=42):
    np.random.seed(random_state)
    
    # 1. Base Customer Attributes
    customer_ids = [f"{np.random.randint(1000,9999)}-{chr(65+i%26)}{chr(65+(i*3)%26)}{chr(65+(i*7)%26)}" for i in range(num_samples)]
    gender = np.random.choice(["Female", "Male"], size=num_samples, p=[0.49, 0.51])
    senior_citizen = np.random.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=num_samples, p=[0.48, 0.52])
    
    # Dependents correlated with partner
    dependents = []
    for p in partner:
        if p == "Yes":
            dependents.append(np.random.choice(["Yes", "No"], p=[0.55, 0.45]))
        else:
            dependents.append(np.random.choice(["Yes", "No"], p=[0.10, 0.90]))
    dependents = np.array(dependents)

    # 2. Service Attributes
    # Tenure in months (1 to 72)
    tenure = np.random.exponential(scale=24, size=num_samples).astype(int)
    tenure = np.clip(tenure, 1, 72)
    
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], 
        size=num_samples, 
        p=[0.55, 0.24, 0.21]
    )
    
    internet_service = np.random.choice(
        ["Fiber optic", "DSL", "No"], 
        size=num_samples, 
        p=[0.44, 0.34, 0.22]
    )
    
    phone_service = np.random.choice(["Yes", "No"], size=num_samples, p=[0.90, 0.10])
    
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
    multiple_lines = np.array(multiple_lines)

    # Internet-dependent services
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []

    for net in internet_service:
        if net == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.38, 0.62]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.33, 0.67]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.48, 0.52]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.49, 0.51]))

    paperless_billing = np.random.choice(["Yes", "No"], size=num_samples, p=[0.59, 0.41])
    
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=num_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )

    # 3. Monthly Charges & Total Charges
    base_charge = np.where(internet_service == "No", 20.0, np.where(internet_service == "DSL", 50.0, 75.0))
    addons = (
        (np.array(online_security) == "Yes").astype(int) * 8 +
        (np.array(online_backup) == "Yes").astype(int) * 7 +
        (np.array(device_protection) == "Yes").astype(int) * 7 +
        (np.array(tech_support) == "Yes").astype(int) * 9 +
        (np.array(streaming_tv) == "Yes").astype(int) * 10 +
        (np.array(streaming_movies) == "Yes").astype(int) * 10
    )
    noise_charge = np.random.normal(0, 2.5, size=num_samples)
    monthly_charges = np.round(np.clip(base_charge + addons + noise_charge, 18.25, 118.75), 2)
    
    # Total charges based on tenure and monthly charge with slight variance
    total_charges = np.round(np.clip(monthly_charges * tenure + np.random.normal(0, 15, size=num_samples), 18.25, 8800.0), 2)

    # 4. Support Tickets (0 to 9) - correlated with internet service and tech support
    support_tickets = []
    for net, ts in zip(internet_service, tech_support):
        if net == "No":
            tickets = np.random.poisson(lam=0.5)
        elif ts == "No":
            tickets = np.random.poisson(lam=3.2)
        else:
            tickets = np.random.poisson(lam=1.1)
        support_tickets.append(min(tickets, 9))
    support_tickets = np.array(support_tickets)

    # 5. Realistic Churn Probability Calculation (Logit model)
    # Log-odds coefficients
    logit = (
        -1.2  # base intercept
        + 1.5 * (contract == "Month-to-month")
        - 0.8 * (contract == "One year")
        - 1.6 * (contract == "Two year")
        - 0.04 * tenure
        + 0.9 * (internet_service == "Fiber optic")
        - 0.7 * (np.array(tech_support) == "Yes")
        - 0.5 * (np.array(online_security) == "Yes")
        + 0.35 * (support_tickets >= 3)
        + 0.5 * (support_tickets >= 5)
        + 0.015 * (monthly_charges - 65)
        + 0.45 * (payment_method == "Electronic check")
        + 0.2 * (senior_citizen == 1)
        - 0.3 * (partner == "Yes")
    )
    
    churn_prob = 1 / (1 + np.exp(-logit))
    churn_binary = (np.random.uniform(0, 1, size=num_samples) < churn_prob).astype(int)
    churn_status = np.where(churn_binary == 1, "Yes", "No")

    # Assemble DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SupportTickets": support_tickets,
        "Churn": churn_status
    })

    return df

if __name__ == "__main__":
    df = generate_telecom_churn_data(num_samples=5500, random_state=42)
    output_filename = "telecom_churn_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"Successfully generated synthetic dataset with {len(df)} rows and saved to '{output_filename}'.")
    print(f"Overall Churn Rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")
