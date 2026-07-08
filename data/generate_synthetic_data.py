import pandas as pd
import numpy as np
import random

# Setting a seed so results are consistent every time we run this
np.random.seed(42)
random.seed(42)

NUM_USERS = 800

def generate_users(num_users):
    data = []

    for i in range(num_users):
        user_id = f"U{1000 + i}"

        # Mobile recharge how many times per month they top up
        recharge_frequency = np.random.randint(1, 15)

        # Utility payment regularity percentage of bills paid on time in last 12 months
        utility_regularity = round(np.random.uniform(40, 100), 1)

        # E-commerce transaction - number of transactions per month
        ecommerce_frequency = np.random.randint(0, 20)

        # Avg e-commerce transaction amount in indian rupee
        ecommerce_avg_amount = round(np.random.uniform(100, 5000), 2)

        # Savings account balance stability - rough signal
        avg_balance = round(np.random.uniform(500, 50000), 2)

        # --- Deriving a raw weighted score from the signals first ---
        # This raw score is just an internal number before we scale it to 300-900
        raw_score = (
            (utility_regularity * 0.45) +
            (recharge_frequency * 3.5) +
            (ecommerce_frequency * 2.2) +
            (avg_balance / 1000) * 1.8
        )

        # Adding a bit of noise so it's not a perfectly clean formula
        raw_score += np.random.normal(0, 8)

        # --- Scaling raw_score into the real-world 300-900 credit score range ---
        # raw_score realistically falls somewhere between roughly 0 and 160
        # We map that onto 300 (worst) to 900 (best)
        min_raw=0
        max_raw =160
        raw_score_clipped = np.clip(raw_score, min_raw, max_raw)

        credit_score = 300 + ((raw_score_clipped - min_raw) / (max_raw - min_raw)) * (900 - 300)
        credit_score = int(round(credit_score))

        # Mapping the credit score to real-world CIBILs type
        if credit_score >= 750:
            risk_bucket = "Excellent"
        elif credit_score >= 700:
            risk_bucket = "Good"
        elif credit_score >= 650:
            risk_bucket = "Fair"
        else:
            risk_bucket = "Poor"

        data.append({
            "user_id": user_id,
            "recharge_frequency": recharge_frequency,
            "utility_regularity": utility_regularity,
            "ecommerce_frequency": ecommerce_frequency,
            "ecommerce_avg_amount": ecommerce_avg_amount,
            "avg_balance": avg_balance,
            "credit_score": credit_score,
            "risk_bucket": risk_bucket
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    
    df = generate_users(NUM_USERS)

    df.to_csv("data/synthetic_users.csv", index=False)

    print(f"Generated {len(df)} synthetic user records.")

    print("***********RISK BUCKET**************")

    print(df["risk_bucket"].value_counts())

    print(df[["user_id", "credit_score", "risk_bucket"]].head(10))

