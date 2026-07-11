# This module handles the investment advisor side of the project.
# It takes a user's answers to a short risk-assessment quiz, combines that
# with their credit risk bucket (from our scoring model), and suggests
# suitable investment options in plain language.

# --- Step 1: Define the risk-assessment questions ---
# Each question has options, and each option carries a point value.
# More points = more comfortable with risk.

QUESTIONS = [
    {
        "id": "q1",
        "question": "How would you feel if your investment value dropped by 10% in a month?",
        "options": {
            "a": ("I would panic and want to withdraw immediately", 1),
            "b": ("I would feel worried but wait and watch", 2),
            "c": ("I would stay calm, this is normal", 3),
        }
    },
    {
        "id": "q2",
        "question": "How long can you keep this money invested without needing it back?",
        "options": {
            "a": ("Less than 1 year", 1),
            "b": ("1 to 3 years", 2),
            "c": ("More than 3 years", 3),
        }
    },
    {
        "id": "q3",
        "question": "Have you ever invested in mutual funds, stocks, or similar before?",
        "options": {
            "a": ("Never", 1),
            "b": ("A little, I'm still learning", 2),
            "c": ("Yes, I'm fairly comfortable with it", 3),
        }
    },
    {
        "id": "q4",
        "question": "How stable is your monthly income right now?",
        "options": {
            "a": ("Irregular or unpredictable", 1),
            "b": ("Mostly stable, some months vary", 2),
            "c": ("Very stable every month", 3),
        }
    },
    {
        "id": "q5",
        "question": "What is your main goal with this money?",
        "options": {
            "a": ("Keep it safe, growth is secondary", 1),
            "b": ("Balanced - some safety, some growth", 2),
            "c": ("Maximize growth, I can handle ups and downs", 3),
        }
    },
]


def calculate_risk_appetite(answers):
    """
    answers: a dictionary like {"q1": "b", "q2": "c", "q3": "a", "q4": "b", "q5": "c"}
    Returns the total score and a risk appetite category.
    """
    total_score = 0

    for question in QUESTIONS:
        q_id = question["id"]
        selected_option = answers.get(q_id)

        if selected_option and selected_option in question["options"]:
            _, points = question["options"][selected_option]
            total_score += points

    # Max possible score with 5 questions is 15, min is 5
    if total_score <= 8:
        risk_appetite = "Conservative"
    elif total_score <= 11:
        risk_appetite = "Moderate"
    else:
        risk_appetite = "Aggressive"

    return total_score, risk_appetite


def get_investment_recommendation(risk_appetite, credit_risk_bucket):
    """
    Combines the user's risk appetite (from the quiz) with their
    credit risk bucket (from our scoring model) to suggest instruments.
    This is the core "prove it, then grow it" logic - someone with a
    Poor credit bucket gets nudged toward safer options first, regardless
    of what the quiz alone suggests, since they're still building trust.
    """

    # Base recommendations purely from risk appetite
    base_recommendations = {
        "Conservative": ["Recurring Deposits (RD)", "Debt Mutual Funds", "Fixed Deposits (FD)"],
        "Moderate": ["Hybrid Mutual Funds", "Index Funds", "Micro SIPs (₹250-₹500/month)"],
        "Aggressive": ["Equity Mutual Funds", "Index Funds", "Diversified Stock SIPs"],
    }

    instruments = base_recommendations[risk_appetite]

    # Overriding/adjusting based on credit bucket - this is our unique layer
    if credit_risk_bucket in ["Poor", "Fair"]:
        # Regardless of quiz answers, nudge toward safer entry-level instruments first
        instruments = ["Recurring Deposits (RD)", "Micro SIPs (₹250/month)", "Debt Mutual Funds"]
        note = (
            f"Since your current credit score bucket is '{credit_risk_bucket}', we recommend "
            f"starting with safer, smaller investments. As your score improves, you'll unlock "
            f"access to {risk_appetite.lower()}-risk options like equity funds."
        )
    else:
        note = (
            f"Based on your risk appetite ({risk_appetite}) and strong credit bucket "
            f"('{credit_risk_bucket}'), you have access to a wider range of investment options."
        )

    return {
        "risk_appetite": risk_appetite,
        "credit_risk_bucket": credit_risk_bucket,
        "recommended_instruments": instruments,
        "explanation": note,
        "disclaimer": "These suggestions are for educational purposes only and do not constitute regulated financial advice."
    }


# --- Quick test when running this file directly ---
if __name__ == "__main__":
    sample_answers = {
        "q1": "b",
        "q2": "c",
        "q3": "a",
        "q4": "b",
        "q5": "c"
    }

    score, appetite = calculate_risk_appetite(sample_answers)
    print(f"Total score: {score}")
    print(f"Risk appetite: {appetite}")

    # Testing with a Poor credit bucket to see the override logic in action
    recommendation = get_investment_recommendation(appetite, "Poor")
    print("\n--- Recommendation (Poor credit bucket) ---")
    for key, value in recommendation.items():
        print(f"{key}: {value}")

    # Testing with an Excellent credit bucket for comparison
    recommendation2 = get_investment_recommendation(appetite, "Excellent")
    print("\n--- Recommendation (Excellent credit bucket) ---")
    for key, value in recommendation2.items():
        print(f"{key}: {value}")


