import os
import sys
import re
import ollama
import joblib
import pandas as pd

from growth_projection import generate_growth_projection, print_projection_summary

# --- Loading the trained credit scoring model ---
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scoring"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCORING_DIR = os.path.join(BASE_DIR, "..", "scoring")

credit_model = joblib.load(os.path.join(SCORING_DIR, "credit_model.pkl"))
label_encoder = joblib.load(os.path.join(SCORING_DIR, "label_encoder.pkl"))

CREDIT_FEATURE_COLUMNS = [
    "recharge_frequency",
    "utility_regularity",
    "ecommerce_frequency",
    "ecommerce_avg_amount",
    "avg_balance"
]


def get_real_credit_bucket(user_financial_data):
    """
    Takes the user's actual financial habit data and runs it through our
    trained credit scoring model to get their real risk bucket.
    """
    input_df = pd.DataFrame([[
        user_financial_data["recharge_frequency"],
        user_financial_data["utility_regularity"],
        user_financial_data["ecommerce_frequency"],
        user_financial_data["ecommerce_avg_amount"],
        user_financial_data["avg_balance"]
    ]], columns=CREDIT_FEATURE_COLUMNS)

    predicted_class = credit_model.predict(input_df)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label


def interpret_utility_regularity(answer):
    """
    Converts a natural, everyday answer about bill payment habits into
    a percentage the credit model can understand - instead of forcing
    the user to think in numbers themselves.
    """
    answer_lower = answer.lower().strip()

    if "always" in answer_lower or "every time" in answer_lower:
        return 95.0
    elif "mostly" in answer_lower or "usually" in answer_lower or "often" in answer_lower:
        return 80.0
    elif "sometimes" in answer_lower or "occasionally" in answer_lower:
        return 55.0
    elif "rarely" in answer_lower or "never" in answer_lower or "hardly" in answer_lower:
        return 30.0
    else:
        numbers = re.findall(r"\d+", answer)
        if numbers:
            return float(numbers[0])
        return 60.0


RECOMMENDATION_SYSTEM_PROMPT = """
You are a financial education assistant for underserved and first-time investors in India.

You will be given HARD FACTS about a user (exact income, income tier, risk appetite, credit bucket)
and their raw quiz answers for tone/context only.

Your task: write a short, warm, personalized explanation (4-6 sentences) that:
1. References their ACTUAL answers naturally (their time horizon, their stated goal)
2. Uses the EXACT income figure given to you - do not change, round, or invent a different number
3. Explains WHY they landed in this risk appetite category
4. Recommends 2-3 general INSTRUMENT CATEGORIES suitable for them (e.g., "large-cap equity mutual funds",
   "hybrid funds", "debt funds", "index funds", "recurring deposits", "micro-SIPs")
5. If the income tier says "low income," you MUST prioritize very small amounts (₹100-500 micro-SIPs)
   and emergency fund building FIRST - this overrides the risk appetite score
6. If their credit bucket is Poor or Fair, gently note they should start with safer options first

STRICT RULES YOU MUST FOLLOW:
- NEVER name a specific real company, stock ticker, IPO, or fund house
- NEVER guarantee returns or use words like "definitely will grow" or "guaranteed profit"
- ALWAYS end your response with exactly this line on its own:
  "Disclaimer: This is for educational purposes only and does not constitute regulated financial advice."
- Do not answer anything unrelated to this task. Stay strictly in this role.
"""

QUESTIONS = [
    {"id": "q1", "prompt": "How would you feel if your investment value dropped by 10% in a month?"},
    {"id": "q2", "prompt": "How long can you keep this money invested without needing it back?"},
    {"id": "q3", "prompt": "Have you ever invested in mutual funds, stocks, or similar before?"},
    {"id": "q4", "prompt": "What is your monthly income, and how stable is it?"},
    {"id": "q5", "prompt": "What is your main goal with this money - safety or growth?"},
]

# --- RULE-BASED SCORING LAYER ---

NEGATIVE_EXPERIENCE_PATTERNS = [r"\bnever\b", r"\bno\b.*\binvest", r"not invested", r"never invested", r"have not"]
POSITIVE_EXPERIENCE_PATTERNS = [r"\byes\b", r"i have invested", r"i've invested", r"i did invest", r"gained profit"]

PANIC_PATTERNS = [r"\bpanic\b", r"withdraw", r"pull out", r"freak", r"scared", r"sell everything", r"take.*money back", r"take that money"]
CALM_PATTERNS = [r"\bcalm\b", r"no worries", r"fine", r"normal", r"not worried", r"no problem"]
NERVOUS_PATTERNS = [r"nervous", r"worried", r"bit (of )?worried", r"a bit"]

SHORT_TERM_PATTERNS = [r"\b[0-6] months?\b", r"less than a year", r"\bsoon\b", r"need it back"]
LONG_TERM_PATTERNS = [r"\b([5-9]|1[0-9]|[2-9][0-9])\s*years?\b", r"more than three", r"long term"]

SAFE_GOAL_PATTERNS = [r"safe", r"safety"]
GROWTH_GOAL_PATTERNS = [r"growth", r"grow", r"maximize"]


def match_any(text, patterns):
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def score_q1(answer):
    if match_any(answer, PANIC_PATTERNS):
        return 1
    if match_any(answer, CALM_PATTERNS):
        return 3
    if match_any(answer, NERVOUS_PATTERNS):
        return 2
    return 2


def score_q2(answer):
    if match_any(answer, SHORT_TERM_PATTERNS):
        return 1
    if match_any(answer, LONG_TERM_PATTERNS):
        return 3
    return 2


def score_q3(answer):
    if match_any(answer, NEGATIVE_EXPERIENCE_PATTERNS):
        return 1
    if match_any(answer, POSITIVE_EXPERIENCE_PATTERNS):
        return 3
    return 2


def score_q4(answer):
    numbers = re.findall(r"\d+", answer.replace(",", ""))
    if numbers:
        income = int(numbers[0])
        if income <= 15000:
            return 1
        elif income < 50000:
            return 2
        else:
            return 3
    return 2


def score_q5(answer):
    if match_any(answer, SAFE_GOAL_PATTERNS):
        return 1
    if match_any(answer, GROWTH_GOAL_PATTERNS):
        return 3
    return 2


SCORING_FUNCTIONS = {
    "q1": score_q1,
    "q2": score_q2,
    "q3": score_q3,
    "q4": score_q4,
    "q5": score_q5,
}


def score_answer(question_id, answer):
    return SCORING_FUNCTIONS[question_id](answer)


def calculate_risk_appetite(total_score):
    if total_score <= 8:
        return "Conservative"
    elif total_score <= 11:
        return "Moderate"
    else:
        return "Aggressive"


def extract_income(answers_dict):
    income_text = answers_dict.get("q4", "")
    numbers = re.findall(r"\d+", income_text.replace(",", ""))
    return int(numbers[0]) if numbers else 0


def get_income_tier(income_value):
    if income_value < 15000:
        return "low income - focus should be on very small, safe amounts and building an emergency fund first"
    elif income_value < 50000:
        return "moderate income - can consider modest, diversified investments alongside safety net building"
    else:
        return "higher income - has more flexibility to consider a broader mix of instruments"


def generate_personalized_recommendation(answers_dict, risk_appetite, credit_bucket):
    income_value = extract_income(answers_dict)
    income_tier = get_income_tier(income_value)

    context = f"""
HARD FACTS - you MUST use these exact numbers, do not change or estimate them:
- Exact monthly income: ₹{income_value}
- Income tier assessment: {income_tier}
- Calculated risk appetite: {risk_appetite}
- Credit score risk bucket: {credit_bucket}

User's raw answers (for tone/context only, not for extracting numbers):
- Reaction to a 10% drop: {answers_dict.get('q1')}
- Investment time horizon: {answers_dict.get('q2')}
- Past investing experience: {answers_dict.get('q3')}
- Main goal: {answers_dict.get('q5')}

Write the personalized explanation now, referencing the EXACT income figure of ₹{income_value},
following all your rules.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
    )

    return response["message"]["content"].strip(), income_value


def validate_recommendation(text, income_value):
    numbers_in_response = re.findall(r"₹?\s?(\d{4,})", text.replace(",", ""))
    if numbers_in_response:
        mentioned_income = int(numbers_in_response[0])
        if income_value > 0 and abs(mentioned_income - income_value) > income_value * 0.2:
            return False
    return True


def suggest_monthly_investment(income_value, risk_appetite):
    """
    Suggests a realistic MICRO-investment amount, strictly within the
    ₹500-₹5,000/month range this project targets (per the problem statement,
    this is specifically for small-ticket investing for underserved users -
    NOT a general wealth management calculator for any income level).
    """
    MIN_AMOUNT = 500
    MAX_AMOUNT = 5000

    if income_value <= 0:
        defaults = {"Conservative": 500, "Moderate": 1500, "Aggressive": 2500}
        return defaults.get(risk_appetite, 500)

    if income_value < 15000:
        percentage = 0.03
    elif income_value < 50000:
        percentage = 0.06
    else:
        percentage = 0.08

    suggested_amount = income_value * percentage

    # HARD CAP - never suggest outside our project's actual micro-investment scope
    suggested_amount = max(MIN_AMOUNT, min(MAX_AMOUNT, suggested_amount))
    suggested_amount = round(suggested_amount / 50) * 50

    return suggested_amount


def process_full_recommendation(quiz_answers, financial_data):
    """
    Non-interactive version of the full flow - takes data directly
    (instead of asking via input()) so it can be called from an API.

    quiz_answers: dict like {"q1": "...", "q2": "...", "q3": "...", "q4": "...", "q5": "..."}
    financial_data: dict like {"recharge_frequency": 10, "utility_regularity": 85, ...}
    """
    total_score = 0
    for question in QUESTIONS:
        q_id = question["id"]
        answer = quiz_answers.get(q_id, "")
        score = score_answer(q_id, answer)
        total_score += score

    appetite = calculate_risk_appetite(total_score)

    real_credit_bucket = get_real_credit_bucket(financial_data)

    recommendation_text, income_value = generate_personalized_recommendation(
        quiz_answers, appetite, real_credit_bucket
    )

    is_valid = validate_recommendation(recommendation_text, income_value)

    suggested_amount = suggest_monthly_investment(income_value, appetite)
    projection = generate_growth_projection(appetite, monthly_amount=suggested_amount)

    return {
        "total_score": total_score,
        "risk_appetite": appetite,
        "credit_risk_bucket": real_credit_bucket,
        "recommendation_text": recommendation_text,
        "recommendation_validated": is_valid,
        "suggested_monthly_investment": suggested_amount,
        "growth_projection": projection,
    }


def run_interactive_quiz():
    print("=== Nivesh Mitra - Your Personal Investment Guide ===\n")
    total_score = 0
    answers_dict = {}

    for question in QUESTIONS:
        print(f"Q: {question['prompt']}")
        user_input = input("Your answer: ")

        answers_dict[question["id"]] = user_input
        score = score_answer(question["id"], user_input)
        print(f"  → Score: {score}/3\n")

        total_score += score

    appetite = calculate_risk_appetite(total_score)
    print(f"\nTotal score: {total_score}")
    print(f"Risk appetite: {appetite}")

    print("\nNow let's calculate your real credit score based on your financial habits.\n")

    recharge_frequency = float(input("How many times do you usually recharge your mobile in a month? "))

    utility_answer = input("Do you pay your electricity/water/gas bills on time? (always / mostly / sometimes / rarely) ")
    utility_regularity = interpret_utility_regularity(utility_answer)

    ecommerce_frequency = float(input("How many times do you shop online in a typical month? "))
    ecommerce_avg_amount = float(input("On average, how much do you usually spend per online purchase (in ₹)? "))
    avg_balance = float(input("Roughly, how much money do you usually keep in your bank account (in ₹)? "))

    user_financial_data = {
        "recharge_frequency": recharge_frequency,
        "utility_regularity": utility_regularity,
        "ecommerce_frequency": ecommerce_frequency,
        "ecommerce_avg_amount": ecommerce_avg_amount,
        "avg_balance": avg_balance
    }

    real_credit_bucket = get_real_credit_bucket(user_financial_data)
    print(f"\nYour real credit score bucket: {real_credit_bucket}\n")

    print("Generating your personalized recommendation...\n")
    recommendation_text, income_value = generate_personalized_recommendation(answers_dict, appetite, real_credit_bucket)

    if not validate_recommendation(recommendation_text, income_value):
        print("⚠️ Warning: AI may have misreported the income figure. Please verify manually.\n")

    print("--- Nivesh Mitra says ---")
    print(recommendation_text)

    suggested_amount = suggest_monthly_investment(income_value, appetite)

    print(f"\nBased on your income, we suggest starting with ₹{suggested_amount}/month.")
    print("Here's how that could grow over time:\n")

    projection = generate_growth_projection(appetite, monthly_amount=suggested_amount)
    print_projection_summary(projection)

    return total_score, appetite, recommendation_text, projection


if __name__ == "__main__":
    run_interactive_quiz()

