import re
from sentence_transformers import SentenceTransformer, util

print("Loading semantic understanding model... (only happens once)")
model_nlp = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.\n")

QUESTIONS = [
    {
        "id": "q1",
        "prompt": "How would you feel if your investment value dropped by 10% in a month?",
        "options": {
            "a": ("panic withdraw immediately fear scared", 1),
            "b": ("worried but wait and watch", 2),
            "c": ("stay calm this is normal expected", 3),
        }
    },
    {
        "id": "q2",
        "prompt": "How long can you keep this money invested without needing it back?",
        "options": {
            "a": ("less than a year short term soon", 1),
            "b": ("one to three years medium term", 2),
            "c": ("more than three years long term", 3),
        }
    },
    {
        "id": "q3",
        "prompt": "Have you ever invested in mutual funds, stocks, or similar before?",
        "options": {
            "a": ("never invested no experience completely new beginner", 1),
            "b": ("little experience still learning some exposure", 2),
            "c": ("yes experienced comfortable have invested before", 3),
        }
    },
    {
        "id": "q4",
        "prompt": "What is your monthly income, and how stable is it?",
        "options": {
            "a": ("low irregular unpredictable income", 1),
            "b": ("moderate mostly stable some variation income", 2),
            "c": ("high very stable reliable income", 3),
        }
    },
    {
        "id": "q5",
        "prompt": "What is your main goal with this money - safety or growth?",
        "options": {
            "a": ("keep money completely safe growth not important", 1),
            "b": ("balance between safety and some growth", 2),
            "c": ("maximize growth handle ups and downs", 3),
        }
    },
]

# --- RULE LAYER: catches obvious, unambiguous cases before ML even runs ---
# This directly fixes negation confusion (yes vs never) and number detection

NEGATIVE_EXPERIENCE_PATTERNS = [r"\bnever\b", r"\bno\b.*\binvest", r"not invested", r"never invested"]
POSITIVE_EXPERIENCE_PATTERNS = [r"\byes\b", r"i have invested", r"i've invested", r"i did invest"]

PANIC_PATTERNS = [r"\bpanic\b", r"withdraw", r"pull out", r"freak", r"scared"]
CALM_PATTERNS = [r"\bcalm\b", r"fine", r"normal", r"not worried", r"no problem"]


def check_rules(question_id, text):
    """Fast, explicit checks for unambiguous phrasing. Returns points or None if no rule fires."""
    text_lower = text.lower()

    if question_id == "q3":
        for pattern in NEGATIVE_EXPERIENCE_PATTERNS:
            if re.search(pattern, text_lower):
                return 1
        for pattern in POSITIVE_EXPERIENCE_PATTERNS:
            if re.search(pattern, text_lower):
                # "yes but lost money" still counts as real experience - points stay at 3,
                # the loss itself doesn't erase that they've actually invested before
                return 3

    if question_id == "q1":
        for pattern in PANIC_PATTERNS:
            if re.search(pattern, text_lower):
                return 1
        for pattern in CALM_PATTERNS:
            if re.search(pattern, text_lower):
                return 3

    return None


def extract_income_number(text):
    numbers = re.findall(r"\d+", text.replace(",", ""))
    return int(numbers[0]) if numbers else None


def semantic_score(user_text, options):
    user_embedding = model_nlp.encode(user_text, convert_to_tensor=True)
    best_option = None
    best_similarity = -1

    for option_key, (option_text, points) in options.items():
        option_embedding = model_nlp.encode(option_text, convert_to_tensor=True)
        similarity = util.cos_sim(user_embedding, option_embedding).item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_option = (option_key, points, option_text)

    return best_option, best_similarity


def interpret_answer(question, user_text):
    q_id = question["id"]

    # Income question relies primarily on the actual number, not semantic guessing
    if q_id == "q4":
        amount = extract_income_number(user_text)
        if amount is not None:
            if amount < 15000:
                points = 1
            elif amount < 40000:
                points = 2
            else:
                points = 3
            return points, f"(detected income: ₹{amount})"

    # Try the fast rule layer first
    rule_result = check_rules(q_id, user_text)
    if rule_result is not None:
        return rule_result, "(matched by rule-based pattern)"

    # Fall back to ML semantic matching only when rules don't catch it
    matched_option, similarity = semantic_score(user_text, question["options"])
    option_key, points, option_text = matched_option
    return points, f"(ML semantic match, confidence: {similarity:.2f})"


def calculate_risk_appetite(total_score):
    if total_score <= 8:
        return "Conservative"
    elif total_score <= 11:
        return "Moderate"
    else:
        return "Aggressive"


def get_investment_recommendation(risk_appetite, credit_risk_bucket):
    base_recommendations = {
        "Conservative": ["Recurring Deposits (RD)", "Debt Mutual Funds", "Fixed Deposits (FD)"],
        "Moderate": ["Hybrid Mutual Funds", "Index Funds", "Micro SIPs (₹250-₹500/month)"],
        "Aggressive": ["Equity Mutual Funds", "Index Funds", "Diversified Stock SIPs"],
    }
    instruments = base_recommendations[risk_appetite]

    if credit_risk_bucket in ["Poor", "Fair"]:
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


def run_interactive_quiz():
    print("=== Risk Profiling Assessment (Hybrid NLP + ML) ===")
    print("Answer naturally, in your own words.\n")

    total_score = 0
    for question in QUESTIONS:
        print(f"Q: {question['prompt']}")
        user_input = input("Your answer: ")
        points, detail = interpret_answer(question, user_input)
        print(f"  → Interpreted score: {points}/3 {detail}\n")
        total_score += points

    return total_score


if __name__ == "__main__":
    total_score = run_interactive_quiz()
    appetite = calculate_risk_appetite(total_score)

    print(f"\nTotal score: {total_score}")
    print(f"Risk appetite: {appetite}")

    sample_credit_bucket = "Fair"
    recommendation = get_investment_recommendation(appetite, sample_credit_bucket)

    print("\n--- Your Investment Recommendation ---")
    for key, value in recommendation.items():
        print(f"{key}: {value}")