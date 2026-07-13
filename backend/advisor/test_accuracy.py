from nivesh_mitra import score_answer

# This is our "ground truth" test set - we manually decided the correct
# score for each answer, based on what a human would reasonably judge.
# This lets us measure how accurately our rule-based scoring matches
# expected human judgment.

TEST_CASES = [
    {"question_id": "q1", "answer": "I would panic and sell everything", "expected": 1},
    {"question_id": "q1", "answer": "no worries, that's normal", "expected": 3},
    {"question_id": "q1", "answer": "I would be a bit nervous but wait", "expected": 2},
    {"question_id": "q2", "answer": "10 years", "expected": 3},
    {"question_id": "q2", "answer": "I need it back in 6 months", "expected": 1},
    {"question_id": "q2", "answer": "maybe 2 years", "expected": 2},
    {"question_id": "q3", "answer": "No i have not done that", "expected": 1},
    {"question_id": "q3", "answer": "yes and also gained profit", "expected": 3},
    {"question_id": "q3", "answer": "a little bit, still learning", "expected": 2},
    {"question_id": "q4", "answer": "15000rs, irregular", "expected": 1},
    {"question_id": "q4", "answer": "1000000rs stable job", "expected": 3},
    {"question_id": "q4", "answer": "30000 mostly stable", "expected": 2},
    {"question_id": "q5", "answer": "growth", "expected": 3},
    {"question_id": "q5", "answer": "keep it safe", "expected": 1},
    {"question_id": "q5", "answer": "balance of both", "expected": 2},
]


def run_accuracy_test():
    correct = 0
    total = len(TEST_CASES)

    print("Running accuracy test on rule-based scoring...\n")

    for i, case in enumerate(TEST_CASES, 1):
        predicted = score_answer(case["question_id"], case["answer"])
        is_correct = predicted == case["expected"]

        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} Test {i}: answer=\"{case['answer']}\" | expected={case['expected']} | got={predicted}")

    accuracy = (correct / total) * 100
    print(f"\n=== Accuracy: {correct}/{total} = {accuracy:.2f}% ===")

    return accuracy


if __name__ == "__main__":
    run_accuracy_test()

    