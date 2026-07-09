import pandas as pd
import shap
import joblib

# Load the model and label encoder we already trained and saved
model = joblib.load("backend/scoring/credit_model.pkl")
label_encoder = joblib.load("backend/scoring/label_encoder.pkl")

# Load our data again so we have real users to explain
df = pd.read_csv("data/synthetic_users.csv")

feature_columns = [
    "recharge_frequency",
    "utility_regularity",
    "ecommerce_frequency",
    "ecommerce_avg_amount",
    "avg_balance"
]

X = df[feature_columns]

# Creating a SHAP explainer for our Random Forest model
# This studies how the model makes decisions
explainer = shap.TreeExplainer(model)

def explain_user(user_index):
    # Grab one user's data
    user_data = X.iloc[[user_index]]

    # Get the model's actual prediction for this user
    predicted_class = model.predict(user_data)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    # Get SHAP values - these tell us how much each feature pushed
    # the prediction toward or away from the predicted class
    shap_values = explainer.shap_values(user_data)

    # shap_values shape depends on SHAP version, this handles the common case
    # where we get values for each class - we pick the values for the predicted class
    if isinstance(shap_values, list):
        class_shap_values = shap_values[predicted_class][0]
    else:
        class_shap_values = shap_values[0, :, predicted_class]

    # Pairing each feature name with how much it influenced the result
    feature_impact = list(zip(feature_columns, class_shap_values))

    # Sorting by strength of impact (ignoring positive/negative direction for ranking)
    feature_impact_sorted = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)

    top_3 = feature_impact_sorted[:3]

    print(f"\nUser ID: {df.iloc[user_index]['user_id']}")
    print(f"Predicted Risk Bucket: {predicted_label}")
    print("Top 3 factors influencing this score:")

    for feature_name, impact_value in top_3:
        direction = "increased" if impact_value > 0 else "decreased"
        print(f"  - {feature_name}: {direction} the score (impact: {impact_value:.2f})")


if __name__ == "__main__":
    # Testing this on a few sample users
    explain_user(0)
    explain_user(1)
    explain_user(2)

    