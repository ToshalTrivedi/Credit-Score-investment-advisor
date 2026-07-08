import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the synthetic dataset we have generated in data folder
df = pd.read_csv("data/synthetic_users.csv")

# These are the signals our model will learn from
feature_columns = [
    "recharge_frequency",
    "utility_regularity",
    "ecommerce_frequency",
    "ecommerce_avg_amount",
    "avg_balance"
]

X = df[feature_columns]
y = df["risk_bucket"]

# Encoding the risk bucket labels (Excellent/Good/Fair/Poor) into numbers
# since the model needs numeric targets internally
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting data - 80% to train the model, 20% to test how well it actually learned
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Training a Random Forest a solid default choice for tabular data like this
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# Checking how well it performs on data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy on test set: {accuracy:.2%}")
print("\nDetailed performance by risk bucket:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Saving the trained model and the label encoder together
# We need the label encoder later to convert predictions back to Excellent/Good/Fair/Poor
joblib.dump(model, "backend/scoring/credit_model.pkl")
joblib.dump(label_encoder, "backend/scoring/label_encoder.pkl")

print("\nModel and label encoder saved successfully.")

# Quick sanity check test the model on the sample user
sample_user = X_test.iloc[[0]]

predicted_class = model.predict(sample_user)

predicted_label = label_encoder.inverse_transform(predicted_class)

print(f"\nSample prediction for one test user: {predicted_label[0]}")




