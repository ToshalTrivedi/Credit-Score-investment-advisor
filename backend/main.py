import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

app = FastAPI(title="Transparent Credit Scoring API")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model = joblib.load(os.path.join(BASE_DIR, "scoring", "credit_model.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "scoring", "label_encoder.pkl"))

feature_columns = [
    "recharge_frequency",
    "utility_regularity",
    "ecommerce_frequency",
    "ecommerce_avg_amount",
    "avg_balance"
]

explainer = shap.TreeExplainer(model)



class UserData(BaseModel):
    recharge_frequency: float
    utility_regularity: float
    ecommerce_frequency: float
    ecommerce_avg_amount: float
    avg_balance: float


@app.get("/")
def home():
    return {"message": "Transparent Credit Scoring API is running"}


@app.post("/predict")
def predict_score(user: UserData):
   
    input_df = pd.DataFrame([[
        user.recharge_frequency,
        user.utility_regularity,
        user.ecommerce_frequency,
        user.ecommerce_avg_amount,
        user.avg_balance
    ]], columns=feature_columns)


    predicted_class = model.predict(input_df)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        class_shap_values = shap_values[predicted_class][0]
    else:
        class_shap_values = shap_values[0, :, predicted_class]

    feature_impact = list(zip(feature_columns, class_shap_values))
    feature_impact_sorted = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)
    top_3 = feature_impact_sorted[:3]

    top_3_explanation = [
        {
            "feature": name,
            "impact": round(float(value), 2),
            "direction": "increased" if value > 0 else "decreased"
        }
        for name, value in top_3
    ]

    return {
        "risk_bucket": predicted_label,
        "top_3_factors": top_3_explanation
    }



