import joblib
import pandas as pd

pipeline = joblib.load("models/churn_pipeline.pkl")


def predict(customer):

    df = pd.DataFrame([customer])

    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    return prediction, probability 