from fastapi import FastAPI
from app.schemas import CustomerData
from app.predictor import predict

app = FastAPI(title="Customer Churn Prediction API")


@app.post("/predict")
def predict_customer(customer: CustomerData):

    prediction, probability = predict(customer.model_dump())

    return {
        "prediction": "Yes" if prediction else "No",
        "probability": float(probability)
    } 
