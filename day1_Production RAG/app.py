from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Prediction API")


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"message": "ML API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array(data.features).reshape(1, -1)

    # Example prediction
    prediction = float(np.mean(features))

    return {
        "prediction": prediction
    }