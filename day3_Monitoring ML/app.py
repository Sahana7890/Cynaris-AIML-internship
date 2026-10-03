from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

app = FastAPI(title="ML Prediction API")


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "ml-api"
    }


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    prediction = predict(request.features)

    return {
        "prediction": prediction
    }