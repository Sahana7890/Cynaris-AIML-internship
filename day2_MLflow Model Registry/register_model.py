import mlflow
from mlflow import MlflowClient

RUN_ID = "PASTE_YOUR_BEST_RUN_ID_HERE"

MODEL_NAME = "Iris_RandomForest_Best_Model"

model_uri = f"runs:/{RUN_ID}/random_forest_model"

client = MlflowClient()

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("Model registered successfully.")
print("Model name:", result.name)
print("Model version:", result.version)