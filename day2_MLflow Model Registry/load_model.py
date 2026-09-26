import mlflow

model_uri = "models:/Iris_RandomForest_Best_Model/1"

model = mlflow.sklearn.load_model(model_uri)

print("Registered model loaded successfully.")
print(model)