import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Load Iris dataset
# --------------------------------------------------

data = load_iris()

X = data.data
y = data.target


# --------------------------------------------------
# 2. Split dataset into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Create MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("Iris_RandomForest_Experiments")


# --------------------------------------------------
# 4. Define 5 different experiments
# --------------------------------------------------

experiments = [
    {
        "n_estimators": 50,
        "max_depth": 2
    },
    {
        "n_estimators": 100,
        "max_depth": 3
    },
    {
        "n_estimators": 150,
        "max_depth": 4
    },
    {
        "n_estimators": 200,
        "max_depth": 5
    },
    {
        "n_estimators": 250,
        "max_depth": 6
    }
]


# --------------------------------------------------
# 5. Run all 5 experiments
# --------------------------------------------------

for params in experiments:

    print("\n----------------------------------------")
    print("Starting new experiment")
    print("n_estimators:", params["n_estimators"])
    print("max_depth:", params["max_depth"])
    print("----------------------------------------")

    with mlflow.start_run():

        # Create Random Forest model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        # --------------------------------------------------
        # Log parameters
        # --------------------------------------------------

        mlflow.log_param(
            "n_estimators",
            params["n_estimators"]
        )

        mlflow.log_param(
            "max_depth",
            params["max_depth"]
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # --------------------------------------------------
        # Log metric
        # --------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # --------------------------------------------------
        # Log Random Forest model
        #
        # Fix for:
        # UntrustedTypesFoundException:
        # sklearn.tree._tree.Tree
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # --------------------------------------------------
        # Display experiment result
        # --------------------------------------------------

        print("Experiment completed successfully.")
        print("Accuracy:", accuracy)


# --------------------------------------------------
# 6. Finished
# --------------------------------------------------

print("\n========================================")
print("ALL 5 EXPERIMENTS COMPLETED SUCCESSFULLY")
print("========================================")
print("Open MLflow UI using:")
print("mlflow ui")
print("========================================")