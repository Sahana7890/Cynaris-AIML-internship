from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load sample dataset
data = load_iris()

# Train a simple ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(data.data, data.target)


def predict(features):
    """Return the predicted Iris class."""
    prediction = model.predict([features])[0]
    return int(prediction)