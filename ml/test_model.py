from pathlib import Path

import joblib
from sklearn.datasets import load_iris


# Find the saved model
current_directory = Path(__file__).resolve().parent
model_path = current_directory / "saved_model" / "model.joblib"


# Load the Iris dataset
iris = load_iris()


# Load the saved model
model = joblib.load(model_path)


# Sample flower data
sample = [[5.1, 3.5, 1.4, 0.2]]


# Make prediction
prediction = model.predict(sample)


# Convert prediction number to flower name
predicted_class = iris.target_names[prediction[0]]


print("Model loaded successfully!")
print(f"Input: {sample[0]}")
print(f"Predicted class: {predicted_class}")