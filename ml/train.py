from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Find the project and dataset
# ============================================================

project_root = Path(__file__).resolve().parent.parent

dataset_path = project_root / "ml" / "data" / "iris_dataset.csv"


# ============================================================
# 2. Load the Iris CSV dataset
# ============================================================

data = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print(f"Dataset shape: {data.shape}")


# ============================================================
# 3. Separate features and target
# ============================================================

X = data.drop("species", axis=1)

y = data["species"]


# ============================================================
# 4. Split the dataset into training and testing data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. Create the machine learning model
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 6. Train the model
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 7. Make predictions
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. Evaluate the model
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print()
print("Model Training Completed")
print("------------------------")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 9. Save the trained model
# ============================================================

model_directory = project_root / "ml" / "saved_model"

model_directory.mkdir(
    parents=True,
    exist_ok=True
)

model_path = model_directory / "model.joblib"

joblib.dump(model, model_path)


print("------------------------")
print(f"Model saved successfully to: {model_path}")