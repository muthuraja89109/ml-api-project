from pathlib import Path
from datetime import datetime
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Find the project and dataset
# ============================================================

project_root = Path(__file__).resolve().parent.parent

dataset_path = (
    project_root
    / "ml"
    / "data"
    / "iris_dataset.csv"
)


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
# 4. Split dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. Create ML model
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 6. Train model
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 7. Make predictions
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. Evaluate model
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("Model Training Completed")
print("------------------------")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 9. Save trained model
# ============================================================

model_directory = (
    project_root
    / "ml"
    / "saved_model"
)

model_directory.mkdir(
    parents=True,
    exist_ok=True
)


model_path = (
    model_directory
    / "model.joblib"
)

joblib.dump(
    model,
    model_path
)


# ============================================================
# 10. Save model metadata
# ============================================================

metadata = {

    "model_type": type(model).__name__,

    "model_version": "1.0.0",

    "training_date": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),

    "accuracy": round(
        float(accuracy),
        4
    ),

    "features": list(X.columns)
}


metadata_path = (
    model_directory
    / "metadata.json"
)


with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 11. Final output
# ============================================================

print("------------------------")

print(
    f"Model saved successfully to: {model_path}"
)

print(
    f"Metadata saved successfully to: {metadata_path}"
)