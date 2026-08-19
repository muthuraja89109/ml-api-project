from pathlib import Path

import joblibgit status
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Load the Iris dataset
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target


# ============================================================
# 2. Split the dataset into training and testing data
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. Create the machine learning model
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 4. Train the model
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 5. Make predictions on the test data
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 6. Evaluate the model
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("Model Training Completed")
print("------------------------")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# ============================================================
# 7. Save the trained model
# ============================================================

project_root = Path(__file__).resolve().parent.parent

model_directory = project_root / "ml" / "saved_model"
model_directory.mkdir(parents=True, exist_ok=True)

model_path = model_directory / "model.joblib"

joblib.dump(model, model_path)


print("------------------------")
print(f"Model saved successfully to: {model_path}")