from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. Find the saved model
# ============================================================

project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "ml" / "saved_model" / "model.joblib"


# ============================================================
# 2. Load the trained model
# ============================================================

model = joblib.load(model_path)

print("Model loaded successfully!")


# ============================================================
# 3. Create test input
# ============================================================

input_data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)


# ============================================================
# 4. Make prediction
# ============================================================

prediction = model.predict(input_data)


# ============================================================
# 5. Display result
# ============================================================

print("Input:", input_data.iloc[0].tolist())
print("Predicted class:", prediction[0])