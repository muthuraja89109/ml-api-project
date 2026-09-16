import httpx

BASE_URL = "http://localhost:8000"
API_KEY = "9c3ee05ef58a17be33bd620d46b33e19ae568574eb11a4ed3802fbad6a72067f"  # your real key
HEADERS = {"X-API-Key": API_KEY}


def test_integration_health():
    response = httpx.get(f"{BASE_URL}/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_integration_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = httpx.post(f"{BASE_URL}/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["setosa", "versicolor", "virginica"]


def test_integration_predict_batch():
    payload = {
        "inputs": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3},
        ]
    }
    response = httpx.post(f"{BASE_URL}/api/v1/predict-batch", json=payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2


def test_integration_metrics():
    response = httpx.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    assert "ml_predictions_total" in response.text