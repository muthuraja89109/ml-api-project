def test_predict_valid_input(client, auth_headers):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers=auth_headers
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "request_id" in data

    assert data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert 0 <= data["confidence"] <= 1


def test_predict_missing_field_returns_422(client, auth_headers):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers=auth_headers
    )

    assert response.status_code == 422


def test_predict_invalid_field_type_returns_422(client, auth_headers):
    payload = {
        "sepal_length": "invalid",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers=auth_headers
    )

    assert response.status_code == 422