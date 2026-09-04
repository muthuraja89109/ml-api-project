def test_model_info_returns_200(client):
    response = client.get("/api/v1/model-info")

    assert response.status_code == 200


def test_model_info_contains_expected_keys(client):
    response = client.get("/api/v1/model-info")

    data = response.json()

    assert "model_type" in data
    assert "model_version" in data
    assert "training_date" in data
    assert "accuracy" in data
    assert "features" in data


def test_model_info_has_correct_model_type(client):
    response = client.get("/api/v1/model-info")

    data = response.json()

    assert data["model_type"] == "RandomForestClassifier"