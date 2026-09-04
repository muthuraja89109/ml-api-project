def test_health_returns_200(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_health_returns_correct_shape(client):
    response = client.get("/api/v1/health")

    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "ok"
    assert data["model_loaded"] is True