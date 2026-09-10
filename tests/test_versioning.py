def test_v1_and_v2_have_different_response_shapes(client, auth_headers):

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    # Call v1
    v1_response = client.post(
        "/api/v1/predict",
        json=payload,
        headers=auth_headers
    )

    # Call v2
    v2_response = client.post(
        "/api/v2/predict",
        json=payload,
        headers=auth_headers
    )

    # Both endpoints must work
    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # v1 response shape
    assert "prediction" in v1_data
    assert "confidence" in v1_data
    assert "request_id" in v1_data

    # v2 response shape
    assert "prediction" in v2_data
    assert "probabilities" in v2_data
    assert "request_id" in v2_data

    # Prove the shapes are different
    assert "confidence" not in v2_data
    assert "probabilities" not in v1_data

    # Validate prediction
    assert v1_data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert v2_data["prediction"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    # Validate v1 confidence
    assert 0 <= v1_data["confidence"] <= 1

    # Validate v2 probabilities
    assert set(v2_data["probabilities"].keys()) == {
        "setosa",
        "versicolor",
        "virginica"
    }

    assert abs(sum(v2_data["probabilities"].values()) - 1.0) < 0.001