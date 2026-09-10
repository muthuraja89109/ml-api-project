from app.config import settings


def test_predict_batch_oversized_rejected(client, auth_headers):

    inputs = [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        for _ in range(settings.MAX_BATCH_SIZE + 1)
    ]

    response = client.post(
        "/api/v1/predict-batch",
        json={"inputs": inputs},
        headers=auth_headers
    )

    assert response.status_code == 400