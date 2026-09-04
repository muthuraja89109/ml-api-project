import logging
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import PredictionInput, PredictionV2Output


router = APIRouter(
    prefix="/api/v2"
)


logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]


@router.post(
    "/predict",
    response_model=PredictionV2Output
)
def predict_v2(
    data: PredictionInput,
    request: Request
):

    model = request.app.state.model

    features = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=FEATURE_NAMES
    )

    request_id = str(uuid4())

    try:

        prediction_probabilities = model.predict_proba(
            features
        )

        probabilities = prediction_probabilities[0]

        prediction_index = probabilities.argmax()

        prediction = model.classes_[prediction_index]

        full_probabilities = {
            str(class_name): float(probability)
            for class_name, probability
            in zip(model.classes_, probabilities)
        }

        logger.info(
            f"request_id={request_id} | "
            f"V2 prediction successful | "
            f"prediction={prediction}"
        )

        return {
            "prediction": str(prediction),
            "probabilities": full_probabilities,
            "request_id": request_id
        }

    except Exception:

        logger.exception(
            f"request_id={request_id} | "
            f"V2 prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )