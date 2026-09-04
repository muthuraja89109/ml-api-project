# v2 plan: /api/v2/predict will return an extra field (e.g. full class
# probability distribution instead of just top confidence, or a
# model_version field). v1 stays untouched — v2 gets its own router,
# schema, and prefix so existing clients calling /api/v1/predict never break.

import json
import logging
import time
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.config import settings

from app.models.schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionBatchInput,
    PredictionBatchOutput,
    ModelInfo
)


router = APIRouter(
    prefix="/api/v1"
)


logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health(request: Request):

    model = request.app.state.model

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# ============================================================
# SINGLE PREDICTION
# ============================================================

@router.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(
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

        prediction_index = (
            prediction_probabilities.argmax(axis=1)[0]
        )

        prediction = model.classes_[prediction_index]

        confidence = (
            prediction_probabilities[0][prediction_index]
        )

        logger.info(
            f"request_id={request_id} | "
            f"Prediction successful | "
            f"prediction={prediction} | "
            f"confidence={confidence:.4f}"
        )

        return {
            "prediction": str(prediction),
            "confidence": float(confidence),
            "request_id": request_id
        }

    except Exception:

        logger.exception(
            f"request_id={request_id} | Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


# ============================================================
# BATCH PREDICTION
# ============================================================

@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput
)
def predict_batch(
    data: PredictionBatchInput,
    request: Request
):

    start_time = time.perf_counter()

    model = request.app.state.model

    batch_size = len(data.inputs)

    # --------------------------------------------------------
    # Validate batch size using configuration
    # --------------------------------------------------------

    if batch_size < 1 or batch_size > settings.MAX_BATCH_SIZE:

        logger.warning(
            f"Invalid batch size | "
            f"batch_size={batch_size} | "
            f"max={settings.MAX_BATCH_SIZE}"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch size must be between "
                f"1 and {settings.MAX_BATCH_SIZE}"
            )
        )

    try:

        logger.info(
            f"Batch prediction started | "
            f"batch_size={batch_size}"
        )

        # ----------------------------------------------------
        # Convert complete batch into DataFrame
        # ----------------------------------------------------

        rows = [
            [
                item.sepal_length,
                item.sepal_width,
                item.petal_length,
                item.petal_width
            ]
            for item in data.inputs
        ]

        features = pd.DataFrame(
            rows,
            columns=FEATURE_NAMES
        )

        # ----------------------------------------------------
        # Predict entire batch at once
        # ----------------------------------------------------

        prediction_probabilities = model.predict_proba(
            features
        )

        prediction_indices = (
            prediction_probabilities.argmax(axis=1)
        )

        predictions = model.classes_[
            prediction_indices
        ]

        results = []

        for index, prediction in enumerate(predictions):

            prediction_index = prediction_indices[index]

            confidence = prediction_probabilities[
                index
            ][prediction_index]

            results.append(
                PredictionOutput(
                    prediction=str(prediction),
                    confidence=float(confidence),
                    request_id=str(uuid4())
                )
            )

        # ----------------------------------------------------
        # Calculate duration
        # ----------------------------------------------------

        duration = time.perf_counter() - start_time

        logger.info(
            f"Batch prediction successful | "
            f"batch_size={batch_size} | "
            f"duration={duration:.4f}s"
        )

        return {
            "predictions": results
        }

    except Exception:

        duration = time.perf_counter() - start_time

        logger.exception(
            f"Batch prediction failed | "
            f"batch_size={batch_size} | "
            f"duration={duration:.4f}s"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

@router.get(
    "/model-info",
    response_model=ModelInfo
)
def model_info():

    try:

        with open(
            settings.METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        logger.info(
            "Model metadata retrieved successfully"
        )

        return metadata

    except Exception:

        logger.exception(
            "Failed to load model metadata"
        )

        raise HTTPException(
            status_code=500,
            detail="Model metadata could not be loaded"
        )