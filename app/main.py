from contextlib import asynccontextmanager
from uuid import uuid4
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import PredictionInput, PredictionOutput
from app.logging_config import setup_logging


logger = setup_logging()

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = joblib.load("ml/saved_model/model.joblib")

    logger.info("Model loaded successfully")

    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):

    request_id = str(uuid4())

    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        "request_id=%s | method=%s | path=%s | status=%s | duration=%.4fs",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration
    )

    response.headers["X-Request-ID"] = request_id

    return response


@app.get("/")
def root():
    return {
        "message": "ML API is alive"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError
):

    logger.error(
        "request_id=%s | ValueError | %s",
        request.state.request_id,
        str(exc)
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Invalid prediction data"
        }
    )


@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(
    data: PredictionInput,
    request: Request
):

    request_id = request.state.request_id

    features = pd.DataFrame(
        [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

    try:

        prediction_probabilities = model.predict_proba(features)

        prediction_index = prediction_probabilities.argmax(axis=1)[0]

        prediction = model.classes_[prediction_index]

        confidence = prediction_probabilities[0][prediction_index]

    except Exception as e:

        logger.error(
            "request_id=%s | Prediction failed | error=%s",
            request_id,
            str(e),
            exc_info=True
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    logger.info(
        "request_id=%s | Prediction successful | prediction=%s | confidence=%.4f",
        request_id,
        prediction,
        confidence
    )

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "request_id": request_id
    }