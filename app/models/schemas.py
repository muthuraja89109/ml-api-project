from typing import List

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    sepal_length: float = Field(
        ...,
        gt=0,
        description="Sepal length must be greater than 0"
    )

    sepal_width: float = Field(
        ...,
        gt=0,
        description="Sepal width must be greater than 0"
    )

    petal_length: float = Field(
        ...,
        gt=0,
        description="Petal length must be greater than 0"
    )

    petal_width: float = Field(
        ...,
        gt=0,
        description="Petal width must be greater than 0"
    )


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    request_id: str

class PredictionV2Output(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    request_id: str


class PredictionBatchInput(BaseModel):
    inputs: List[PredictionInput]


class PredictionBatchOutput(BaseModel):
    predictions: List[PredictionOutput]

class ModelInfo(BaseModel):
    model_type: str
    model_version: str
    training_date: str
    accuracy: float
    features: List[str]