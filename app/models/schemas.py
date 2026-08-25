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