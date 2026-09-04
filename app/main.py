from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from app.routers.v2 import router as v2_router
from app.config import settings
from app.routers.v1 import router as v1_router


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model = joblib.load(
        settings.MODEL_PATH
    )

    print("Model loaded successfully!")

    yield


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.API_TITLE,
    lifespan=lifespan
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "ML API is alive"
    }


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(v1_router)
app.include_router(v2_router)