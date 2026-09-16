from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your frontend's origin — adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# PROMETHEUS MONITORING
# ============================================================

# Instruments the app with default metrics (request count, latency, etc.)
# and exposes them at GET /metrics for Prometheus to scrape.
Instrumentator().instrument(app).expose(app)


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