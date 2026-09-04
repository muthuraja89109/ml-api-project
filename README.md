# ML Model Deployment as a Monitored REST API

A machine learning model (Iris species classifier) served through a versioned, tested, and configurable REST API built with **FastAPI**, as part of the Python Intern task program.

> **Status:** Tasks 1–14 complete ✅ (Foundation + Phase 2: Core Development + Phase 3: Feature Development)

---

## 📌 Project Overview

This project trains a simple ML model and serves predictions through a REST API.

- **Dataset:** Iris flower dataset (`scikit-learn` built-in / CSV)
- **Problem type:** Classification — predict the Iris species from flower measurements
- **API contract:**
  - `POST /api/v1/predict` — accepts flower measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) as a Pydantic-validated JSON body, and returns the predicted species, a confidence score, and a request ID.
  - `POST /api/v1/predict-batch` — accepts a list of measurement sets (1–`MAX_BATCH_SIZE`) and returns a prediction for each, computed in a single batched model call.
  - `GET /api/v1/model-info` — returns real model metadata (type, version, training date, accuracy, feature names).
  - `GET /api/v1/health` — health check returning `{"status": "ok", "model_loaded": true}`.
  - `POST /api/v2/predict` — same input as v1, but returns a richer response: full class **probability distribution** and a `model_version` field, alongside the original `prediction`, `confidence`, and `request_id`.
  - `GET /` — root/liveness check.
- **Versioning:** All endpoints are namespaced under `/api/v1/...` or `/api/v2/...` via `APIRouter`, so v1 and v2 can evolve independently without breaking existing clients.
- **Validation:** Request bodies validated with `PredictionInput` / `PredictionBatchInput` Pydantic models (all measurements required, must be greater than zero).
- **Response contract:** Enforced with `response_model` (`PredictionOutput`, `PredictionOutputV2`, `PredictionBatchOutput`, `ModelInfo`); errors return a clean, documented shape — never a raw traceback.
- **Configuration:** Driven entirely by environment variables via `pydantic-settings` — no hardcoded paths, limits, or titles.
- **Observability:** Every request and prediction is logged (console + `app.log`) with a unique request ID, HTTP method, path, status code, and duration.
- **Testing:** A pytest suite (10 tests) covers success paths, validation errors, batch limits, and version compatibility — with proof that the tests catch real regressions.

---

## 🗂️ Project Structure

```
ml-api-project/
├── app/
│   ├── main.py               # FastAPI app entry point (lifespan model load, router registration)
│   ├── config.py              # pydantic-settings Settings class, loaded from .env
│   ├── logging_config.py      # Console + rotating file logging setup
│   ├── models/
│   │   └── schemas.py          # PredictionInput / PredictionOutput / PredictionOutputV2 /
│   │                            # PredictionBatchInput / PredictionBatchOutput / ModelInfo
│   └── routers/
│       ├── v1.py                # /api/v1: health, predict, predict-batch, model-info
│       └── v2.py                # /api/v2: predict (probabilities + model_version)
├── ml/
│   ├── train.py                 # Model training script
│   ├── data/
│   │   └── iris_dataset.csv
│   ├── saved_model/
│   │   ├── model.joblib          # Trained, serialized model
│   │   └── metadata.json         # Model metadata served by /api/v1/model-info
│   └── test_model.py             # Script to reload model & verify predictions
├── tests/
│   ├── conftest.py               # Shared FastAPI TestClient fixture
│   ├── test_health.py
│   ├── test_predict.py
│   ├── test_predict_batch.py
│   ├── test_model_info.py
│   └── test_versioning.py        # Proves v1 and v2 return different-but-correct shapes
├── venv/
├── app.log                       # Runtime logs (request IDs, status, duration, predictions)
├── .env                          # Local config values (git-ignored)
├── .env.example                  # Template of required env vars (committed)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/muthuraja89109/ml-api-project.git
cd ml-api-project

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your local environment file
cp .env.example .env
# then edit .env with your local paths/values if needed
```

---

## 🚀 Usage

### Train and save the model
```bash
cd ml
python train.py
```
This trains the model on the Iris dataset and saves it to `ml/saved_model/model.joblib`.

### Verify the saved model
```bash
python test_model.py
```
Loads the saved model and runs a sample prediction to confirm it works.

### Run the API server
```bash
cd ..
uvicorn app.main:app --reload
```
The server starts at: **http://127.0.0.1:8000**. The model is loaded once at startup (not per request) via FastAPI's lifespan handler, and all configuration is read from `.env`.

### Try the endpoints
- Root/liveness check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Health check: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
- Interactive API docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Example `POST /api/v1/predict`:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
Response:
```json
{
  "prediction": "setosa",
  "confidence": 0.98,
  "request_id": "1ff6fd92-66dd-4d2f-8d03-e676cfc0d035"
}
```

**Example `POST /api/v2/predict`** (same input, richer response):
```json
{
  "prediction": "setosa",
  "confidence": 0.48,
  "request_id": "399dea79-27cd-49d3-bcc2-7b674ec82692",
  "probabilities": {
    "setosa": 0.48,
    "versicolor": 0.43,
    "virginica": 0.09
  },
  "model_version": "2.0.0"
}
```

**Example `POST /api/v1/predict-batch`:**
```json
{
  "inputs": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}
  ]
}
```
Returns a `predictions` array, one result per input. Batches outside `1`–`MAX_BATCH_SIZE` (configured in `.env`) are rejected with a `400` error.

Sending malformed input (missing field, wrong type, or a non-positive measurement) returns a `422 Unprocessable Entity` with a clear validation error instead of crashing the service. Unexpected internal failures return a clean `500` — never a raw Python traceback.

### Run the tests
```bash
python -m pytest -v
```
10 tests covering health, prediction (success + validation errors), batch limits, model info, and v1/v2 version compatibility.

### Logs
Every request and prediction is written to both the console and `app.log` (project root), tagged with a unique `request_id`:
```
2026-08-28 12:12:26,996 | INFO | Model loaded successfully
2026-08-28 12:13:10,136 | INFO | request_id=ebbe3d3e-... | Prediction successful | prediction=setosa | confidence=1.0000
2026-08-28 12:13:10,145 | INFO | request_id=ebbe3d3e-... | method=POST | path=/api/v1/predict | status=200 | duration=0.0655s
```

---

## ✅ Progress Log

| Task | Description | Status |
|------|-------------|--------|
| 1 | Project planning & dataset selection | ✅ Complete |
| 2 | Python environment & project structure setup | ✅ Complete |
| 3 | ML model training, evaluation & saving | ✅ Complete |
| 4 | FastAPI integration — `GET /` and `POST /predict`, Swagger testing | ✅ Complete |
| 5 | Load the real trained model into FastAPI at startup | ✅ Complete |
| 6 | Pydantic input validation (`PredictionInput`) | ✅ Complete |
| 7 | Milestone — core API: prediction + confidence + request ID, `/health` endpoint | ✅ Complete |
| 8 | Response models, status codes & error handling | ✅ Complete |
| 9 | Structured logging (console + file, request IDs) | ✅ Complete |
| 10 | API versioning — routes moved under `/api/v1` | ✅ Complete |
| 11 | Batch prediction (`/predict-batch`) and model metadata (`/model-info`) | ✅ Complete |
| 12 | Configuration management via `pydantic-settings` and `.env` | ✅ Complete |
| 13 | Automated pytest test suite, with break/restore proof | ✅ Complete |
| 14 | Milestone — non-breaking `/api/v2/predict`, tested against v1 | ✅ Complete |

### Tasks 1–9 — Foundation & Core Development
- Planned the project, set up the environment and folder structure, trained and saved the Iris model.
- Built a FastAPI app that loads the model once at startup and returns real predictions.
- Added Pydantic input validation, a `/health` endpoint, `response_model` enforcement, and structured console + file logging with request IDs.

### Task 10 — API Versioning
- Moved `/predict` and `/health` into `app/routers/v1.py` using `APIRouter(prefix="/api/v1")`.
- `main.py` now uses `include_router()` instead of defining routes directly.
- Documented the v1 → v2 plan directly as a code comment at the top of `v1.py`.

### Task 11 — Batch Prediction & Model Info
- Added `/api/v1/predict-batch`, predicting on the whole batch in one `model.predict_proba()` call (not a per-row loop) for efficiency.
- Added `/api/v1/model-info`, returning real metadata (`model_type`, `model_version`, `training_date`, `accuracy`, `features`) loaded from `metadata.json`.
- Both endpoints log batch size / duration, consistent with `/predict`.

### Task 12 — Configuration Management
- Added `app/config.py` with a `Settings(BaseSettings)` class loading `MODEL_PATH`, `METADATA_PATH`, `LOG_LEVEL`, `MAX_BATCH_SIZE`, `API_TITLE` from `.env`.
- `.env` is git-ignored; `.env.example` is the committed template.
- `MAX_BATCH_SIZE` is genuinely enforced — proven by a `400 "Batch size must be between 1 and N"` response on oversized batches.

### Task 13 — Automated Testing
- Built a 9-test pytest suite (`tests/`) covering `/health`, `/predict` (success + 422s), `/predict-batch` (oversized → 400), and `/model-info`.
- Proved the tests are meaningful: disabled the batch-size check, watched `test_predict_batch_oversized_rejected` genuinely **fail** (`assert 200 == 400`), then restored the code and confirmed all tests passed again.

### Task 14 — Milestone: `/api/v2/predict`
- Added `PredictionOutputV2` (adds `probabilities` and `model_version`) without touching the existing `PredictionOutput` used by v1.
- Built `app/routers/v2.py` on its own `APIRouter(prefix="/api/v2")`.
- Added `tests/test_versioning.py`, calling v1 and v2 with the same input and asserting v1's response shape is exactly unchanged while v2 returns the richer shape — proving, in code, that old clients are protected.
- Full suite now passes at **10/10**.

---

## 🛠️ Tech Stack

- **Python 3.12**
- **FastAPI** — web framework, with `APIRouter` for versioning
- **Uvicorn** — ASGI server
- **scikit-learn** — model training
- **joblib** — model serialization
- **pandas** — data handling
- **Pydantic** — request/response validation
- **pydantic-settings** — environment-variable-driven configuration
- **pytest** + **httpx** (`TestClient`) — automated testing
- **logging** (stdlib) — console + file logging with request correlation IDs

---

## 📅 What's Next

- **Task 15:** Package the application into a Docker container.
- Add CI (e.g. GitHub Actions) to run `pytest` automatically on every push.
- Consider deprecating v1 once request logs show traffic has shifted to v2.

---

## 👤 Author

**Muthuraja**
Python Intern — ML Model Deployment Project