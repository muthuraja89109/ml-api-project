# ML Model Deployment as a Monitored REST API

A machine learning model (Iris species classifier) served through a versioned, tested, containerized, and secured REST API built with **FastAPI**, as part of the Python Intern task program.

> **Status:** Tasks 1–17 complete ✅ (Foundation → Core Development → Feature Development → Advanced Development)

---

## 📌 Project Overview

This project trains a simple ML model and serves predictions through a REST API.

- **Dataset:** Iris flower dataset (`scikit-learn` built-in / CSV)
- **Problem type:** Classification — predict the Iris species from flower measurements
- **API contract:**
  - `POST /api/v1/predict` 🔒 — accepts flower measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) as a Pydantic-validated JSON body, and returns the predicted species, a confidence score, and a request ID.
  - `POST /api/v1/predict-batch` 🔒 — accepts a list of measurement sets (1–`MAX_BATCH_SIZE`) and returns a prediction for each, computed in a single batched model call.
  - `GET /api/v1/model-info` — returns real model metadata (type, version, training date, accuracy, feature names).
  - `GET /api/v1/health` — health check returning `{"status": "ok", "model_loaded": true}`.
  - `POST /api/v2/predict` 🔒 — same input as v1, but returns a richer response: full class **probability distribution** alongside `prediction` and `request_id`.
  - `GET /` — root/liveness check.

  🔒 = requires a valid `X-API-Key` header (see [Security](#-security)).

- **Versioning:** All endpoints are namespaced under `/api/v1/...` or `/api/v2/...` via `APIRouter`, so v1 and v2 can evolve independently without breaking existing clients.
- **Validation:** Request bodies validated with `PredictionInput` / `PredictionBatchInput` Pydantic models — all measurements required and must be greater than zero, and any unexpected extra fields are rejected (`extra="forbid"`).
- **Response contract:** Enforced with `response_model` (`PredictionOutput`, `PredictionV2Output`, `PredictionBatchOutput`, `ModelInfo`); errors return a clean, documented shape — never a raw traceback.
- **Configuration:** Driven entirely by environment variables via `pydantic-settings` — no hardcoded paths, limits, titles, or secrets.
- **Security:** Protected endpoints require an `X-API-Key` header (checked via a FastAPI dependency), and CORS is explicitly scoped to allowed origins only.
- **Observability:** Every request and prediction is logged (console + `app.log`) with a unique request ID, HTTP method, path, status code, and duration.
- **Testing:** A pytest suite (13 tests) covers success paths, validation errors, batch limits, version compatibility, and security edge cases — with proof that the tests catch real regressions.
- **Containerized:** Runs identically anywhere via Docker, orchestrated with Docker Compose — no local Python environment required.

---

## 🗂️ Project Structure

```
ml-api-project/
├── app/
│   ├── main.py               # FastAPI app entry point (lifespan model load, CORS, router registration)
│   ├── config.py              # pydantic-settings Settings class, loaded from .env
│   ├── logging_config.py      # Console + rotating file logging setup
│   ├── security.py            # API key verification dependency
│   ├── models/
│   │   └── schemas.py          # PredictionInput / PredictionOutput / PredictionV2Output /
│   │                            # PredictionBatchInput / PredictionBatchOutput / ModelInfo
│   └── routers/
│       ├── v1.py                # /api/v1: health, predict🔒, predict-batch🔒, model-info
│       └── v2.py                # /api/v2: predict🔒 (full probability distribution)
├── ml/
│   ├── train.py                 # Model training script
│   ├── data/
│   │   └── iris_dataset.csv
│   ├── saved_model/
│   │   ├── model.joblib          # Trained, serialized model
│   │   └── metadata.json         # Model metadata served by /api/v1/model-info
│   └── test_model.py             # Script to reload model & verify predictions
├── tests/
│   ├── conftest.py               # Shared TestClient fixture + auth_headers fixture
│   ├── test_health.py
│   ├── test_predict.py
│   ├── test_predict_batch.py
│   ├── test_model_info.py
│   ├── test_versioning.py        # Proves v1 and v2 return different-but-correct shapes
│   └── test_security.py          # Missing/invalid API key, extra-field rejection
├── venv/
├── app.log                       # Runtime logs (request IDs, status, duration, predictions)
├── Dockerfile                     # Container image definition
├── .dockerignore                  # Excludes venv/, .git/, __pycache__/, .env from the image
├── docker-compose.yml              # Single-command orchestration (build, port, env, model volume)
├── requirements-docker.txt         # Pinned dependencies used inside the container
├── .env                            # Local config values, incl. API_KEY (git-ignored)
├── .env.example                    # Template of required env vars (committed, no real secrets)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation (local, without Docker)

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
# then edit .env and set a real API_KEY (see Security section below)
```

---

## 🐳 Run with Docker Compose (recommended)

### Prerequisites
- Docker Desktop installed and running

### Steps
```bash
git clone https://github.com/muthuraja89109/ml-api-project.git
cd ml-api-project

# Windows PowerShell
Copy-Item .env.example .env
# macOS/Linux
cp .env.example .env

# edit .env and set a real API_KEY, e.g.:
# python -c "import secrets; print(secrets.token_hex(32))"

docker compose up --build
```

Open:
- Swagger docs: **http://localhost:8000/docs**
- Health check: **http://localhost:8000/api/v1/health**

### Other useful commands
```bash
docker compose up -d        # run in the background
docker compose logs -f      # view live logs
docker compose down         # stop and remove the container
```

The `ml/saved_model/` folder is mounted into the container as a **volume**. You can drop in a retrained `model.joblib` locally and restart the container to pick it up — no image rebuild needed.

### Run with plain Docker (without Compose)
```bash
docker build -t ml-api-project .
docker run -d --name ml-api-container -p 8000:8000 --env-file .env ml-api-project
```

---

## 🚀 Usage

### Train and save the model
```bash
cd ml
python train.py
```
Trains the model on the Iris dataset and saves it to `ml/saved_model/model.joblib`.

### Run the API server (local, non-Docker)
```bash
uvicorn app.main:app --reload
```
The server starts at **http://127.0.0.1:8000**. The model is loaded once at startup (not per request) via FastAPI's lifespan handler, and all configuration is read from `.env`.

### Try the endpoints

**`GET /api/v1/health`** — no API key required:
```json
{"status": "ok", "model_loaded": true}
```

**`POST /api/v1/predict`** 🔒 — requires header `X-API-Key: <your key>`:
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

**`POST /api/v2/predict`** 🔒 — same input, richer response:
```json
{
  "prediction": "setosa",
  "probabilities": {
    "setosa": 0.48,
    "versicolor": 0.43,
    "virginica": 0.09
  },
  "request_id": "399dea79-27cd-49d3-bcc2-7b674ec82692"
}
```

**`POST /api/v1/predict-batch`** 🔒:
```json
{
  "inputs": [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}
  ]
}
```
Returns a `predictions` array, one result per input. Batches outside `1`–`MAX_BATCH_SIZE` (configured in `.env`) are rejected with `400`.

Sending malformed input (missing field, wrong type, non-positive measurement, or an **unexpected extra field**) returns a `422 Unprocessable Entity` with a clear validation error. Requests missing or using an invalid `X-API-Key` return `401 Unauthorized`. Unexpected internal failures return a clean `500` — never a raw Python traceback.

### Run the tests
```bash
python -m pytest -v
```
**13 tests** covering health, prediction (success + validation errors), batch limits, model info, v1/v2 version compatibility, and security (missing/invalid API key, rejected extra fields).

### Logs
Every request and prediction is written to both the console and `app.log` (project root), tagged with a unique `request_id`:
```
2026-08-28 12:12:26,996 | INFO | Model loaded successfully
2026-08-28 12:13:10,136 | INFO | request_id=ebbe3d3e-... | Prediction successful | prediction=setosa | confidence=1.0000
2026-08-28 12:13:10,145 | INFO | request_id=ebbe3d3e-... | method=POST | path=/api/v1/predict | status=200 | duration=0.0655s
```

---

## 🔐 Security

- **API key auth:** `/api/v1/predict`, `/api/v1/predict-batch`, and `/api/v2/predict` all require a valid `X-API-Key` header, verified via a FastAPI dependency (`app/security.py`). Missing or incorrect keys return `401 Unauthorized`. `/health` and `/model-info` remain open for monitoring tools.
- **CORS:** Explicitly configured via `CORSMiddleware` with a defined `allow_origins` list — never left wide open by default.
- **Strict input schemas:** `PredictionInput` and `PredictionBatchInput` use `model_config = ConfigDict(extra="forbid")`, so any unexpected field in a request is rejected with `422` instead of being silently ignored.
- **Secrets management:** `API_KEY` is loaded from `.env` via `pydantic-settings` and has **no default value** — the app refuses to start without one configured, preventing an accidental unprotected deployment. Never commit `.env`; only `.env.example` (with a placeholder) is tracked in git.

To generate a strong API key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
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
| 15 | Containerize the API with Docker | ✅ Complete |
| 16 | Orchestrate with Docker Compose | ✅ Complete |
| 17 | Security, validation edge cases, and robustness | ✅ Complete |

### Tasks 1–9 — Foundation & Core Development
Planned the project, set up the environment and folder structure, trained and saved the Iris model, and built a FastAPI app that loads the model once at startup and returns real predictions — with Pydantic validation, a `/health` endpoint, `response_model` enforcement, and structured console + file logging with request IDs.

### Task 10 — API Versioning
Moved `/predict` and `/health` into `app/routers/v1.py` using `APIRouter(prefix="/api/v1")`. `main.py` uses `include_router()` instead of defining routes directly. The v1 → v2 plan is documented as a code comment at the top of `v1.py`.

### Task 11 — Batch Prediction & Model Info
Added `/api/v1/predict-batch`, predicting on the whole batch in one `model.predict_proba()` call (not a per-row loop) for efficiency, and `/api/v1/model-info`, returning real metadata loaded from `metadata.json`. Both endpoints log batch size / duration, consistent with `/predict`.

### Task 12 — Configuration Management
Added `app/config.py` with a `Settings(BaseSettings)` class loading all configuration from `.env`. `.env` is git-ignored; `.env.example` is the committed template. `MAX_BATCH_SIZE` is genuinely enforced — proven by a `400` response on oversized batches.

### Task 13 — Automated Testing
Built a pytest suite covering `/health`, `/predict` (success + 422s), `/predict-batch` (oversized → 400), and `/model-info`. Proved the tests are meaningful: disabled the batch-size check, watched the relevant test genuinely **fail** (`assert 200 == 400`), then restored the code and confirmed all tests passed again.

### Task 14 — Milestone: `/api/v2/predict`
Added `PredictionV2Output` (full probability distribution) without touching the existing `PredictionOutput` used by v1. Built `app/routers/v2.py` on its own `APIRouter(prefix="/api/v2")`. `tests/test_versioning.py` calls v1 and v2 with the same input and asserts v1's response shape is exactly unchanged while v2 returns the richer shape — proving, in code, that old clients are protected.

### Task 15 — Containerize with Docker
Wrote a `Dockerfile` (Python 3.11-slim base, dependencies installed, app copied in, `EXPOSE 8000`). `CMD` uses `uvicorn --host 0.0.0.0` — documented in a code comment explaining that `0.0.0.0` is required so Docker can forward the host's port into the container, since `127.0.0.1` would only accept connections from inside the container itself. `.dockerignore` excludes `venv/`, `.git/`, `__pycache__/`, and `.env`. Build and run verified: `docker build` completes cleanly, and `/docs` + `/api/v1/predict` were confirmed working live from inside the running container.

### Task 16 — Orchestrate with Docker Compose
Wrote `docker-compose.yml` defining an `api` service that builds from the `Dockerfile`, maps port `8000:8000`, loads all configuration via `env_file: .env` (nothing hardcoded in the compose file), and mounts `ml/saved_model/` as a volume so a retrained model can be swapped in without rebuilding the image. `docker compose up --build` verified working end-to-end — `/docs`, `/api/v1/predict`, and `/api/v2/predict` all confirmed responding correctly through Compose.

### Task 17 — Security, Validation Edge Cases & Robustness
Added `X-API-Key` header authentication via a FastAPI dependency (`app/security.py`), protecting `/predict`, `/predict-batch`, and `/api/v2/predict` while leaving `/health` and `/model-info` open. Configured `CORSMiddleware` with an explicit `allow_origins` list. Locked down `PredictionInput` and `PredictionBatchInput` with `extra="forbid"` so unexpected fields are rejected with `422`. Added `tests/test_security.py` (missing API key → 401, invalid API key → 401, extra field → 422) and updated all existing tests to include the API key header. Full suite verified at **13/13 passing**, plus manual end-to-end proof in Swagger: no key → `401`, wrong key → `401`, correct key → `200` with a real prediction.

---

## 🛠️ Tech Stack

- **Python 3.12** (3.11-slim inside Docker)
- **FastAPI** — web framework, with `APIRouter` for versioning and `CORSMiddleware` for CORS
- **Uvicorn** — ASGI server
- **scikit-learn** — model training
- **joblib** — model serialization
- **pandas** — data handling
- **Pydantic** — request/response validation, `extra="forbid"` for strict schemas
- **pydantic-settings** — environment-variable-driven configuration and secrets
- **pytest** + **httpx** (`TestClient`) — automated testing
- **logging** (stdlib) — console + file logging with request correlation IDs
- **Docker** + **Docker Compose** — containerization and orchestration

---

## 📅 What's Next

- **Task 18:** Add a `/metrics` endpoint for real monitoring (Prometheus-style).
- Add CI (e.g. GitHub Actions) to run `pytest` automatically on every push.
- Consider deprecating v1 once request logs show traffic has shifted to v2.
- Rotate the `API_KEY` periodically and consider per-client keys if usage grows.

---

## 👤 Author

**Muthuraja**
Python Intern — ML Model Deployment Project