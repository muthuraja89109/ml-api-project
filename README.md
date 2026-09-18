# ML Model Deployment as a Monitored REST API

A machine learning model (Iris species classifier) served through a versioned, tested, containerized, secured, and monitored REST API built with **FastAPI**.

This README is written so that anyone new to the project — with zero prior context — can understand what this is, set it up, run it, test it, and extend it, without asking the original author a single question.

> **Status:** Tasks 1–20 complete ✅ — project finished
>
> **Live demo:** https://ml-api-project-8mja.onrender.com/docs
> (hosted on Render's free tier — the first request after a period of inactivity may take 30–60 seconds while the service wakes up; subsequent requests are fast)

---

## ✨ Key Features

- 🌸 Iris species classification using a trained `RandomForestClassifier`
- 🚀 FastAPI REST API with versioned `/api/v1` and `/api/v2` routes
- 🔐 `X-API-Key` authentication for prediction endpoints
- ✅ Strict Pydantic validation with `extra="forbid"`
- 📦 Batch prediction with configurable `MAX_BATCH_SIZE`
- 📊 Prometheus metrics and a dedicated Prometheus container
- 🧪 Unit, integration, and load testing
- 📝 Structured console + rotating file logging with request IDs
- 🐳 Docker + Docker Compose support
- ⚙️ Environment-based configuration with `pydantic-settings`
- 🔄 GitHub Actions CI — full test suite runs automatically on every push
- 🌍 Deployed and publicly reachable (Render)

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Quick Start (Docker Compose — recommended)](#quick-start-docker-compose--recommended)
6. [Quick Start (Local, without Docker)](#quick-start-local-without-docker)
7. [Configuration (Environment Variables)](#configuration-environment-variables)
8. [Authentication](#authentication)
9. [API Reference](#api-reference)
10. [Monitoring with Prometheus](#monitoring-with-prometheus)
11. [Testing](#testing)
12. [Continuous Integration](#continuous-integration)
13. [Logging](#logging)
14. [Troubleshooting](#troubleshooting)
15. [Progress Log (Tasks 1–20)](#progress-log-tasks-1-20)
16. [Tech Stack](#tech-stack)
17. [What I Learned](#what-i-learned)
18. [Self-Assessment](#self-assessment)
19. [Independent Extension](#independent-extension)
20. [Author](#author)

---

## What This Project Is

This project trains a simple machine learning model (a classifier that predicts Iris flower species from four measurements) and serves it through a production-style REST API. It is **not** just a model wrapped in a script — it's built the way a real, small production service would be:

- Real predictions from a trained model (not hardcoded responses)
- Input validated automatically (bad data is rejected with clear errors, never silently accepted)
- Two API versions running side by side (`/api/v1` and `/api/v2`) so the API contract can evolve without breaking existing clients
- All configuration (paths, limits, secrets) comes from environment variables — nothing is hardcoded
- Every request and prediction is logged, with a unique ID per request for traceability
- Protected with API-key authentication and explicit CORS rules
- Fully containerized with Docker and Docker Compose — runs identically on any machine
- Instrumented with Prometheus metrics for live operational monitoring
- Covered by unit tests, integration tests (against the real running container), and a basic load test
- Automatically tested on every push via GitHub Actions CI
- Deployed and publicly reachable

**Dataset:** Iris flower dataset (`scikit-learn` built-in / CSV)
**Problem type:** Classification — predict the Iris species from `sepal_length`, `sepal_width`, `petal_length`, `petal_width`

---

## Architecture

```
                     +-------------------------+
   curl / Swagger    |                         |
   / any HTTP client |      Docker network     |
        |            |                         |
        v            |   +-----------------+   |
  +-----------+      |   |   api container |   |
  |  Client   |------+-->|   (FastAPI +    |   |
  +-----------+      |   |    Uvicorn)     |   |
        |            |   +--------+--------+   |
        |            |            |            |
        |  X-API-Key |            v            |
        |  header    |   +-----------------+   |
        +----------->|   |  Request flow:  |   |
                     |   |  1. CORS check  |   |
                     |   |  2. API key     |   |
                     |   |     check       |   |
                     |   |  3. Pydantic    |   |
                     |   |     validation  |   |
                     |   |  4. Route to    |   |
                     |   |     v1 or v2    |   |
                     |   |  5. Model       |   |
                     |   |     inference   |   |
                     |   |  6. Log request |   |
                     |   |     + metric    |   |
                     |   |  7. JSON        |   |
                     |   |     response    |   |
                     |   +--------+--------+   |
                     |            |            |
                     |   +--------+--------+   |
                     |   |  app.log file   |   |
                     |   |  (mounted out)  |   |
                     |   +-----------------+   |
                     |                         |
                     |   +-----------------+   |
                     |   | prometheus      |   |
                     |   | container       |<--+-- scrapes GET /metrics
                     |   | (localhost:9090)|   |   every 5 seconds
                     |   +-----------------+   |
                     +-------------------------+

   GitHub push --> GitHub Actions CI --> runs full pytest suite
                                          (fails the build if anything breaks)

   GitHub push --> Render --> rebuilds Docker image --> redeploys live URL
```

At startup, the API loads the trained model (`model.joblib`) once into memory via FastAPI's `lifespan` handler — it is **not** reloaded from disk on every request.

---

## Project Structure

```
ml-api-project/
|-- .github/
|   `-- workflows/
|       `-- tests.yml               # GitHub Actions CI: runs pytest on every push
|-- app/
|   |-- main.py                     # FastAPI app entry point: lifespan model load,
|   |                                  CORS, Prometheus instrumentation, router registration
|   |-- config.py                   # pydantic-settings Settings class, loaded from .env
|   |-- logging_config.py           # Console + rotating file logging setup
|   |-- security.py                 # X-API-Key header verification dependency
|   |-- models/
|   |   `-- schemas.py               # PredictionInput / PredictionOutput / PredictionV2Output /
|   |                                 # PredictionBatchInput / PredictionBatchOutput / ModelInfo
|   `-- routers/
|       |-- v1.py                     # /api/v1: health, predict, predict-batch, model-info
|       `-- v2.py                     # /api/v2: predict (full probability distribution)
|-- ml/
|   |-- train.py                      # Model training script
|   |-- data/
|   |   `-- iris_dataset.csv
|   |-- saved_model/
|   |   |-- model.joblib                # Trained, serialized model
|   |   `-- metadata.json               # Model metadata served by /api/v1/model-info
|   `-- test_model.py                   # Script to reload model & verify predictions
|-- tests/
|   |-- conftest.py                     # Shared TestClient fixture + auth_headers fixture
|   |-- test_health.py                   # Unit tests (in-process, fast)
|   |-- test_predict.py
|   |-- test_predict_batch.py
|   |-- test_model_info.py
|   |-- test_versioning.py               # Proves v1 and v2 return different-but-correct shapes
|   |-- test_security.py                 # Missing/invalid API key, extra-field rejection
|   `-- test_integration.py              # Integration tests -- real HTTP against the running container
|-- load_test.py                        # Standalone load-testing script (NOT a pytest file)
|-- venv/                               # Local Python virtual environment (not committed)
|-- app.log                             # Runtime logs (git-ignored)
|-- Dockerfile                           # Container image definition
|-- .dockerignore                        # Excludes venv/, .git/, __pycache__/, .env from the image
|-- docker-compose.yml                    # Orchestrates the api + prometheus containers
|-- prometheus.yml                        # Prometheus scrape configuration
|-- requirements-docker.txt               # Pinned dependencies used inside the container
|-- requirements.txt                      # Dependencies for local (non-Docker) development
|-- .env                                  # Local config values, incl. API_KEY (git-ignored, never commit)
|-- .env.example                          # Template of required env vars (committed, no real secrets)
|-- .gitignore
|-- TESTING.md                            # Integration/load testing process, bugs found & fixed
`-- README.md                             # This file
```

---

## Prerequisites

- **Docker Desktop** (recommended path) — includes Docker Engine and Docker Compose
- *or*, for local (non-Docker) development: **Python 3.11+** and `pip`

No other software is required. You do not need to install scikit-learn, FastAPI, or anything else on your host machine if you use Docker. You also don't need to install anything at all if you just want to try the [live demo](https://ml-api-project-8mja.onrender.com/docs).

---

## Quick Start (Docker Compose — recommended)

**Don't want to run anything locally?** Try the live deployment instead: **https://ml-api-project-8mja.onrender.com/docs**. Note: on the free tier, the service spins down after inactivity, so the first request may take 30–60 seconds while it wakes up.

This is the fastest, most reliable way to run the whole project locally — it starts both the API and a Prometheus monitoring container together, with zero local Python setup required.

```bash
# 1. Clone the repository
git clone https://github.com/muthuraja89109/ml-api-project.git
cd ml-api-project

# 2. Create your local environment file from the template
# Windows PowerShell:
Copy-Item .env.example .env
# macOS/Linux:
cp .env.example .env

# 3. Open .env and set a real API_KEY (see "Authentication" section below
#    for how to generate one). The app will refuse to start without it.

# 4. Build and start everything
docker compose up --build
```

Wait until you see this in the terminal output:
```
Model loaded successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

You now have two services running:

- API Swagger docs: http://localhost:8000/docs — interactive documentation, try every endpoint from the browser
- API health check: http://localhost:8000/api/v1/health — confirm the service and model are up
- API raw metrics: http://localhost:8000/metrics — Prometheus-format operational metrics
- Prometheus UI: http://localhost:9090 — query and graph the API's metrics over time

### Useful Compose commands
```bash
docker compose up -d        # run in the background (detached)
docker compose logs -f      # stream live logs from all containers
docker compose ps           # see which containers are running
docker compose down         # stop and remove all containers
```

### Swapping in a retrained model without rebuilding
The `ml/saved_model/` folder is mounted into the container as a **volume**. If you retrain the model and replace `model.joblib` locally, just restart the container (`docker compose restart`) — no image rebuild needed.

---

## Quick Start (Local, without Docker)

Use this only if you specifically need to run the app outside a container (e.g. for debugging with an IDE debugger attached).

```bash
# 1. Clone and enter the repo
git clone https://github.com/muthuraja89109/ml-api-project.git
cd ml-api-project

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your environment file
cp .env.example .env
# then edit .env and set a real API_KEY

# 5. (If not already trained) Train and save the model
cd ml
python train.py
cd ..

# 6. Run the server
uvicorn app.main:app --reload
```

The server starts at **http://127.0.0.1:8000**. Note: Prometheus monitoring (the second container) is only available via the Docker Compose path, not this local path.

---

## Configuration (Environment Variables)

All configuration is driven by environment variables, loaded from a `.env` file via `pydantic-settings`. **Never commit `.env`** — only `.env.example` (with placeholder values) is tracked in git.

- `MODEL_PATH` — Path to the trained model file. Example: `ml/saved_model/model.joblib`
- `METADATA_PATH` — Path to the model metadata JSON file. Example: `ml/saved_model/metadata.json`
- `LOG_LEVEL` — Logging verbosity. Example: `INFO`
- `MAX_BATCH_SIZE` — Maximum number of items allowed in one `/predict-batch` request. Example: `100`
- `API_TITLE` — Title shown in the Swagger docs. Example: `ML Prediction API`
- `API_KEY` — Secret key required to call protected endpoints. Has no default — the app will not start without this set.

To generate a strong `API_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Paste the output into your `.env` file as the value of `API_KEY`.

---

## Authentication

Some endpoints require a valid API key, sent as a header on every request:
```
X-API-Key: <your key from .env>
```

Endpoints that do **NOT** require a key: `GET /`, `GET /api/v1/health` (left open so monitoring tools can poll it freely), `GET /api/v1/model-info`, `GET /metrics` (Prometheus needs to scrape this without a key).

Endpoints that **DO** require a key: `POST /api/v1/predict`, `POST /api/v1/predict-batch`, `POST /api/v2/predict`.

Requests to a protected endpoint without a valid key return `401 Unauthorized`:
```json
{"detail": "Missing API key"}
```
or
```json
{"detail": "Invalid API key"}
```

CORS is also explicitly configured (`CORSMiddleware` with a defined `allow_origins` list) — browser-based cross-origin requests are only allowed from origins you've explicitly listed in `app/main.py`, never left open by default.

---

## API Reference

Full interactive documentation (with a "Try it out" button for every endpoint) is always available at **http://localhost:8000/docs** (or the [live demo](https://ml-api-project-8mja.onrender.com/docs)) once the server is running. The reference below is for quick copy-paste use.

### Root / liveness check
```bash
curl http://localhost:8000/
```
```json
{"message": "ML API is alive"}
```

### Health check
```bash
curl http://localhost:8000/api/v1/health
```
```json
{"status": "ok", "model_loaded": true}
```

### Model info
```bash
curl http://localhost:8000/api/v1/model-info
```
```json
{
  "model_type": "RandomForestClassifier",
  "model_version": "1.0.0",
  "training_date": "2026-09-01 12:58:06",
  "accuracy": 0.9,
  "features": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
}
```

### Single prediction — v1
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your key>" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```
```json
{
  "prediction": "setosa",
  "confidence": 0.98,
  "request_id": "1ff6fd92-66dd-4d2f-8d03-e676cfc0d035"
}
```

### Single prediction — v2 (full probability distribution)
```bash
curl -X POST http://localhost:8000/api/v2/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your key>" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```
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
`/api/v1/predict` is never modified when `/api/v2/predict` changes — v1's response shape is guaranteed stable for existing clients (see `tests/test_versioning.py`, which asserts this in code).

### Batch prediction (v1)
```bash
curl -X POST http://localhost:8000/api/v1/predict-batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your key>" \
  -d '{
    "inputs": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3}
    ]
  }'
```
```json
{
  "predictions": [
    {"prediction": "setosa", "confidence": 1.0, "request_id": "..."},
    {"prediction": "virginica", "confidence": 0.99, "request_id": "..."}
  ]
}
```
A batch of 0 items, or more than `MAX_BATCH_SIZE` items (configured in `.env`), is rejected with `400`:
```json
{"detail": "Batch size must be between 1 and 100"}
```

> Note: v2 currently only supports single predictions (`/api/v2/predict`), not batch. A `/api/v2/predict-batch` endpoint mirroring v1's batch behaviour with the richer v2 response shape is a natural next extension, not yet built.

### Error responses

- Missing a required field → `422`, body like `{"detail": [{"loc": [...], "msg": "Field required", ...}]}`
- Wrong field type (e.g. string instead of number) → `422`, same shape as above
- Extra, unexpected field in the request → `422` — schemas use `extra="forbid"`, unknown fields are rejected, not silently ignored
- Non-positive measurement (e.g. `-1`) → `422` — fields are constrained with `gt=0`
- Missing or wrong `X-API-Key` → `401`, `{"detail": "Missing API key"}` or `{"detail": "Invalid API key"}`
- Unexpected internal failure → `500`, `{"detail": "Prediction failed"}` — never a raw Python traceback

### Raw Prometheus metrics
```bash
curl http://localhost:8000/metrics
```
Returns plain-text Prometheus exposition format — see the next section for details.

---

## Monitoring with Prometheus

The API is instrumented with `prometheus-fastapi-instrumentator`, which automatically tracks:
- Request counts by endpoint, method, and status code (`http_requests_total`)
- Request latency histograms (`http_request_duration_seconds`)
- Request/response payload sizes

**Custom metric:** `ml_predictions_total{predicted_class="..."}` — a counter that increments every time a v1 prediction is made, labeled by the predicted species. This lets you see, over time, how many `setosa` vs `versicolor` vs `virginica` predictions the service has served.

### Viewing metrics directly
```
http://localhost:8000/metrics
```

### Viewing metrics in Prometheus's own UI (via Docker Compose)
When running via `docker compose up`, a second container runs the official Prometheus server, pre-configured (via `prometheus.yml`) to scrape the API's `/metrics` endpoint every 5 seconds.

1. Open **http://localhost:9090**
2. Go to **Status → Targets** — you should see the `ml-api` target listed as **UP**
3. Go to **Query**, type `ml_predictions_total`, click **Execute** — see live data from real predictions made against the API

---

## Testing

There are three distinct layers of testing in this project — each catches different kinds of problems.

### 1. Unit tests (fast, in-process, no server needed)
Run with FastAPI's `TestClient`, which calls the app directly in-process — no real network, no Docker required.

```bash
python -m pytest -v
```
Covers: health check shape, prediction success/failure paths, batch size limits, model metadata, API-key enforcement, extra-field rejection, and v1/v2 response-shape compatibility. This suite runs automatically on every push via CI, and should also be run manually before every commit.

### 2. Integration tests (real HTTP, against the running Docker container)
These call the **actual running container** over real HTTP — they prove the whole system (Docker, config, networking, the real model file) works together, not just the application code in isolation.

```bash
# Terminal 1 — start the stack and leave it running
docker compose up --build

# Terminal 2 — once you see "Application startup complete.", run:
python -m pytest tests/test_integration.py -v
```
**Important:** these tests will fail with a `ConnectError` if the container isn't already running — that's expected and correct behavior, not a bug in the test. This is also why `test_integration.py` is deliberately excluded from the CI workflow (GitHub Actions doesn't run the Docker Compose stack).

### 3. Load testing (concurrency / performance)
A standalone script (not a pytest file — run it directly) that fires many concurrent requests at `/predict` and reports success/failure counts and response-time statistics.

```bash
# with docker compose up --build still running in another terminal:
python load_test.py
```

See **`TESTING.md`** for the full write-up of what was tested, what broke (a startup crash was found and fixed, and a concurrency performance bottleneck was identified and partially mitigated), and the before/after results.

---

## Continuous Integration

Every push to `main` (and every pull request) automatically triggers a GitHub Actions workflow (`.github/workflows/tests.yml`) that:

1. Checks out the repository on a clean Ubuntu runner
2. Sets up Python 3.12
3. Installs dependencies from `requirements.txt`
4. Builds a `.env` file from a repository secret (`API_KEY`)
5. Runs the full unit test suite (`test_integration.py` excluded, since it needs a live Docker container CI doesn't run)

This means a broken commit is caught automatically, before it's ever manually verified — the same safety net a real engineering team relies on before merging code. See the **Actions** tab on the GitHub repository for the run history.

---

## Logging

Every request and prediction is logged to both the console and a rotating file (`app.log`, in the project root), tagged with a unique `request_id` so a single request's full journey can be traced:

```
2026-08-28 12:12:26,996 | INFO | Model loaded successfully
2026-08-28 12:13:10,136 | INFO | request_id=ebbe3d3e-... | Prediction successful | prediction=setosa | confidence=1.0000
2026-08-28 12:13:10,145 | INFO | request_id=ebbe3d3e-... | method=POST | path=/api/v1/predict | status=200 | duration=0.0655s
```

`LOG_LEVEL` in `.env` controls verbosity (`INFO`, `DEBUG`, `WARNING`, etc).

---

## Troubleshooting

**"failed to connect to the docker API... npipe" (Windows)**
Docker Desktop's engine isn't running. Open Docker Desktop from the Start menu (or run `docker desktop start`), wait 30–60 seconds, then confirm with `docker info` before retrying.

**Container starts but every request returns `401`**
You haven't set `X-API-Key` on the request, or the value doesn't match `API_KEY` in your `.env`. Double-check there's no extra text in the header value (it should be *only* the key itself, not `API_KEY=<value>`).

**App won't start at all: `pydantic_settings.exceptions... API_KEY field required`**
`API_KEY` has no default value by design — you must set it in `.env` before the app will boot. This is intentional, so the API can never accidentally run unprotected.

**Integration tests fail with `httpx.ConnectError` / "connection actively refused"**
The Docker container isn't running (or isn't reachable yet). Run `docker compose up --build` in a separate terminal first, confirm `curl http://localhost:8000/api/v1/health` returns `200`, then re-run the integration tests.

**`InconsistentVersionWarning` about scikit-learn in the container logs**
The scikit-learn version used to train the model didn't exactly match the version installed in the container. Fixed by pinning exact versions in `requirements-docker.txt` to match the local training environment.

**`numpy==X.X.X` not found while building the Docker image**
This happens if `requirements-docker.txt` is pinned to a numpy version that requires a newer Python than the Dockerfile's base image provides. Fix by either aligning the `Dockerfile`'s Python version with the one used to freeze `requirements.txt`, or letting `numpy` resolve automatically (unpinned) rather than pinning it exactly.

**PowerShell `curl` command not working the way you expect**
PowerShell aliases `curl` to `Invoke-WebRequest`, which uses different syntax (`-Headers @{...}` instead of `-H`, backtick `` ` `` instead of `\` for line continuation). Use `curl.exe` explicitly to get real curl behavior, or use the Swagger UI at `/docs` instead.

---

## Progress Log (Tasks 1-20)

Live public deployment: **https://ml-api-project-8mja.onrender.com**

- Task 1 — Project planning & dataset selection — Complete
- Task 2 — Python environment & project structure setup — Complete
- Task 3 — ML model training, evaluation & saving — Complete
- Task 4 — Basic FastAPI app: GET /, POST /predict, Swagger testing — Complete
- Task 5 — Load the real trained model into FastAPI at startup — Complete
- Task 6 — Pydantic input validation — Complete
- Task 7 — Milestone: core API with prediction + confidence + request ID, /health — Complete
- Task 8 — Response models, status codes & error handling — Complete
- Task 9 — Structured logging (console + file, request IDs) — Complete
- Task 10 — API versioning: routes moved under /api/v1 — Complete
- Task 11 — Batch prediction and model metadata endpoints — Complete
- Task 12 — Configuration management via pydantic-settings and .env — Complete
- Task 13 — Automated pytest test suite, with break/restore proof — Complete
- Task 14 — Milestone: non-breaking /api/v2/predict, tested against v1 — Complete
- Task 15 — Containerize the API with Docker — Complete
- Task 16 — Orchestrate with Docker Compose — Complete
- Task 17 — Security: API-key auth, CORS, strict schema validation — Complete
- Task 18 — Prometheus /metrics endpoint + custom metric + Prometheus container — Complete
- Task 19 — Integration testing, load testing, bug found and fixed (TESTING.md) — Complete
- Task 20 — Final polish, deployment, documentation, independent extension — **Complete**

---

## How to Use / Verify Each Task

This section maps every task directly to a concrete command or request you can run yourself to see that task's work in action — useful for reviewing the project feature-by-feature rather than reading code.

**Task 1–2 — Planning & project structure**
```bash
git clone https://github.com/muthuraja89109/ml-api-project.git
cd ml-api-project
ls          # inspect the app/, ml/, tests/ folder layout described above
cat README.md
```

**Task 3 — Model training & saving**
```bash
cd ml
python train.py              # trains and saves ml/saved_model/model.joblib
python test_model.py         # reloads it and runs one sample prediction
cd ..
```

**Task 4 — Basic FastAPI app**
```bash
uvicorn app.main:app --reload
# open http://127.0.0.1:8000/docs and try GET / from the browser
```

**Task 5 — Real model loaded at startup**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" -H "X-API-Key: <your key>" \
  -d '{"sepal_length":1,"sepal_width":1,"petal_length":1,"petal_width":1}'
# then try again with different numbers — the prediction genuinely changes
```

**Task 6 — Pydantic input validation**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" -H "X-API-Key: <your key>" \
  -d '{"sepal_length":"not-a-number","sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# returns 422, not a crash
```

**Task 7 — Confidence + request ID + /health**
```bash
curl http://localhost:8000/api/v1/health
# {"status": "ok", "model_loaded": true}
```

**Task 8 — Response models & error handling**
Open `/docs` → expand `POST /api/v1/predict` → the `Schema` tab shows the enforced `PredictionOutput` response shape, and the `422`/`500` response examples show the guaranteed error shapes.

**Task 9 — Structured logging**
```bash
tail -f app.log
# make a prediction in another terminal/tab and watch the log line with its request_id appear
```

**Task 10–11 — Versioning, batch, model-info**
```bash
curl http://localhost:8000/api/v1/model-info
curl -X POST http://localhost:8000/api/v1/predict-batch \
  -H "Content-Type: application/json" -H "X-API-Key: <your key>" \
  -d '{"inputs":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}]}'
```

**Task 12 — Configuration via .env**
```bash
cat .env.example      # see every variable the app needs
# change MAX_BATCH_SIZE in your own .env, restart, and send a batch that now
# exceeds the new limit — confirm it's rejected with 400
```

**Task 13 — Automated tests**
```bash
python -m pytest -v          # 17 passed
```

**Task 14 — v1/v2 milestone**
```bash
python -m pytest tests/test_versioning.py -v
curl -X POST http://localhost:8000/api/v2/predict \
  -H "Content-Type: application/json" -H "X-API-Key: <your key>" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# compare the response shape to the v1 call in Task 5 — different, both correct
```

**Task 15–16 — Docker & Docker Compose**
```bash
docker compose up --build
docker compose ps             # see both containers running
curl http://localhost:8000/api/v1/health
```

**Task 17 — Security**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
# 401, no X-API-Key sent — try again adding -H "X-API-Key: <your key>" for 200
```

**Task 18 — Prometheus monitoring**
```bash
curl http://localhost:8000/metrics | grep ml_predictions_total
# open http://localhost:9090 -> Status -> Targets -> confirm "ml-api" is UP
```

**Task 19 — Integration & load testing**
```bash
python -m pytest tests/test_integration.py -v     # needs docker compose up running
python load_test.py                                 # fires 100 concurrent requests
cat TESTING.md                                       # full write-up of what was found
```

**Task 20 — Deployment & CI**
```bash
# no local setup needed at all:
open https://ml-api-project-8mja.onrender.com/docs
# check the Actions tab on GitHub to see the CI workflow run history
```

---

## Tech Stack

- **Python 3.12** (locally); Docker base image aligned to match
- **FastAPI** — web framework, with `APIRouter` for versioning and `CORSMiddleware` for CORS
- **Uvicorn** — ASGI server (run with `--workers 2` in the container for better concurrency)
- **scikit-learn** — model training (`RandomForestClassifier`)
- **joblib** — model serialization
- **pandas** — data handling
- **Pydantic** — request/response validation, `extra="forbid"` for strict schemas
- **pydantic-settings** — environment-variable-driven configuration and secrets
- **pytest** + **httpx** (`TestClient` for unit tests, direct `httpx` for integration tests) — automated testing
- **prometheus-fastapi-instrumentator** + **prometheus_client** — monitoring
- **logging** (stdlib) — console + file logging with request correlation IDs
- **Docker** + **Docker Compose** — containerization and orchestration
- **GitHub Actions** — continuous integration
- **Render** — deployment / hosting

---

## What I Learned

Building this project end-to-end, in order, actually clarified something that's easy to miss when just following tutorials: almost every "production" concept exists to solve one specific, concrete problem, not as abstract best practice for its own sake.

- **Versioning isn't bureaucracy — it's a promise.** Once `/api/v1/predict` exists and a client depends on it, changing its shape breaks that client. `/api/v2/predict` exists purely so improvements don't come at the cost of breaking what already works.
- **Integration and load testing catch a completely different category of bug than unit tests.** My 17 unit tests all passed the whole time a real startup-crashing `NameError` existed in the codebase — because unit tests never actually boot the real app process. Only testing against the real running container surfaced it.
- **"It works on my machine" is a specific, fixable problem, not an excuse.** The `numpy`/Python-version mismatch between local dev and the Docker image was a real, reproducible failure with a real root cause (a version pin requiring a newer Python than the container's base image) — not a vague reliability concern.
- **Monitoring only matters if you actually look at it.** Adding `/metrics` was easy; the more valuable moment was watching Prometheus's own `Status → Targets` page and query UI to confirm data was genuinely flowing, not just present in theory.
- **Security is a series of small, deliberate decisions, not one big feature.** Deciding *which* endpoints need a key (`/predict`, yes) and which shouldn't (`/health`, `/metrics` — so monitoring tools can reach them freely) mattered more than the auth mechanism itself.

---

## Self-Assessment

**Could I explain, end-to-end, how a request flows through the entire system without looking at the code?**
Yes. A request hits Uvicorn inside the Docker container, passes through CORS middleware, then `verify_api_key` checks the `X-API-Key` header, Pydantic validates the JSON body against `PredictionInput` (rejecting bad types, negative values, or unexpected extra fields), the router pulls the already-loaded model from `app.state`, calls `predict_proba()`, logs the result with a `request_id` to console and `app.log`, increments the `ml_predictions_total` Prometheus counter, and returns the JSON response.

**Is the README good enough that a new teammate could get this running without asking a single question?**
Yes — it includes a full architecture diagram, two complete setup paths (Docker Compose and local), every environment variable explained, every endpoint with real curl examples and real responses, a troubleshooting section built from problems genuinely hit during development (not hypothetical ones), and a live URL as a zero-setup alternative.

**What's the one thing about this project I'm least confident explaining in an interview?**
The concurrency/GIL discussion from Task 19's load testing — specifically, *why* Python's GIL limits true parallelism for CPU-bound work across threads, and why multiple Uvicorn worker processes (separate interpreters) help but don't fully solve it on a CPU-constrained container. This is documented in detail in `TESTING.md`, and is the section worth re-reading most closely before discussing this project out loud.

---

## Independent Extension

Chosen extension: GitHub Actions CI (`.github/workflows/tests.yml`)

This was chosen over the other options (model retraining, Grafana dashboard, response caching) because it directly builds on and protects everything already built — the 17-test suite from Tasks 13, 14, and 17 — with the least new infrastructure and the clearest, most universally-applicable payoff: **every future change to this project is now automatically checked before it can silently break something.**

The workflow runs on every push to `main`: checks out the repo, sets up Python 3.12, installs dependencies, builds a `.env` from a GitHub repository secret (`API_KEY` — never hardcoded or committed), and runs the full unit test suite. Integration tests are deliberately excluded from CI since they require the live Docker Compose stack, which the CI runner doesn't have running.

This is a decision made independently, beyond the guided task list — not something explicitly scripted step-by-step in the project brief.

---

## Author

Muthuraja
Python Intern — ML Model Deployment Project