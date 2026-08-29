# ML Model Deployment as a Monitored REST API

A machine learning model (Iris species classifier) served through a REST API built with **FastAPI**, as part of the Python Intern task program.

> **Status:** Tasks 1–9 complete ✅ (Foundation + Phase 2: Core Development)

---

## 📌 Project Overview

This project trains a simple ML model and serves predictions through a REST API.

- **Dataset:** Iris flower dataset (`scikit-learn` built-in / CSV)
- **Problem type:** Classification — predict the Iris species from flower measurements
- **API contract:**
  - `POST /predict` — accepts flower measurements (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) as a Pydantic-validated JSON body, and returns the predicted species, a confidence score, and a request ID.
  - `GET /` — root/liveness check confirming the API is running.
  - `GET /health` — health check returning `{"status": "ok", "model_loaded": true}` for monitoring systems.
- **Validation:** Request body validated with a `PredictionInput` Pydantic model (all four measurements required, must be greater than zero).
- **Response contract:** Enforced with a `PredictionOutput` Pydantic model (`prediction`, `confidence`, `request_id`); errors return a clean, documented shape — never a raw traceback.
- **Observability:** Every request and prediction is logged (console + `app.log`) with a unique request ID, HTTP method, path, status code, and duration.

---

## 🗂️ Project Structure

```
ml-api-project/
├── app/
│   ├── main.py             # FastAPI app entry point (lifespan model load, routes, error handlers)
│   ├── logging_config.py    # Console + rotating file logging setup
│   ├── models/
│   │   └── schemas.py        # PredictionInput / PredictionOutput Pydantic schemas
│   └── routers/                # API route files
├── ml/
│   ├── train.py                 # Model training script
│   ├── data/
│   │   └── iris_dataset.csv
│   ├── saved_model/
│   │   └── model.joblib          # Trained, serialized model
│   └── test_model.py             # Script to reload model & verify predictions
├── tests/
├── venv/
├── app.log                       # Runtime logs (request IDs, status, duration, predictions)
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
The server starts at: **http://127.0.0.1:8000**. The model is loaded once at startup (not per request) via FastAPI's lifespan handler.

### Try the endpoints
- Root/liveness check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Interactive API docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Example `POST /predict` request body:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
Example response:
```json
{
  "prediction": "setosa",
  "confidence": 0.98,
  "request_id": "1ff6fd92-66dd-4d2f-8d03-e676cfc0d035"
}
```
Sending malformed input (missing field, wrong type, or a non-positive measurement) returns a `422 Unprocessable Entity` with a clear validation error instead of crashing the service. Unexpected internal failures return a clean `500` with a safe message — never a raw Python traceback.

### Logs
Every request and prediction is written to both the console and `app.log` (project root), tagged with a unique `request_id`:
```
2026-08-28 12:12:26,996 | INFO | Model loaded successfully
2026-08-28 12:13:10,136 | INFO | request_id=ebbe3d3e-... | Prediction successful | prediction=setosa | confidence=1.0000
2026-08-28 12:13:10,145 | INFO | request_id=ebbe3d3e-... | method=POST | path=/predict | status=200 | duration=0.0655s
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

### Task 1 — Project Planning
- Selected the Iris dataset as the ML problem.
- Defined the input/output API contract in plain English.
- Sketched the request → validation → model → response flow.

### Task 2 — Environment Setup
- Created and activated a Python virtual environment.
- Set up the standard project folder structure.
- Installed FastAPI, Uvicorn, scikit-learn, and pandas.
- Added `.gitignore` and committed structure to GitHub.

### Task 3 — Model Training & Saving
- Trained a classification model on the Iris dataset in `ml/train.py`.
- Evaluated accuracy on the test set.
- Saved the trained model using `joblib` to `ml/saved_model/model.joblib`.
- Verified the saved model reloads correctly and predicts accurately via `test_model.py`.

### Task 4 — FastAPI Integration
- Built a minimal FastAPI app (`app/main.py`) with `GET /` and `POST /predict` endpoints.
- Ran the server locally with `uvicorn app.main:app --reload`.
- Verified both endpoints respond correctly.
- Tested endpoints using the automatic Swagger UI at `/docs`.

### Task 5 — Real Model Loaded at Startup
- Used FastAPI's `lifespan` context manager to load `model.joblib` once at app boot, confirmed via a startup log line.
- Removed all model-loading code from inside the endpoint function.
- Updated `/predict` to convert the request into the array/DataFrame shape the model expects and call `model.predict(...)`.
- Verified different inputs return different, genuine predictions (e.g. `[1,1,1,1]` → `setosa`, `[6.0, 2.9, 4.5, 1.5]` → `versicolor`).

### Task 6 — Pydantic Input Validation
- Defined a `PredictionInput` schema in `app/models/schemas.py` with all four measurements required and constrained to be positive (`Field(..., gt=0)`).
- Changed `/predict` to accept `data: PredictionInput` so FastAPI validates automatically before the function runs.
- Confirmed Swagger UI now shows a schema-driven Request body, and bad input returns `422` instead of crashing.

### Task 7 — Milestone: Core API Assembled
- Rebuilt `/predict` combining the Pydantic schema, the startup-loaded model, and `model.predict_proba()` for a confidence score.
- Added a `request_id` (via `uuid4()`) to every response.
- Added `GET /health` returning `{"status": "ok", "model_loaded": true}`, safely checking the model state.
- Verified `/docs`, `/health`, and `/predict` all work from a single `uvicorn` command.

### Task 8 — Response Models & Error Handling
- Defined a `PredictionOutput` response model (`prediction`, `confidence`, `request_id`) and attached it via `response_model=PredictionOutput`.
- Wrapped inference in `try/except`, raising `HTTPException(status_code=500, detail="Prediction failed")` on failure while logging the real error internally.
- Added a custom exception handler for a specific failure mode so no raw Python traceback ever reaches the client.

### Task 9 — Logging & Monitoring
- Added `logging_config.py` with both a console handler and a rotating file handler writing to `app.log`.
- Logged model startup (`Model loaded successfully`).
- Added request-level logging: unique `request_id`, HTTP method, path, status code, and duration for every request.
- Added prediction-level logging: predicted class and confidence for every successful `/predict` call.
- Removed all `print()` calls from the logging flow.
- Moved `app.log` from `app/` to the project root for a cleaner structure.
- Verified via `app.log` that startup, request, and prediction events are all captured correctly with matching request IDs.

---

## 🛠️ Tech Stack

- **Python 3.12**
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **scikit-learn** — model training
- **joblib** — model serialization
- **pandas** — data handling
- **Pydantic** — request/response validation
- **logging** (stdlib) — console + file logging with request correlation IDs

---

## 📅 What's Next

- **Task 10:** API versioning.
- Add automated tests under `tests/`.
- Containerize the app (Docker) and add basic monitoring/metrics.

---

## 👤 Author

**Muthuraja**
Python Intern — ML Model Deployment Project