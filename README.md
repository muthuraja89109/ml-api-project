# ML Model Deployment as a Monitored REST API

A machine learning model (Iris species classifier) served through a REST API built with **FastAPI**, as part of the Python Intern task program.

> **Status:** Phase 1 — Foundation (Tasks 1–4) complete ✅

---

## 📌 Project Overview

This project trains a simple ML model and serves predictions through a REST API.

- **Dataset:** Iris flower dataset (`scikit-learn` built-in / CSV)
- **Problem type:** Classification — predict the Iris species from flower measurements
- **API contract:**
  - `POST /predict` — accepts flower measurements (sepal length, sepal width, petal length, petal width) as input and returns the predicted Iris species as output.
  - `GET /` — health check endpoint confirming the API is running.

---

## 🗂️ Project Structure

```
ml-api-project/
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── models/            # Pydantic schemas
│   └── routers/            # API route files
├── ml/
│   ├── train.py            # Model training script
│   ├── data/
│   │   └── iris_dataset.csv
│   ├── saved_model/
│   │   └── model.joblib     # Trained, serialized model
│   └── test_model.py        # Script to reload model & verify predictions
├── tests/
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
The server starts at: **http://127.0.0.1:8000**

### Try the endpoints
- Root/health check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Interactive API docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ✅ Progress Log

| Day | Task | Status |
|-----|------|--------|
| Day 1 | Project planning & dataset selection | ✅ Complete |
| Day 2 | Python environment & project structure setup | ✅ Complete |
| Day 3 | ML model training, evaluation & saving | ✅ Complete |
| Day 4 | FastAPI integration — `GET /` and `POST /predict`, Swagger testing | ✅ Complete |

### Day 1 — Project Planning
- Selected the Iris dataset as the ML problem.
- Defined the input/output API contract in plain English.
- Sketched the request → validation → model → response flow.

### Day 2 — Environment Setup
- Created and activated a Python virtual environment.
- Set up the standard project folder structure.
- Installed FastAPI, Uvicorn, scikit-learn, and pandas.
- Added `.gitignore` and committed structure to GitHub.

### Day 3 — Model Training & Saving
- Trained a classification model on the Iris dataset in `ml/train.py`.
- Evaluated accuracy on the test set.
- Saved the trained model using `joblib` to `ml/saved_model/model.joblib`.
- Verified the saved model reloads correctly and predicts accurately via `test_model.py`.

### Day 4 — FastAPI Integration
- Built a minimal FastAPI app (`app/main.py`) with `GET /` and `POST /predict` endpoints.
- Ran the server locally with `uvicorn app.main:app --reload`.
- Verified both endpoints respond correctly.
- Tested endpoints using the automatic Swagger UI at `/docs`.

---

## 🛠️ Tech Stack

- **Python 3.12**
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **scikit-learn** — model training
- **joblib** — model serialization
- **pandas** — data handling

---

## 📅 What's Next

- **Task 5:** Replace the hardcoded `/predict` response with real predictions from the saved model.
- Add input validation with Pydantic schemas.
- Add automated tests.
- Add monitoring/logging to the API.

---

## 👤 Author

**Muthuraja**
Python Intern — ML Model Deployment Project