# SaaS Churn MLOps (Production-Style)

End-to-end machine learning project to predict SaaS customer churn using a public Telco churn dataset. The project includes reproducible training, experiment tracking with MLflow, model serving through FastAPI, input validation with Pydantic, containerization, logging, and unit testing.

## 1) Business Problem

SaaS companies lose recurring revenue when customers churn. The objective is to predict churn risk early so customer success and retention teams can prioritize interventions, reduce attrition, and improve LTV.

**Target:** `Churn` (Yes/No)

**Dataset:** IBM Telco Customer Churn (public dataset).

---

## 2) Architecture

```text
scripts/download_data.py
        |
        v
 data/telco_churn.csv
        |
        v
src/churn_mlops/train.py
  - cleaning
  - feature engineering
  - split (no leakage)
  - train LR + XGBoost
  - evaluate (ROC-AUC, Precision, Recall, F1, confusion matrix)
  - MLflow tracking
  - save best model
        |
        v
 models/best_model.joblib
        |
        v
src/churn_mlops/api/main.py (FastAPI)
  - /health
  - /predict
```

Key production choices:
- Uses `Pipeline` + `ColumnTransformer` to prevent leakage (fit transformations on train only).
- Tracks every run in MLflow (`mlruns/`).
- Saves confusion matrix and metrics artifacts.
- Exposes prediction API with strict Pydantic request schema.
- Central logging configuration.
- Docker + docker-compose for deployment workflow.

---

## 3) Project Structure

```text
.
├── artifacts/                    # metrics JSON + confusion matrix plots
├── data/
│   └── telco_churn.csv          # downloaded raw data
├── mlruns/                      # MLflow tracking backend (local file store)
├── models/
│   └── best_model.joblib        # persisted best model
├── scripts/
│   ├── download_data.py
│   └── train_model.py
├── src/churn_mlops/
│   ├── api/main.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── logging_config.py
│   ├── modeling.py
│   ├── schemas.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   └── test_data_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 4) How to Run Locally

### Prerequisites
- Python 3.11+
- `pip`

### Setup
```bash
make install
make download-data
make train
```

### Start API
```bash
make run-api
```

API docs: `http://localhost:8000/docs`

Example `/predict` request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
  }'
```

### MLflow UI
```bash
make mlflow-ui
```
Open: `http://localhost:5000`

---

## 5) How to Run with Docker

```bash
make docker-up
```

Services:
- API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`

Stop:
```bash
make docker-down
```

---

## 6) Testing

```bash
make test
```

Includes:
- Data cleaning + feature engineering tests.
- FastAPI `/predict` endpoint test.

---

## 7) Clean Code / Best Practices Applied

- Modular package layout with clear responsibilities.
- Reusable configuration constants.
- Explicit logging for training and inference lifecycle.
- Deterministic train/test split with `random_state` and stratification.
- No leakage: preprocessing encapsulated in sklearn pipelines.
- Artifacts and model persistence separated from business logic.
- API schema validation via Pydantic.
- Unit tests for critical pipeline and endpoint behavior.

---

## Model evaluation: leakage-free splits

The training command uses a **60% train / 20% validation / 20% held-out test**
stratified split with a fixed random seed.

1. Train Logistic Regression and XGBoost on the **train** split.
2. Fit missing-value imputation, scaling, and one-hot encoding *inside* each
   sklearn pipeline, so the validation and test data cannot influence fitted
   preprocessing statistics.
3. Compare candidate models by **validation ROC-AUC**.
4. Refit the winning algorithm on **train + validation** data.
5. Evaluate the held-out **test split once**, after selecting the algorithm.

Run `make download-data`, then `make train`.
Validation metrics and the final test metrics are saved separately in
`artifacts/` and MLflow; the winning refitted pipeline is persisted under
`models/best_model.joblib`.

Run automated tests with `make test`. The CI workflow checks Python tests and
builds the API Docker image. **A successful CI run does not by itself verify
prediction accuracy:** actual dataset metrics must come from a training run.
