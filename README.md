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


## End-to-end verification

A passing unit-test job alone does not prove that the model can train and serve
real data. The GitHub Actions **Train model and test real API prediction** job:

1. Downloads the IBM Telco Customer Churn dataset from the upstream source.
2. Trains both candidate models, picks one by validation ROC-AUC and saves it.
3. Starts the FastAPI test client, loads the persisted model and checks
   `GET /health` and a real `POST /predict` built from an input CSV row.
4. Checks that the model and final test metrics were written, and uploads
   evaluation metric JSON files as workflow artifacts.

Run the same check locally after `make download-data` and `make train`:

```bash
make smoke-test
```

The smoke test checks integration and prediction shape; it does **not**
establish the model's suitability for real-world customer decisions.


## Validation-tuned decision threshold

The project selects the winning algorithm by **validation ROC-AUC**, then
tunes the positive-class decision threshold using **validation data only**.
The tuning objective maximizes **F2** (which weights recall more heavily
than precision) over thresholds from 0.15 to 0.85 in steps of 0.01.
Candidate thresholds must achieve at least **55% precision on validation**;
if none does, the project uses 0.50. This precision floor is a provisional
learning-project requirement, **not** an estimated retention-intervention cost.

The model is refitted on train + validation, and the final threshold is saved
**together with the estimator** in a single joblib artifact. The FastAPI
\`POST /predict\` response now includes \`decision_threshold\`, and its
\`churn_prediction\` is computed from \`churn_probability >= decision_threshold\`.
Older plain-estimator artifacts remain usable at the default 0.50 threshold.

The CI training job checks end-to-end inference and publishes metrics for
validation, validation after threshold tuning, and the held-out test split.
Do not choose or readjust the threshold based on held-out test metrics;
that would invalidate the final test as an unbiased check. Further product
decisions would need intervention-cost analysis and monitoring on new data.


## Interactive portfolio demo

After creating the trained artifact (run \`make download-data && make train\`),
start FastAPI with \`make run-api\` and open **http://localhost:8000/**.

The landing page offers a Spanish-language interactive form with the complete
set of 19 prediction inputs, illustrative customer profiles, the resulting
estimated churn probability, and the **actual saved decision threshold**.
The interface submits a JSON request to the existing \`POST /predict\` route,
so its result uses the real trained model rather than mock predictions.

- Open \`/docs\` for Swagger/OpenAPI and \`/health\` for the API health endpoint.
- The app processes inputs on demand and the demo does not intentionally
  persist them; do not enter real personal or sensitive customer records.
- If you open the page before training, the server still serves the form but
  prediction requests return HTTP 503 with instructions to train first.
- \`make smoke-test\` validates the root page and model-backed API response.
- **Not yet deployed publicly.** The Docker image does not bundle the trained
  artifact (excluded from Git); for runtime predictions you must supply
  \`models/best_model.joblib\` (e.g. with the existing Compose volume mount).
  Choose a hosting provider and a secure model-artifact delivery method
  before publishing an online version.

This is a portfolio demonstration based on IBM Telco Customer Churn data,
not a validated commercial retention tool. Its outputs should not drive
automated decisions about individuals.


## Deployable self-contained demo image

The default `Dockerfile` is a development/inference image that requires a
separately trained `models/best_model.joblib` at runtime.

For a portfolio demonstration, use the opt-in `Dockerfile.demo`.
It downloads the public IBM Telco dataset and trains the model **during the
image build**, then copies only the final model artifact and minimal runtime
dependencies into a second stage. The runtime container does **not** download
data or retrain on every startup.

```bash
docker build -f Dockerfile.demo -t churn-demo:local .
docker run --rm -p 8000:8000 churn-demo:local
```

Open **http://localhost:8000/** to use the dashboard, or
**http://localhost:8000/docs** for Swagger/OpenAPI.
In another terminal, verify the container with:

```bash
python scripts/check_demo_http.py
```

The smoke test checks the HTML page and an actual `POST /predict`, using only
Python's standard library. GitHub Actions builds the dedicated image and runs
the same HTTP checks automatically. Providers that accept Dockerfiles can
build `Dockerfile.demo` with a service port given by the `PORT` environment
variable (default 8000).

**Public hosting has not been configured or performed.** Choose a host and
control costs/quotas before deploying. The build requires Internet access
to the IBM dataset; pin the source artifact/checksum before promising long-term
build reproducibility. Do not submit real customer data to a public demo.
The educational model does not establish fairness, calibrated business costs
or suitability for automated retention decisions.
