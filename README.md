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
- Tracks every run in MLflow (`mlflow.db`, a local SQLite database).
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
├── mlflow.db                    # MLflow experiment tracking (SQLite database)
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


## Public portfolio demo: Render (optional)

An opt-in `render.yaml` Blueprint now describes a **Free** Render web service
for the Dockerized demo; merely committing it does not create a service.

To publish it manually:

1. Sign into [Render](https://dashboard.render.com/) and connect GitHub.
2. Select **New > Blueprint** and choose `arnaumarti-dev/saas-churn-mlops`,
   branch `main`. Render should discover the root `render.yaml`.
3. Review the resource details **before approving**: web service
   `churn-mlops-demo`, Docker runtime, `Dockerfile.demo`, Frankfurt region,
   `free` compute plan and health check `/health`.
4. Start the Blueprint deployment. Render builds the trained model into the
   container image and then serves the FastAPI home page at its **assigned**
   `onrender.com` URL. No URL exists until the service is actually created.
5. Check the generated home page, `/health`, `/docs`, and submit a
   demonstrative (non-personal) client profile to confirm `POST /predict`.

**Important limitations and costs:** Render's Free web services can spin down
after 15 minutes of inactivity and take approximately a minute to start again.
The plan has monthly compute, build and bandwidth limits. Review current
billing and usage limits before enabling deployment; extra bandwidth can incur
charges. Free instances and this tutorial are not intended for production.
Because `Dockerfile.demo` fetches the public IBM data source during build,
deployment also depends on availability of that upstream URL.

This application has no accounts, authentication or request throttling and
should be used only with fictional/demonstration profiles. Never enter real
customer records or other sensitive information. Public hosting has not been
performed by this repository change.


## Security controls (portfolio demo)

- **Input protection:** `POST /predict` accepts JSON only; request bodies are
  capped at 16 KiB even for streamed/chunked bodies. Numeric inputs reject
  NaN/infinity, unexpected fields, and extremely large values.
- **Abuse controls:** up to 30 predictions per 60 seconds, per apparent client
  IP, are accepted, with HTTP 429 + Retry-After after the limit. This is
  **in-process best-effort limiting**, not distributed edge protection:
  counters reset on restarts and Render proxies may aggregate client IPs.
- **Browser headers:** CSP disallows inline JS/CSS on the demo and serves both
  resources locally under `/assets`. HSTS, no-sniff, deny-framing, no-referrer
  and permissions policy are added. Built-in Swagger (`/docs`, `/redoc`)
  uses a **narrow exception** for its third-party CDN and inline initializer.
- **Dependencies:** FastAPI/Pydantic/Uvicorn and MLflow have been updated for
  security maintenance; CI verifies compatibility, including training in Docker.
  Automated `pip-audit` and Dependabot checks detect vulnerable Python
  packages and propose updates. Treat audit failures as blockers.
- **Supply chain:** the runtime uses a non-root user and loads only the
  model artifact produced by our own trusted Docker training stage. Loading
  an attacker-controlled pickle/joblib file can execute Python code.
- **Deployment:** changing GitHub does not update the existing Render service
  because automatic deploys were disabled. Review and test before each
  explicit redeploy. No API keys or customer data are necessary to use
  this educational demo.

This does not establish that the service is free of CVEs, denial-of-service
vectors, or new supply-chain vulnerabilities. For serious external traffic,
add provider-side/WAF rate limiting, request monitoring and operational alerts.


## Batch predictions and dashboard (demo)

The home page now offers two modes:

- **Individual customer:** the existing interactive calculator using `POST /predict`.
- **CSV & dashboard:** either generate 12 **synthetic** customer profiles or
  import a small UTF-8 CSV containing exactly the 19 model input columns.
  Column order is arbitrary, but the headers must match the example exactly;
  optional customer identifiers or other unknown columns are rejected. The
  interface uses a quote-aware CSV parser and displays aggregated classification
  counts, mean predicted probability, threshold, monthly charges **associated
  with high-risk profiles** and per-row scores. The monthly amount is **not**
  a prediction of lost revenue.

The "Download sample CSV" button generates synthetic data in the browser.
On the server, `POST /predict/batch` accepts up to **200 profiles**, in the
same schema as `POST /predict`:

```json
{
  "customers": [
    {
      "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
      "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
      "MultipleLines": "No", "InternetService": "Fiber optic",
      "OnlineSecurity": "No", "OnlineBackup": "No",
      "DeviceProtection": "No", "TechSupport": "No",
      "StreamingTV": "No", "StreamingMovies": "No",
      "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check", "MonthlyCharges": 89.50,
      "TotalCharges": 179.0
    }
  ]
}
```

The response contains `predictions` in input order and an aggregated
`summary`. The endpoint evaluates all rows in one vectorized model call,
rejects requests over **256 KiB**, and shares the existing rate limiter.
It does not persist the submitted profiles or echo them in its responses.
The original 16 KiB `/predict` limit remains unchanged.

**Privacy and suitability:** do not submit real customer information,
identifiers or commercially sensitive records to this public educational demo.
The application does not include accounts, per-tenant isolation, storage,
retention policies or an independently validated business risk model.
Charts represent predictions and classifications, **not measured customer
outcomes**.

## Segmented portfolio analytics (experimental)

The optional batch dashboard now includes **client-side** analysis of model
results: contract cohorts (monthly / one-year / two-year), counts above the
validation threshold, filtering by classification and contract, sorting by
probability, and exporting the **currently filtered** result rows as CSV.

- Aggregate headline metrics still refer to the **entire analyzed batch**.
  Filters only affect the row table and the downloaded CSV.
- Contract cohorts display **counts and proportions**, not causal effects.
  The model has not been validated for actual retention interventions.
- The downloaded results file contains only the **source CSV row number**,
  a restricted contract category, prediction probability, and classification.
  It never includes customer IDs, names, email addresses, or other original
  fields. Contract category labels are normalized to an allowlist so malicious
  spreadsheet formulas from arbitrary imported data cannot be included.
- Original profiles and predictions exist only in the browser session's memory
  and are cleared when another dataset is selected; no new backend storage,
  cookies, accounts or third-party analytics are introduced.
- Data still travels to the API for model scoring. **Use synthetic profiles
  only** until the service has real authentication, tenant isolation,
  compliance review, access logging policy and appropriate retention controls.

Unit tests for these analytical helpers use the built-in Node test runner;
there are no new runtime or browser dependencies.


## Interface usability and accessibility improvements (v1.4 candidate)

The existing individual/batch views have a keyboard-accessible tablist:
Left/Right/Home/End select and focus the corresponding tab, plus a "skip to
content" link. The batch flow now includes three visible steps, an accessible
live status message while scoring, a **Cancel analysis** action and a shortcut
to the results.

- Choosing a new file or loading an example clears outdated results and cancels
  a pending batch request. Slow or out-of-order file reads cannot replace newer
  selections. Editing an individual profile invalidates its previous result.
- Empty result filters display a clear explanation and **Clear filters**;
  export is unavailable when no rows match.
- CSS includes narrow-phone layouts (down to 375 px), minimum 44 px action
  targets and reduced-motion support. No new CDN assets, tracking scripts,
  third-party browser dependencies, authentication or server storage.
- Existing CSP, rate limits, payload limits and schema validation are unchanged.

CI executes JavaScript behavior tests in a simulated DOM, source-level
accessibility checks and the full Docker HTTP smoke test. **These are not a
substitute for manual visual QA in real desktop/mobile browsers.** Before
deploying, manually inspect at 375 px, 620 px and desktop widths; navigate the
tablist using only the keyboard, upload the example CSV, test cancelled and
empty analyses, verify focus indicators and try 200% text zoom. Continue to use
synthetic profiles only.


## Next SaaS step: tenant isolation prototype (not enabled)

An **explicitly opt-in, disabled-by-default** developer-only security prototype
adds tenant-scoped `/saas/v1/tenants/{tenant_uuid}/me` and
`/saas/v1/tenants/{tenant_uuid}/predict/batch` endpoints. Its purpose is to
test isolated membership, authorization and permission failures before
building real customer login. **The public Churn Lab demo remains unchanged.**

The sandbox accepts only short-lived, server-provisioned random bearer tokens
stored as SHA-256 hashes in private environment configuration. It verifies
company membership and role before serving an in-memory result; cross-company
lookups fail with 404. Viewer-only roles cannot run predictions.
There is **no database, no user signup, no persistent profile storage,
no browser authentication and no access to real customer data**.

The prototype must **not be enabled on public Render or considered
production-grade authentication**. Before an actual SaaS launch, replace it
with a managed OIDC provider, persistent role/membership management,
PostgreSQL tenant-scoped storage with RLS, GDPR review, key lifecycle,
auditing and production security testing.

Read the threat model, test configuration and release gates in
[docs/saas-auth-tenant-architecture.md](docs/saas-auth-tenant-architecture.md).

## Optional OIDC + PostgreSQL tenant staging (v1.6 candidate)

The next security increment adds a **separate, feature-disabled** integration
namespace at `/saas/v2/tenants/{uuid}/` with trusted OIDC access-token
verification (pinned RS256, issuer, audience, expiry, subject, signing key)
and authoritative PostgreSQL organization memberships. Roles are obtained
from the database, **never** from caller-provided IDs or JWT roles.

A new migration defines organizations, membership and synthetic-only
run metadata tables with **FORCE ROW LEVEL SECURITY** and least-privileged
role grants. The GitHub Actions integration suite validates signed/invalid
JWTs and cross-tenant isolation against ephemeral real PostgreSQL.

There is **no hosted identity provider, database service, registration UI
or company data** in the public app. Optional dependencies are isolated in
`requirements.saas.txt`, **not** installed in `Dockerfile.demo`; the
staging feature is off unless explicitly configured.

See [docs/oidc-postgres-staging.md](docs/oidc-postgres-staging.md).
Never enable it on public Render or submit real customer records before
a full security, privacy and operations review.
