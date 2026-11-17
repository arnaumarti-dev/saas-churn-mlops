# Compatibility review — Python dependency maintenance (17 Nov 2026)

The repository is intentionally built and tested with **Python 3.11**, including all training, staging, demo and CI images.

## Accepted candidate versions (subject to passing CI)

- pandas: 2.2.2 → 3.0.6
- scikit-learn: 1.5.1 → 1.9.1
- matplotlib (training only): 3.9.2 → 3.11.2
- joblib: 1.4.2 → 1.6.0
- uvicorn: 0.53.0 → 0.54.0

All dependencies shared by the trainer and runtime images must be pinned to matching versions in `requirements.txt` and `requirements.serve.txt`. The IBM Telco training job produces the model artifact at image build time and the serving image loads this artifact under the same scikit-learn and joblib versions.

## Rejected automatic proposal

**NumPy 2.5.4 cannot install on Python 3.11.** GitHub Actions and Docker reported `Requires-Python >=3.12` and `No matching distribution found`; keep `numpy==1.26.4` in both dependency files for now. Any move to Python 3.12+ and NumPy 2 must be an explicit, independently tested migration and must check XGBoost, pandas, scikit-learn, MLflow, Docker, serialization, API and staging.

## Required checks

Do not merge this branch unless full **CI**, **Dependency security** and **Real-browser visual QA** pass on the current PR commit with the runner-OS fix from the merged workflow-maintenance PR #40. The optional SaaS integration tests are relevant when any of the staging or auth files change; the browser QA already exercises the protected staging console. Review the rendered public demo and verify the Telco model can still be trained and served.

A successful test suite does **not** validate the offline SaaS model with actual customer data, prove privacy compliance or guarantee that future upgrades cannot cause drift. Keep the public demo and private staging distinct.
