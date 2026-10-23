"""Smoke test the trained model using the real FastAPI prediction route.

Run after downloading the dataset and training:
    PYTHONPATH=src python scripts/smoke_test.py

This checks the data -> fitted model -> API path, not predictive accuracy.
"""
import csv
from pathlib import Path

from fastapi.testclient import TestClient

from churn_mlops.api.main import app
from churn_mlops.config import BEST_MODEL_PATH, RAW_DATA_PATH


def main() -> None:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset missing: {RAW_DATA_PATH}")
    if not BEST_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model missing: {BEST_MODEL_PATH}")

    with Path(RAW_DATA_PATH).open("r", encoding="utf-8", newline="") as source:
        first_row = next(csv.DictReader(source))

    # Use all raw feature names: inference.py builds derived features.
    payload = {
        key: value
        for key, value in first_row.items()
        if key not in {"customerID", "Churn"}
    }

    # TestClient with a context manager executes the real startup hook
    # that loads the serialized model from disk.
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text

        prediction = client.post("/predict", json=payload)
        assert prediction.status_code == 200, prediction.text
        result = prediction.json()
        assert 0.0 <= result["churn_probability"] <= 1.0, result
        assert result["churn_prediction"] in {0, 1}, result

    print("End-to-end training/artifact/API smoke test passed.")
    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
