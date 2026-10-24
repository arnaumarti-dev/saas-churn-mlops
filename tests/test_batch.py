"""Batch API regression tests, including size caps and model compatibility."""
from fastapi.testclient import TestClient

from churn_mlops.api.main import app
from churn_mlops.api.security import MAX_BATCH_BODY


SAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5,
    "TotalCharges": 179.0,
}


class BatchDummyModel:
    def predict_proba(self, dataframe):
        # Ensure a single vectorized call matches every incoming row.
        return [[0.2, 0.8] if tenure < 12 else [0.9, 0.1]
                for tenure in dataframe["tenure"]]


def test_batch_predictions_and_aggregates():
    app.state.model = {"model": BatchDummyModel(), "decision_threshold": 0.4}
    stable = {**SAMPLE, "tenure": 36, "MonthlyCharges": 40, "TotalCharges": 1440}
    response = TestClient(app).post("/predict/batch", json={"customers": [SAMPLE, stable]})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    data = response.json()
    assert [item["churn_prediction"] for item in data["predictions"]] == [1, 0]
    assert [item["churn_probability"] for item in data["predictions"]] == [0.8, 0.1]
    assert data["summary"] == {
        "total_customers": 2, "high_risk_customers": 1,
        "low_risk_customers": 1, "average_churn_probability": 0.45,
        "decision_threshold": 0.4, "high_risk_monthly_charges": 89.5,
    }
    assert "customers" not in data


def test_batch_rejects_empty_or_more_than_200_rows():
    app.state.model = BatchDummyModel()
    client = TestClient(app)
    for customers in ([], [SAMPLE] * 201):
        response = client.post("/predict/batch", json={"customers": customers})
        assert response.status_code == 422


def test_batch_accepts_exactly_200_profiles_and_legacy_model():
    app.state.model = BatchDummyModel()
    response = TestClient(app).post("/predict/batch", json={"customers": [SAMPLE] * 200})
    assert response.status_code == 200, response.text
    assert response.json()["summary"]["total_customers"] == 200
    assert response.json()["summary"]["high_risk_customers"] == 200
    assert response.json()["summary"]["decision_threshold"] == 0.5


def test_batch_rejects_invalid_rows_and_extra_fields():
    app.state.model = BatchDummyModel()
    client = TestClient(app)
    invalid = {**SAMPLE, "MonthlyCharges": -10}
    assert client.post("/predict/batch", json={"customers": [invalid]}).status_code == 422
    assert client.post("/predict/batch", json={
        "customers": [SAMPLE], "internal": "secret",
    }).status_code == 422
    assert client.post("/predict/batch", json={
        "customers": [{**SAMPLE, "customerID": "do-not-collect"}],
    }).status_code == 422


def test_batch_rejects_large_bodies_and_unsupported_media():
    client = TestClient(app)
    assert client.post(
        "/predict/batch",
        content="x" * (MAX_BATCH_BODY + 1),
        headers={"Content-Type": "application/json"},
    ).status_code == 413
    assert client.post(
        "/predict/batch",
        content="x,y\n",
        headers={"Content-Type": "text/csv"},
    ).status_code == 415


def test_batch_returns_503_without_model():
    app.state.model = None
    response = TestClient(app).post("/predict/batch", json={"customers": [SAMPLE]})
    assert response.status_code == 503
