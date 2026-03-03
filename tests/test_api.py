from fastapi.testclient import TestClient

from churn_mlops.api.main import app


class DummyModel:
    def predict_proba(self, X):
        return [[0.2, 0.8]]

    def predict(self, X):
        return [1]


def test_predict_endpoint() -> None:
    app.state.model = DummyModel()
    client = TestClient(app)

    payload = {
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
        "TotalCharges": 840.0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] == 1
    assert 0 <= body["churn_probability"] <= 1
