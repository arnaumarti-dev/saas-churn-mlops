"""Checks for the portfolio UI served by the existing FastAPI application."""

from fastapi.testclient import TestClient

from churn_mlops.api.main import app


def test_root_serves_interactive_spanish_demo() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert '<html lang="es">' in html
    assert 'id="prediction-form"' in html
    assert 'id="result-data"' in html
    assert 'fetch("/predict"' in html
    assert "decision_threshold" in html


def test_demo_includes_all_required_churn_features() -> None:
    html = TestClient(app).get("/").text
    fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    for field in fields:
        # Inputs/selects are authored directly or built from the static field list.
        assert f'name="{field}"' in html or f'["{field}",' in html


def test_swagger_docs_remains_accessible() -> None:
    response = TestClient(app).get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()
