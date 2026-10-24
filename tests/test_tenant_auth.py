"""Isolation and fail-closed guarantees for the opt-in SaaS sandbox.

All tokens are generated at test time. No real company credentials or
customer records should be used with these test-only endpoints.
"""
import hashlib
import json
import secrets
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from churn_mlops.api.main import app
from churn_mlops.api.security import MAX_BATCH_BODY


class DummyModel:
    def predict_proba(self, frame):
        return [[0.2, 0.8] if month < 12 else [0.9, 0.1]
                for month in frame["tenure"]]


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


@pytest.fixture()
def sandbox(monkeypatch):
    company_a, company_b = uuid4(), uuid4()
    actor_a, actor_b, viewer = uuid4(), uuid4(), uuid4()
    owner_token, analyst_token, viewer_token = (
        secrets.token_urlsafe(32) for _ in range(3)
    )
    expiry = int(time.time()) + 3600
    entries = [
        {"token_sha256": hashlib.sha256(owner_token.encode()).hexdigest(),
         "tenant_id": str(company_a), "actor_id": str(actor_a),
         "role": "owner", "expires_at": expiry},
        {"token_sha256": hashlib.sha256(analyst_token.encode()).hexdigest(),
         "tenant_id": str(company_b), "actor_id": str(actor_b),
         "role": "analyst", "expires_at": expiry},
        {"token_sha256": hashlib.sha256(viewer_token.encode()).hexdigest(),
         "tenant_id": str(company_a), "actor_id": str(viewer),
         "role": "viewer", "expires_at": expiry},
    ]
    monkeypatch.setenv("CHURN_SAAS_PROTOTYPE_ENABLED", "1")
    monkeypatch.setenv("CHURN_SAAS_PROTOTYPE_TOKENS_JSON", json.dumps(entries))
    app.state.model = {"model": DummyModel(), "decision_threshold": 0.4}
    return {
        "a": company_a, "b": company_b,
        "owner": owner_token, "analyst": analyst_token, "viewer": viewer_token,
        "entries": entries,
    }


def header(token):
    return {"Authorization": "Bearer " + token}


def path(company, endpoint="me"):
    return f"/saas/v1/tenants/{company}/{endpoint}"


def test_disabled_by_default_and_never_exposes_memberships(monkeypatch):
    monkeypatch.delenv("CHURN_SAAS_PROTOTYPE_ENABLED", raising=False)
    monkeypatch.delenv("CHURN_SAAS_PROTOTYPE_TOKENS_JSON", raising=False)
    company = uuid4()
    response = TestClient(app).get(path(company))
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    response = TestClient(app).post(
        path(company, "predict/batch"), json={"customers": [SAMPLE]},
    )
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"


def test_bearer_auth_and_membership_are_server_verified(sandbox):
    client = TestClient(app)
    route = path(sandbox["a"])
    assert client.get(route).status_code == 401
    assert client.get(route, headers=header("fake")).status_code == 401
    valid = client.get(route, headers=header(sandbox["owner"]))
    assert valid.status_code == 200
    assert valid.headers["cache-control"] == "no-store"
    data = valid.json()
    assert data["tenant_id"] == str(sandbox["a"])
    assert data["role"] == "owner"
    assert data["mode"] == "prototype-only"
    assert "token" not in str(data).lower()
    assert client.get(path(sandbox["b"]), headers=header(sandbox["owner"])).status_code == 404
    assert client.get(path(sandbox["a"]), headers={
        **header(sandbox["analyst"]), "X-Tenant-ID": str(sandbox["a"])
    }).status_code == 404


def test_role_based_prediction_is_tenant_scoped(sandbox):
    client = TestClient(app)
    request = {"customers": [SAMPLE, {**SAMPLE, "tenure": 30}]}
    a = path(sandbox["a"], "predict/batch")
    b = path(sandbox["b"], "predict/batch")
    success = client.post(a, json=request, headers=header(sandbox["owner"]))
    assert success.status_code == 200
    assert success.json()["summary"]["total_customers"] == 2
    assert success.json()["summary"]["high_risk_customers"] == 1
    assert success.headers["cache-control"] == "no-store"
    assert client.post(a, json=request, headers=header(sandbox["viewer"])).status_code == 403
    assert client.post(a, json=request, headers=header(sandbox["analyst"])).status_code == 404
    assert client.post(b, json=request, headers=header(sandbox["analyst"])).status_code == 200
    assert client.post(a, json=request, headers={
        **header(sandbox["analyst"]), "X-Tenant-ID": str(sandbox["a"])
    }).status_code == 404


def test_expired_credentials_and_bad_operator_config_fail_closed(sandbox, monkeypatch):
    client = TestClient(app)
    route = path(sandbox["a"])
    expired = sandbox["entries"]
    expired[0]["expires_at"] = int(time.time()) - 5
    monkeypatch.setenv("CHURN_SAAS_PROTOTYPE_TOKENS_JSON", json.dumps(expired))
    assert client.get(route, headers=header(sandbox["owner"])).status_code == 401
    monkeypatch.setenv("CHURN_SAAS_PROTOTYPE_TOKENS_JSON", "not-json")
    result = client.get(route, headers=header(sandbox["owner"]))
    assert result.status_code == 503
    assert "not-json" not in result.text
    monkeypatch.setenv("CHURN_SAAS_PROTOTYPE_ENABLED", "true")
    assert client.get(route, headers=header(sandbox["owner"])).status_code == 503


def test_duplicate_token_config_fails_closed(sandbox, monkeypatch):
    entries = sandbox["entries"]
    entries.append({**entries[0], "tenant_id": str(sandbox["b"])})
    monkeypatch.setenv("CHURN_SAAS_PROTOTYPE_TOKENS_JSON", json.dumps(entries))
    result = TestClient(app).get(path(sandbox["a"]), headers=header(sandbox["owner"]))
    assert result.status_code == 503


def test_tenant_batch_reuses_body_limits_and_json_only(sandbox):
    client = TestClient(app)
    route = path(sandbox["a"], "predict/batch")
    oversized = client.post(
        route, content="x" * (MAX_BATCH_BODY + 1),
        headers={**header(sandbox["owner"]), "Content-Type": "application/json"},
    )
    assert oversized.status_code == 413
    assert oversized.headers["cache-control"] == "no-store"
    unsupported = client.post(
        route, content="gender,tenure\nFemale,2\n",
        headers={**header(sandbox["owner"]), "Content-Type": "text/csv"},
    )
    assert unsupported.status_code == 415
    invalid_schema = client.post(
        route, json={"customers": [{**SAMPLE, "customerID": "do-not-collect"}]},
        headers=header(sandbox["owner"]),
    )
    assert invalid_schema.status_code == 422
