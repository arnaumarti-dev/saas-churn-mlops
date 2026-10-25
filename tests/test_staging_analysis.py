"""Tenant-scoped synthetic scoring and only aggregate run history.

All people, organization names and scoring profiles are fixtures, never
customer records. PostgreSQL integration requires ephemeral CI database.
"""
import hashlib
import os
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("jwt")
from cryptography.fernet import Fernet

from churn_mlops.api.main import app
from churn_mlops.api import main as api_main
from churn_mlops.api import staging_login as login

ISSUER = "https://login.example.org/"
BASE = "/saas/v3"


@pytest.fixture
def config(monkeypatch):
    env = {
        "CHURN_SAAS_MANAGEMENT_ENABLED": "1",
        "CHURN_SAAS_LOGIN_ENABLED": "1",
        "CHURN_SAAS_OIDC_ENABLED": "1",
        "CHURN_SAAS_OIDC_ISSUER": ISSUER,
        "CHURN_SAAS_OIDC_AUDIENCE": "churn-synthetic-ci",
        "CHURN_SAAS_OIDC_JWKS_URL": "https://login.example.org/.well-known/jwks.json",
        "CHURN_SAAS_OIDC_CLIENT_ID": "churn-synthetic-browser",
        "CHURN_SAAS_AUTHORIZATION_URL": "https://login.example.org/authorize",
        "CHURN_SAAS_TOKEN_URL": "https://login.example.org/oauth/token",
        "CHURN_SAAS_REDIRECT_URI": "https://testserver/saas/v3/auth/callback",
        "CHURN_SAAS_DATABASE_URL": os.getenv("CHURN_TEST_DB_APP_URL", "postgresql://unused.invalid/test"),
        "CHURN_SAAS_SESSION_FERNET_KEY": Fernet.generate_key().decode("ascii"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


def browser(subject):
    digest = hashlib.sha256(subject.encode()).digest()
    client = TestClient(
        app, base_url="https://testserver", follow_redirects=False,
        client=(f"10.{digest[0]}.{digest[1]}.{digest[2]}", 55000),
    )
    cfg = login._login_config()
    client.cookies.set(
        login.SESSION_COOKIE,
        login._seal(cfg, {
            "iss": ISSUER, "sub": subject, "exp": int(time.time()) + 600
        }), path="/",
    )
    return client


def sample(client, tenant, payload=None):
    return client.post(
        f"{BASE}/organizations/{tenant}/runs/sample",
        json={} if payload is None else payload,
        headers={"Origin": "https://testserver"},
    )


def test_public_demo_cannot_access_sample_routes(monkeypatch):
    monkeypatch.delenv("CHURN_SAAS_MANAGEMENT_ENABLED", raising=False)
    tenant = uuid4()
    with TestClient(app, base_url="https://testserver") as guest:
        assert guest.get(f"{BASE}/organizations/{tenant}/runs").status_code == 404
        assert sample(guest, tenant).status_code == 404


def test_valid_session_and_same_origin_required(config):
    tenant = uuid4()
    with TestClient(app, base_url="https://testserver") as guest:
        assert guest.get(f"{BASE}/organizations/{tenant}/runs").status_code == 401
        assert sample(guest, tenant).status_code == 401
    with browser("staging-actor-" + str(uuid4())) as actor:
        assert actor.post(
            f"{BASE}/organizations/{tenant}/runs/sample", json={}
        ).status_code == 403
        assert actor.post(
            f"{BASE}/organizations/{tenant}/runs/sample", json={},
            headers={"Origin": "https://not-trusted.example"}
        ).status_code == 403
        assert sample(actor, tenant, {"customers": []}).status_code == 422


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"), reason="Ephemeral PostgreSQL required")
def test_synthetic_runs_are_tenant_scoped_and_role_authorized(config, monkeypatch):
    owner_name = "owner-" + str(uuid4())
    viewer_name = "viewer-" + str(uuid4())
    outsider_name = "outsider-" + str(uuid4())

    with browser(owner_name) as owner, browser(viewer_name) as viewer, browser(outsider_name) as outsider:
        # FastAPI model startup without fixture artifacts is expected; only
        # vectorized scoring is deterministic-patched here, NOT the RLS DB.
        monkeypatch.setattr(app.state, "model", object())
        monkeypatch.setattr(
            api_main, "predict_many",
            lambda model, profiles: [(0.8, 1, 0.5), (0.1, 0, 0.5), (0.7, 1, 0.5)],
        )
        own = owner.post(
            f"{BASE}/organizations", json={"name": "Synthetic A"},
            headers={"Origin": "https://testserver"},
        )
        other = outsider.post(
            f"{BASE}/organizations", json={"name": "Synthetic B"},
            headers={"Origin": "https://testserver"},
        )
        assert own.status_code == 201, own.text
        assert other.status_code == 201, other.text
        tenant_a, tenant_b = own.json()["tenant_id"], other.json()["tenant_id"]

        assert sample(viewer, tenant_a).status_code == 403
        assert viewer.get(f"{BASE}/organizations/{tenant_a}/runs").status_code == 403
        assert owner.get(f"{BASE}/organizations/{tenant_b}/runs").status_code == 403
        assert sample(owner, tenant_b).status_code == 403

        created = sample(owner, tenant_a)
        assert created.status_code == 201, created.text
        response = created.json()
        assert response["synthetic_only"] is True
        assert response["summary"]["total_customers"] == 3
        assert response["summary"]["high_risk_customers"] == 2
        assert len(response["id"]) == 36
        assert "customers" not in response

        runs = owner.get(f"{BASE}/organizations/{tenant_a}/runs")
        assert runs.status_code == 200, runs.text
        assert len(runs.json()["runs"]) == 1
        assert runs.json()["runs"][0]["high_risk_customers"] == 2
        assert runs.json()["runs"][0]["total_profiles"] == 3
        assert "created_by_subject" not in str(runs.json())

        issued = owner.post(
            f"{BASE}/organizations/{tenant_a}/invitations",
            json={"role": "viewer"}, headers={"Origin": "https://testserver"},
        )
        assert issued.status_code == 201
        code = issued.json()["code"]
        joined = viewer.post(
            f"{BASE}/invitations/accept", json={"code": code},
            headers={"Origin": "https://testserver"},
        )
        assert joined.status_code == 200, joined.text
        assert viewer.get(f"{BASE}/organizations/{tenant_a}/runs").status_code == 200
        assert sample(viewer, tenant_a).status_code == 403
        assert owner.patch(
            f"{BASE}/organizations/{tenant_a}/members/{viewer_name}",
            json={"role": "analyst"}, headers={"Origin": "https://testserver"},
        ).status_code == 200
        assert sample(viewer, tenant_a).status_code == 201
        assert len(owner.get(f"{BASE}/organizations/{tenant_a}/runs").json()["runs"]) == 2
        assert outsider.get(f"{BASE}/organizations/{tenant_b}/runs").json()["runs"] == []

        assert owner.delete(
            f"{BASE}/organizations/{tenant_a}/members/{viewer_name}",
            headers={"Origin": "https://testserver"},
        ).status_code == 200
        assert viewer.get(f"{BASE}/organizations/{tenant_a}/runs").status_code == 403
        assert sample(viewer, tenant_a).status_code == 403

        with psycopg.connect(os.environ["CHURN_TEST_DB_ADMIN_URL"]) as db:
            row = db.execute(
                "SELECT count(*), sum(total_profiles), sum(over_threshold) "
                "FROM saas.sandbox_runs WHERE tenant_id=%s", (tenant_a,)
            ).fetchone()
            assert row == (2, 6, 4)
        with psycopg.connect(os.environ["CHURN_TEST_DB_APP_URL"]) as db:
            assert db.execute("SELECT count(*) FROM saas.sandbox_runs").fetchone()[0] == 0


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"), reason="Ephemeral PostgreSQL required")
def test_missing_model_does_not_create_history(config, monkeypatch):
    user = "no-model-" + str(uuid4())
    with browser(user) as owner:
        # Startup could load an artifact on another CI runner; force absent.
        monkeypatch.setattr(app.state, "model", None)
        result = owner.post(
            f"{BASE}/organizations", json={"name": "Unavailable model"},
            headers={"Origin": "https://testserver"},
        )
        assert result.status_code == 201, result.text
        tenant = result.json()["tenant_id"]
        assert sample(owner, tenant).status_code == 503
        assert owner.get(f"{BASE}/organizations/{tenant}/runs").json()["runs"] == []
