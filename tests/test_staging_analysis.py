"""Tenant-scoped synthetic scoring and only aggregate run history.

All people, organization names and scoring profiles are fixtures, never
customer records. PostgreSQL integration requires ephemeral CI database.
"""
import hashlib
import os
import time
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("jwt")
from cryptography.fernet import Fernet

from churn_mlops.api.main import app
from churn_mlops.api import main as api_main
from churn_mlops.api import staging_login as login
from churn_mlops.api import staging_analysis as analysis

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


def compare(client, tenant, payload=None):
    return client.post(
        f"{BASE}/organizations/{tenant}/runs/compare",
        json={} if payload is None else payload,
        headers={"Origin": "https://testserver"},
    )


def test_public_demo_cannot_access_sample_routes(monkeypatch):
    monkeypatch.delenv("CHURN_SAAS_MANAGEMENT_ENABLED", raising=False)
    tenant = uuid4()
    with TestClient(app, base_url="https://testserver") as guest:
        assert guest.get(f"{BASE}/organizations/{tenant}/runs").status_code == 404
        assert sample(guest, tenant).status_code == 404
        assert compare(guest, tenant).status_code == 404


def test_valid_session_and_same_origin_required(config):
    tenant = uuid4()
    with TestClient(app, base_url="https://testserver") as guest:
        assert guest.get(f"{BASE}/organizations/{tenant}/runs").status_code == 401
        assert sample(guest, tenant).status_code == 401
        assert compare(guest, tenant).status_code == 401
    with browser("staging-actor-" + str(uuid4())) as actor:
        assert actor.post(
            f"{BASE}/organizations/{tenant}/runs/sample", json={}
        ).status_code == 403
        assert actor.post(
            f"{BASE}/organizations/{tenant}/runs/sample", json={},
            headers={"Origin": "https://not-trusted.example"}
        ).status_code == 403
        assert sample(actor, tenant, {"customers": []}).status_code == 422
        assert sample(actor, tenant, {"scenario": "invented"}).status_code == 422
        assert sample(actor, tenant, {"scenario": "loyal", "customers": []}).status_code == 422
        assert compare(actor, tenant, {"customers": []}).status_code == 422
        assert compare(actor, tenant, {"scenario": "loyal"}).status_code == 422
        assert actor.post(
            f"{BASE}/organizations/{tenant}/runs/compare", json={}
        ).status_code == 403


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"), reason="Ephemeral PostgreSQL required")
def test_synthetic_runs_are_tenant_scoped_and_role_authorized(config, monkeypatch):
    owner_name = "owner-" + str(uuid4())
    viewer_name = "viewer-" + str(uuid4())
    outsider_name = "outsider-" + str(uuid4())

    with browser(owner_name) as owner, browser(viewer_name) as viewer, browser(outsider_name) as outsider:
        # FastAPI model startup without fixture artifacts is expected; only
        # vectorized scoring is deterministic-patched here, NOT the RLS DB.
        monkeypatch.setattr(app.state, "model", object())
        processed_tenures = []

        def fake_predict_many(model, profiles):
            processed_tenures.append([profile["tenure"] for profile in profiles])
            return [(0.8, 1, 0.5), (0.1, 0, 0.5), (0.7, 1, 0.5)]

        monkeypatch.setattr(api_main, "predict_many", fake_predict_many)
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
        assert response["scenario"] == "baseline"
        assert response["summary"]["total_customers"] == 3
        assert response["summary"]["high_risk_customers"] == 2
        assert len(response["id"]) == 36
        assert "customers" not in response

        runs = owner.get(f"{BASE}/organizations/{tenant_a}/runs")
        assert runs.status_code == 200, runs.text
        assert len(runs.json()["runs"]) == 1
        assert runs.json()["runs"][0]["high_risk_customers"] == 2
        assert runs.json()["runs"][0]["total_profiles"] == 3
        assert runs.json()["runs"][0]["scenario"] == "baseline"
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
        loyal_run = sample(viewer, tenant_a, {"scenario": "loyal"})
        assert loyal_run.status_code == 201
        assert loyal_run.json()["scenario"] == "loyal"
        assert len(owner.get(f"{BASE}/organizations/{tenant_a}/runs").json()["runs"]) == 2
        risky_run = sample(owner, tenant_a, {"scenario": "at_risk"})
        assert risky_run.status_code == 201
        assert risky_run.json()["scenario"] == "at_risk"
        keys = [run["scenario"] for run in owner.get(
            f"{BASE}/organizations/{tenant_a}/runs").json()["runs"]]
        assert set(keys) == {"baseline", "loyal", "at_risk"}
        assert [tenures[0] for tenures in processed_tenures] == [2, 48, 1]
        # The API actually scores different fixed example rows, not merely
        # relabels identical requests in the aggregate history.
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
            assert row == (3, 9, 6)
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


def test_server_curated_scenarios_are_distinct_and_have_no_client_fields():
    assert set(analysis.SYNTHETIC_SCENARIOS) == {"baseline", "at_risk", "loyal"}
    scenarios = analysis.SYNTHETIC_SCENARIOS
    assert all(len(item.customers) == 3 for item in scenarios.values())
    first_tenures = [scenarios[k].customers[0].tenure
                     for k in ("baseline", "at_risk", "loyal")]
    assert first_tenures == [2, 1, 48]
    assert all(customer.MonthlyCharges >= 0 and customer.TotalCharges >= 0
               for scenario in scenarios.values() for customer in scenario.customers)
    assert analysis.EmptySyntheticRun().scenario == "baseline"


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"), reason="Ephemeral PostgreSQL required")
def test_atomic_three_scenario_comparison_is_tenant_scoped(config, monkeypatch):
    owner_subject = "compare-owner-" + str(uuid4())
    outsider_subject = "compare-outsider-" + str(uuid4())
    viewer_subject = "compare-viewer-" + str(uuid4())
    seen_tenures = []

    def fake_predict_many(model, profiles):
        tenure = profiles[0]["tenure"]
        seen_tenures.append(tenure)
        values = {
            2: (0.8, 0.1, 0.7),
            1: (0.95, 0.9, 0.8),
            48: (0.15, 0.1, 0.05),
        }[tenure]
        return [(prob, int(prob >= 0.5), 0.5) for prob in values]

    with (browser(owner_subject) as owner,
          browser(outsider_subject) as outsider,
          browser(viewer_subject) as viewer):
        monkeypatch.setattr(app.state, "model", object())
        monkeypatch.setattr(api_main, "predict_many", fake_predict_many)
        organization = owner.post(
            f"{BASE}/organizations", json={"name": "Compare Synthetic A"},
            headers={"Origin": "https://testserver"},
        )
        other_org = outsider.post(
            f"{BASE}/organizations", json={"name": "Compare Synthetic B"},
            headers={"Origin": "https://testserver"},
        )
        assert organization.status_code == 201
        assert other_org.status_code == 201
        tenant = organization.json()["tenant_id"]
        foreign_tenant = other_org.json()["tenant_id"]

        assert compare(outsider, tenant).status_code == 403
        assert compare(owner, foreign_tenant).status_code == 403
        assert compare(viewer, tenant).status_code == 403
        assert compare(owner, tenant, {"customers": []}).status_code == 422
        assert compare(owner, tenant, {"scenario": "loyal"}).status_code == 422
        assert owner.post(
            f"{BASE}/organizations/{tenant}/runs/compare", json={}
        ).status_code == 403
        assert owner.get(f"{BASE}/organizations/{tenant}/runs").json()["runs"] == []

        response = compare(owner, tenant)
        assert response.status_code == 201, response.text
        comparison = response.json()
        assert comparison["synthetic_only"] is True
        assert comparison["tenant_id"] == tenant
        assert [item["scenario"] for item in comparison["results"]] == [
            "baseline", "at_risk", "loyal"]
        assert [item["high_risk_customers"] for item in comparison["results"]] == [2, 3, 0]
        assert [item["total_profiles"] for item in comparison["results"]] == [3, 3, 3]
        assert seen_tenures == [2, 1, 48]
        assert set(comparison["results"][0]) == {
            "scenario", "total_profiles", "high_risk_customers",
            "average_churn_probability",
        }
        assert "customers" not in str(comparison)

        history = owner.get(f"{BASE}/organizations/{tenant}/runs")
        assert history.status_code == 200
        assert len(history.json()["runs"]) == 3
        assert {run["scenario"] for run in history.json()["runs"]} == {
            "baseline", "at_risk", "loyal"}
        assert outsider.get(f"{BASE}/organizations/{tenant}/runs").status_code == 403
        assert outsider.get(f"{BASE}/organizations/{foreign_tenant}/runs").json()["runs"] == []

        invite = owner.post(
            f"{BASE}/organizations/{tenant}/invitations",
            json={"role": "viewer"}, headers={"Origin": "https://testserver"},
        )
        assert invite.status_code == 201
        assert viewer.post(
            f"{BASE}/invitations/accept", json={"code": invite.json()["code"]},
            headers={"Origin": "https://testserver"},
        ).status_code == 200
        assert len(viewer.get(f"{BASE}/organizations/{tenant}/runs").json()["runs"]) == 3
        assert compare(viewer, tenant).status_code == 403
        assert owner.patch(
            f"{BASE}/organizations/{tenant}/members/{viewer_subject}",
            json={"role": "analyst"}, headers={"Origin": "https://testserver"},
        ).status_code == 200
        assert compare(viewer, tenant).status_code == 201
        assert len(owner.get(f"{BASE}/organizations/{tenant}/runs").json()["runs"]) == 6

        # Failing on the second scenario MUST roll back the earlier INSERT.
        def fail_after_first(model, profiles):
            if profiles[0]["tenure"] == 1:
                raise HTTPException(status_code=503, detail="Model unavailable")
            return fake_predict_many(model, profiles)

        monkeypatch.setattr(api_main, "predict_many", fail_after_first)
        failed = compare(owner, tenant)
        assert failed.status_code == 503
        assert len(owner.get(f"{BASE}/organizations/{tenant}/runs").json()["runs"]) == 6

        with psycopg.connect(os.environ["CHURN_TEST_DB_ADMIN_URL"]) as db:
            count, total, high = db.execute(
                "SELECT count(*), sum(total_profiles), sum(over_threshold) "
                "FROM saas.sandbox_runs WHERE tenant_id = %s", (tenant,),
            ).fetchone()
            assert (count, total, high) == (6, 18, 10)
        with psycopg.connect(os.environ["CHURN_TEST_DB_APP_URL"]) as db:
            assert db.execute("SELECT count(*) FROM saas.sandbox_runs").fetchone()[0] == 0
