"""Gated private organization console: no public assets or sensitive interpolation."""
from __future__ import annotations

import hashlib
import os
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("jwt")
pytest.importorskip("psycopg")
from cryptography.fernet import Fernet

from churn_mlops.api.main import app
from churn_mlops.api import staging_login as login


BASE = "/saas/v3"
STAGING_PAGES = ["/console", "/console.js", "/console.css", "/organizations/list"]


@pytest.fixture
def config(monkeypatch):
    options = {
        "CHURN_SAAS_MANAGEMENT_ENABLED": "1",
        "CHURN_SAAS_LOGIN_ENABLED": "1",
        "CHURN_SAAS_OIDC_ENABLED": "1",
        "CHURN_SAAS_OIDC_ISSUER": "https://login.example.org/",
        "CHURN_SAAS_OIDC_AUDIENCE": "churn-console-api",
        "CHURN_SAAS_OIDC_JWKS_URL": "https://login.example.org/.well-known/jwks.json",
        "CHURN_SAAS_OIDC_CLIENT_ID": "churn-console-client",
        "CHURN_SAAS_AUTHORIZATION_URL": "https://login.example.org/authorize",
        "CHURN_SAAS_TOKEN_URL": "https://login.example.org/oauth/token",
        "CHURN_SAAS_REDIRECT_URI": "https://testserver/saas/v3/auth/callback",
        "CHURN_SAAS_DATABASE_URL": os.getenv("CHURN_TEST_DB_APP_URL", "postgresql://unused.invalid/test"),
        "CHURN_SAAS_SESSION_FERNET_KEY": Fernet.generate_key().decode("ascii"),
    }
    for name, value in options.items():
        monkeypatch.setenv(name, value)
    return options


def browser(subject):
    digest = hashlib.sha256(subject.encode()).digest()
    client = TestClient(
        app, base_url="https://testserver", follow_redirects=False,
        client=(f"10.{digest[0]}.{digest[1]}.{digest[2]}", 50000),
    )
    cfg = login._login_config()
    client.cookies.set(
        login.SESSION_COOKIE,
        login._seal(cfg, {
            "iss": "https://login.example.org/",
            "sub": subject,
            "exp": int(time.time()) + 500,
        }),
        path="/",
    )
    return client


def test_default_off_hides_console_and_assets(monkeypatch):
    monkeypatch.delenv("CHURN_SAAS_MANAGEMENT_ENABLED", raising=False)
    with TestClient(app, base_url="https://testserver") as client:
        for path in STAGING_PAGES:
            response = client.get(BASE + path)
            assert response.status_code == 404, (path, response.text)
            assert response.headers["cache-control"] == "no-store"


def test_console_requires_login_and_is_strictly_private(config):
    with TestClient(app, base_url="https://testserver") as guest:
        for path in STAGING_PAGES:
            response = guest.get(BASE + path)
            assert response.status_code == 401, (path, response.text)
    with browser("console-" + str(uuid4())) as client:
        html = client.get(BASE + "/console")
        assert html.status_code == 200, html.text
        assert "text/html" in html.headers["content-type"]
        assert html.headers["cache-control"] == "no-store"
        assert "Churn Lab" in html.text
        assert "/saas/v3/console.js" in html.text
        assert "/saas/v3/console.css" in html.text
        assert 'id="issue-invite"' in html.text
        assert 'id="create-org"' in html.text
        assert 'id="overview-tools"' in html.text
        assert 'id="overview-runs"' in html.text
        assert 'id="overview-profiles"' in html.text
        assert 'id="overview-rate"' in html.text
        assert 'id="redeem-invite"' in html.text
        assert 'id="alias-tools"' in html.text
        assert 'id="alias-input"' in html.text
        assert 'id="save-alias"' in html.text
        assert "console-" not in html.text  # never embed OIDC subjects
        assert "noindex" in html.text
        assert "<script defer src=" in html.text
        for asset, kind in (("console.js", "javascript"), ("console.css", "css")):
            response = client.get(BASE + "/" + asset)
            assert response.status_code == 200
            assert kind in response.headers["content-type"]
            assert response.headers["cache-control"] == "no-store"
        js = client.get(BASE + "/console.js").text
        assert ".textContent" in js
        assert ".innerHTML" not in js
        assert "localStorage" not in js
        assert "sessionStorage" not in js
        assert "window.confirm" in js
        assert "currentId !== tenantId" in js
        assert "renderOverview(result.runs)" in js
        assert 'member.alias || "Miembro sin alias' in js
        assert "/me/alias" in js
        assert "credentials: \"same-origin\"" in js


def test_invalid_session_cannot_load_assets(config):
    with browser("tamper-" + str(uuid4())) as client:
        for path in STAGING_PAGES[:3]:
            # Override the existing host cookie rather than add another
            # cookie with potentially different test-client domain scoping.
            response = client.get(
                BASE + path,
                headers={"Cookie": login.SESSION_COOKIE + "=corrupted"},
            )
            assert response.status_code == 401


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"), reason="CI ephemeral PostgreSQL required")
def test_console_uses_live_self_memberships_and_owner_permissions(config):
    name_a, name_b = "console-A-" + str(uuid4()), "console-B-" + str(uuid4())
    with browser(name_a) as a, browser(name_b) as b:
        created = a.post(
            BASE + "/organizations", json={"name": "<Demo & Co>"},
            headers={"Origin": "https://testserver"},
        )
        assert created.status_code == 201, created.text
        uuid = created.json()["tenant_id"]

        own = a.get(BASE + "/organizations/list")
        assert own.status_code == 200, own.text
        assert own.headers["cache-control"] == "no-store"
        assert any(o["tenant_id"] == uuid and o["role"] == "owner"
                   and o["name"] == "<Demo & Co>" for o in own.json()["organizations"])

        foreign = b.get(BASE + "/organizations/list")
        assert foreign.status_code == 200
        assert not any(o["tenant_id"] == uuid for o in foreign.json()["organizations"])
        assert "<Demo & Co>" not in b.get(BASE + "/console").text

        # Signed-in owner can view own member list, nonmember is rejected.
        assert a.get(BASE + "/organizations/" + uuid + "/members").status_code == 200
        assert b.get(BASE + "/organizations/" + uuid + "/members").status_code == 403
