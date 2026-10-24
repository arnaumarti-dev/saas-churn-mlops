"""Opt-in browser Code+PKCE login tests; no real IdP or customer data."""
import base64
import hashlib
import os
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

jwt = pytest.importorskip("jwt")
psycopg = pytest.importorskip("psycopg")
cryptography = pytest.importorskip("cryptography")
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa

from churn_mlops.api.main import app
from churn_mlops.api import staging_login as login


ISSUER = "https://login.example.org/"
CLIENT_ID = "churn-staging-browser"
ORG_API = "churn-staging-api"


@pytest.fixture
def oidc(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    values = {
        "CHURN_SAAS_LOGIN_ENABLED": "1",
        "CHURN_SAAS_OIDC_ENABLED": "1",
        "CHURN_SAAS_OIDC_ISSUER": ISSUER,
        "CHURN_SAAS_OIDC_AUDIENCE": ORG_API,
        "CHURN_SAAS_OIDC_JWKS_URL": "https://login.example.org/.well-known/jwks.json",
        "CHURN_SAAS_OIDC_CLIENT_ID": CLIENT_ID,
        "CHURN_SAAS_AUTHORIZATION_URL": "https://login.example.org/authorize",
        "CHURN_SAAS_TOKEN_URL": "https://login.example.org/oauth/token",
        "CHURN_SAAS_REDIRECT_URI": "https://testserver/saas/v3/auth/callback",
        "CHURN_SAAS_DATABASE_URL": "postgresql://unused.invalid/test",
        "CHURN_SAAS_SESSION_FERNET_KEY": Fernet.generate_key().decode(),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(login, "_trusted_jwks_client", lambda _:
        SimpleNamespace(get_signing_key_from_jwt=lambda _token:
            SimpleNamespace(key=key.public_key())))

    def mint(nonce, **change):
        now = int(time.time())
        claims = {
            "iss": ISSUER, "aud": CLIENT_ID, "sub": "alice-demo",
            "iat": now, "exp": now + 300, "nonce": nonce, **change
        }
        return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "demo"})

    return SimpleNamespace(mint=mint, private=key, values=values)


def secure_client():
    return TestClient(app, base_url="https://testserver", follow_redirects=False)


def login_started(client):
    response = client.get("/saas/v3/auth/start")
    assert response.status_code == 302
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert query["response_type"] == ["code"]
    assert query["scope"] == ["openid"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["client_id"] == [CLIENT_ID]
    assert query["redirect_uri"] == ["https://testserver/saas/v3/auth/callback"]
    assert "Secure" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    assert "__Host-churn_oidc_tx" in response.headers["set-cookie"]
    return query


def complete_login(client, oidc, monkeypatch, *, nonce_override=None):
    query = login_started(client)
    def exchange(cfg, code, verifier):
        assert cfg.token_url == "https://login.example.org/oauth/token"
        assert code == "one-time-test-code"
        computed = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        assert computed == query["code_challenge"][0]
        return {"id_token": oidc.mint(
            query["nonce"][0] if nonce_override is None else nonce_override
        )}
    monkeypatch.setattr(login, "_exchange_code", exchange)
    response = client.get(
        "/saas/v3/auth/callback",
        params={"code": "one-time-test-code", "state": query["state"][0]},
    )
    return response


def test_disabled_browser_login_and_selector_return_404(monkeypatch):
    monkeypatch.delenv("CHURN_SAAS_LOGIN_ENABLED", raising=False)
    with secure_client() as client:
        for path in ("/saas/v3/login", "/saas/v3/auth/start", "/saas/v3/organizations"):
            response = client.get(path)
            assert response.status_code == 404
            assert response.headers.get("cache-control") == "no-store"


def test_login_code_pkce_cookie_session_and_logout(oidc, monkeypatch):
    with secure_client() as client:
        no_login = client.get("/saas/v3/organizations")
        assert no_login.status_code == 401
        assert client.get("/saas/v3/login").status_code == 200
        completed = complete_login(client, oidc, monkeypatch)
        assert completed.status_code == 303
        assert completed.headers["location"] == "/saas/v3/organizations"
        assert "Secure" in completed.headers["set-cookie"]
        assert "__Host-churn_saas_session" in completed.headers["set-cookie"]
        monkeypatch.setattr(login, "_my_orgs", lambda principal: [
            {"name": "<script>alert('x')</script>", "role": "owner",
             "tenant_id": str(uuid4())}
        ] if principal.subject == "alice-demo" else [])
        organization_page = client.get("/saas/v3/organizations")
        assert organization_page.status_code == 200
        assert organization_page.headers["cache-control"] == "no-store"
        assert "&lt;script&gt;" in organization_page.text
        assert "<script>" not in organization_page.text
        assert "alice-demo" not in organization_page.text
        assert client.post("/saas/v3/logout", headers={"Origin": "https://evil.example"}).status_code == 403
        logout = client.post("/saas/v3/logout", headers={"Origin": "https://testserver"})
        assert logout.status_code == 303
        assert client.get("/saas/v3/organizations").status_code == 401


def test_reject_bad_state_nonce_tampering_and_wrong_id_token(oidc, monkeypatch):
    with secure_client() as client:
        query = login_started(client)
        monkeypatch.setattr(login, "_exchange_code", lambda *_: pytest.fail("should not exchange code"))
        wrong_state = client.get(
            "/saas/v3/auth/callback", params={"code": "code", "state": "wrong"})
        assert wrong_state.status_code == 401
        bad_nonce = complete_login(client, oidc, monkeypatch, nonce_override="wrong")
        assert bad_nonce.status_code == 401
        query = login_started(client)
        # Explicit request header overrides the test client's existing
        # host-only cookie; do not accidentally test a second valid cookie.
        bad_cookie = client.get(
            "/saas/v3/auth/callback",
            params={"code": "code", "state": query["state"][0]},
            headers={"Cookie": login.STATE_COOKIE + "=tampered"},
        )
        assert bad_cookie.status_code == 401


def test_login_configuration_fails_closed(oidc, monkeypatch):
    with secure_client() as client:
        monkeypatch.setenv("CHURN_SAAS_AUTHORIZATION_URL", "https://evil.example.org/start")
        assert client.get("/saas/v3/auth/start").status_code == 503
        monkeypatch.setenv("CHURN_SAAS_AUTHORIZATION_URL", "https://login.example.org/authorize")
        monkeypatch.setenv("CHURN_SAAS_SESSION_FERNET_KEY", "not-a-key")
        assert client.get("/saas/v3/auth/start").status_code == 503
        monkeypatch.setenv("CHURN_SAAS_SESSION_FERNET_KEY", oidc.values["CHURN_SAAS_SESSION_FERNET_KEY"])
        monkeypatch.setenv("CHURN_SAAS_OIDC_ENABLED", "0")
        assert client.get("/saas/v3/auth/start").status_code == 404


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"),
                    reason="Ephemeral PostgreSQL only in integration CI")
def test_organization_selector_isolated_by_identity_and_revocation(oidc, monkeypatch):
    monkeypatch.setenv("CHURN_SAAS_DATABASE_URL", os.environ["CHURN_TEST_DB_APP_URL"])
    a, b, c = uuid4(), uuid4(), uuid4()
    with psycopg.connect(os.environ["CHURN_TEST_DB_ADMIN_URL"]) as conn:
        for tenant, name in [(a, "Empresa A"), (b, "Empresa B"), (c, "Oculta")]:
            conn.execute("INSERT INTO saas.organizations (id, display_name) VALUES (%s,%s)",
                         (tenant, name))
        for tenant, subject, role, revoked in [
            (a, "alice-demo", "owner", False),
            (b, "alice-demo", "viewer", False),
            (c, "alice-demo", "owner", True),
            (a, "other-person", "analyst", False),
        ]:
            conn.execute(
                """INSERT INTO saas.memberships
                   (tenant_id,oidc_issuer,oidc_subject,role,revoked_at)
                   VALUES (%s,%s,%s,%s,CASE WHEN %s THEN now() ELSE NULL END)""",
                (tenant, ISSUER, subject, role, revoked),
            )
    names = login._my_orgs(login.Principal(ISSUER, "alice-demo"))
    assert {item["name"] for item in names} == {"Empresa A", "Empresa B"}
    assert {item["role"] for item in names} == {"owner", "viewer"}
    assert login._my_orgs(login.Principal(ISSUER, "other-person")) == [
        {"tenant_id": str(a), "name": "Empresa A", "role": "analyst"}
    ]
    assert login._my_orgs(login.Principal(ISSUER, "unknown-person")) == []
    # Current connection-local role context must not leak to the next connection.
    with psycopg.connect(os.environ["CHURN_TEST_DB_APP_URL"]) as conn:
        assert conn.execute("SELECT count(*) FROM saas.memberships").fetchone()[0] == 0
