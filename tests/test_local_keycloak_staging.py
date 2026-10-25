"""Offline checks for localhost-only Keycloak staging configuration.

Tests never connect to the network, log in users, or enable SaaS on Render.
"""
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from churn_mlops.api import oidc_postgres as auth

ROOT = Path(__file__).resolve().parents[1]


def _oidc_env(monkeypatch, *, issuer, jwks):
    monkeypatch.setenv(auth.FLAG, "1")
    monkeypatch.setenv(auth.ISSUER, issuer)
    monkeypatch.setenv(auth.JWKS, jwks)
    monkeypatch.setenv(auth.AUDIENCE, "churn-staging-api")
    monkeypatch.setenv(auth.DATABASE, "postgresql://placeholder.invalid/staging")


def test_keycloak_realm_exact_jwks_endpoint_is_allowed(monkeypatch):
    issuer = "https://idp.localhost/realms/churn-staging"
    jwks = issuer + "/protocol/openid-connect/certs"
    _oidc_env(monkeypatch, issuer=issuer, jwks=jwks)
    settings = auth.settings()
    assert settings.issuer == issuer
    assert settings.jwks_url == jwks


@pytest.mark.parametrize("jwks", [
    "http://idp.localhost/realms/churn-staging/protocol/openid-connect/certs",
    "https://evil.localhost/realms/churn-staging/protocol/openid-connect/certs",
    "https://idp.localhost/realms/another-realm/protocol/openid-connect/certs",
    "https://idp.localhost/realms/churn-staging/../other/protocol/openid-connect/certs",
    "https://idp.localhost:8443/realms/churn-staging/protocol/openid-connect/certs",
    "https://idp.localhost/realms/churn-staging/protocol/openid-connect/certs?x=y",
    "https://idp.localhost/realms/churn-staging/protocol/openid-connect/certs/extra",
    "https://idp.localhost/realms/churn-staging/.well-known/not-the-keys",
])
def test_reject_unsafe_or_wrong_realm_jwks(monkeypatch, jwks):
    _oidc_env(
        monkeypatch, issuer="https://idp.localhost/realms/churn-staging", jwks=jwks
    )
    with pytest.raises(HTTPException) as exc:
        auth.settings()
    assert exc.value.status_code == 503


def test_keep_existing_well_known_jwks_location(monkeypatch):
    _oidc_env(
        monkeypatch, issuer="https://login.example.org/",
        jwks="https://login.example.org/.well-known/jwks.json",
    )
    assert auth.settings().audience == "churn-staging-api"


def test_realm_is_public_pkce_client_without_seeded_users_or_passwords():
    path = ROOT / "dev/private-staging/realm/churn-staging-realm.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["realm"] == "churn-staging"
    assert content["registrationAllowed"] is False
    assert content["sslRequired"] == "all"
    assert "users" not in content  # demo accounts created manually in private admin UI
    assert len(content["clients"]) == 1
    client = content["clients"][0]
    assert client["clientId"] == "churn-staging-browser"
    assert client["publicClient"] is True
    assert client["standardFlowEnabled"] is True
    assert client["directAccessGrantsEnabled"] is False
    assert client["implicitFlowEnabled"] is False
    assert client["attributes"]["pkce.code.challenge.method"] == "S256"
    assert client["redirectUris"] == [
        "https://app.localhost/saas/v3/auth/callback"
    ]
    assert client["webOrigins"] == ["https://app.localhost"]


def test_staging_only_local_port_and_no_committed_credentials():
    compose = (ROOT / "compose.private-staging.yml").read_text(encoding="utf-8")
    assert '127.0.0.1:443:443' in compose
    assert 'CHURN_SAAS_MANAGEMENT_ENABLED: "1"' in compose
    assert 'CHURN_SAAS_DATABASE_URL: postgresql://churn_saas_app:' in compose
    assert 'SSL_CERT_FILE: /run/private-staging-ca/root.crt' in compose
    assert 'dev/private-staging/local-trust:/run/private-staging-ca:ro' in compose
    assert 'ports:' in compose
    assert compose.count('    ports:') == 1
    example = (ROOT / ".env.staging.example").read_text(encoding="utf-8")
    assert "REPLACE_WITH_" in example
    assert "CHURN_SAAS_SESSION_FERNET_KEY=" in example
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env.staging.local" in ignore
    assert "dev/private-staging/local-trust/*.crt" in ignore
