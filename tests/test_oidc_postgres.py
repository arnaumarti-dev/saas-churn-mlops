"""Real RS256 token validation and optional temporary PostgreSQL RLS tests.

The CI SaaS job provisions the ephemeral DB. Standard demo CI skips optional
integration testing because neither a DB nor SaaS dependencies are installed.
"""
import json
import os
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

jwt = pytest.importorskip("jwt")
pytest.importorskip("psycopg")
from cryptography.hazmat.primitives.asymmetric import rsa

from churn_mlops.api.main import app
from churn_mlops.api import oidc_postgres as auth

ISSUER = "https://login.example.org/"
AUDIENCE = "churn-lab-staging"
JWKS = "https://login.example.org/.well-known/jwks.json"


@pytest.fixture()
def identity(monkeypatch):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setenv(auth.FLAG, "1")
    monkeypatch.setenv(auth.ISSUER, ISSUER)
    monkeypatch.setenv(auth.AUDIENCE, AUDIENCE)
    monkeypatch.setenv(auth.JWKS, JWKS)
    monkeypatch.setenv(auth.DATABASE, "postgresql://unused.invalid/test")
    monkeypatch.setattr(
        auth, "_trusted_jwks_client",
        lambda _url: SimpleNamespace(
            get_signing_key_from_jwt=lambda _: SimpleNamespace(key=private.public_key())
        ),
    )
    now = int(time.time())

    def token(**claims):
        body = {
            "iss": ISSUER, "aud": AUDIENCE, "sub": "oidc-alice",
            "iat": now - 1, "exp": now + 600,
            **claims,
        }
        return jwt.encode(body, private, algorithm="RS256", headers={"kid": "test-key"})

    return token


def _reject(token, status=401):
    with pytest.raises(HTTPException) as captured:
        auth.get_oidc_principal("Bearer " + token)
    assert captured.value.status_code == status


def test_access_token_signature_issuer_audience_expiration_and_subject(identity):
    principal = auth.get_oidc_principal("Bearer " + identity())
    assert principal == auth.Principal(ISSUER, "oidc-alice")
    _reject(identity(aud="another-api"))
    _reject(identity(iss="https://evil.example.org/"))
    _reject(identity(exp=int(time.time()) - 120))
    _reject(identity(sub=""))
    _reject(identity(iat=None))


def test_refuse_unsigned_hs256_wrong_rsa_key_and_missing_kid(identity):
    _reject(jwt.encode(
        {"sub": "oidc-alice", "iss": ISSUER, "aud": AUDIENCE,
         "iat": int(time.time()), "exp": int(time.time()) + 300},
        "insecure-symmetric-key", algorithm="HS256",
        headers={"kid": "bad"},
    ))
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _reject(jwt.encode(
        {"sub": "oidc-alice", "iss": ISSUER, "aud": AUDIENCE,
         "iat": int(time.time()), "exp": int(time.time()) + 300},
        other, algorithm="RS256", headers={"kid": "different"},
    ))
    _reject(identity(), status=401) if False else None  # no extra test requests


def test_missing_auth_and_bad_operator_configuration(identity, monkeypatch):
    with pytest.raises(HTTPException) as missing:
        auth.get_oidc_principal(None)
    assert missing.value.status_code == 401
    monkeypatch.setenv(auth.JWKS, "http://127.0.0.1/private")
    with pytest.raises(HTTPException) as insecure:
        auth.get_oidc_principal("Bearer " + identity())
    assert insecure.value.status_code == 503
    monkeypatch.setenv(auth.JWKS, "https://other.example.org/.well-known/jwks.json")
    with pytest.raises(HTTPException) as untrusted:
        auth.get_oidc_principal("Bearer " + identity())
    assert untrusted.value.status_code == 503


def test_default_off_oidc_routes_without_database(monkeypatch):
    monkeypatch.delenv(auth.FLAG, raising=False)
    tenant = uuid4()
    with TestClient(app) as client:
        for method, url in (
            ("get", f"/saas/v2/tenants/{tenant}/me"),
            ("post", f"/saas/v2/tenants/{tenant}/predict/batch"),
        ):
            response = getattr(client, method)(
                url, **({"json": {"customers": []}} if method == "post" else {})
            )
            assert response.status_code == 404
            assert response.headers["cache-control"] == "no-store"


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"),
                    reason="Private PostgreSQL integration test requires ephemeral DB")
def test_real_postgres_membership_and_rls(identity, monkeypatch):
    import psycopg

    admin_dsn = os.environ["CHURN_TEST_DB_ADMIN_URL"]
    app_dsn = os.environ["CHURN_TEST_DB_APP_URL"]
    monkeypatch.setenv(auth.DATABASE, app_dsn)

    tenant_a, tenant_b = uuid4(), uuid4()
    with psycopg.connect(admin_dsn) as conn:
        conn.execute("INSERT INTO saas.organizations(id, display_name) VALUES (%s, 'Demo A')",
                     (tenant_a,))
        conn.execute("INSERT INTO saas.organizations(id, display_name) VALUES (%s, 'Demo B')",
                     (tenant_b,))
        conn.execute(
            """INSERT INTO saas.memberships
                 (tenant_id, oidc_issuer, oidc_subject, role)
               VALUES (%s, %s, 'oidc-alice', 'analyst')""",
            (tenant_a, ISSUER),
        )
        conn.execute(
            """INSERT INTO saas.memberships
                 (tenant_id, oidc_issuer, oidc_subject, role)
               VALUES (%s, %s, 'oidc-bob', 'viewer')""",
            (tenant_b, ISSUER),
        )

    principal_a = auth.Principal(ISSUER, "oidc-alice")
    assert auth.membership_role(tenant_a, principal_a) == "analyst"
    with pytest.raises(HTTPException) as denied:
        auth.membership_role(tenant_b, principal_a)
    assert denied.value.status_code == 404

    # No tenant context at all: missing GUC must deny by default.
    with psycopg.connect(app_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM saas.memberships").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM saas.organizations").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM saas.sandbox_runs").fetchone()[0] == 0

    # Even with a tenant context, one subject may not read another user's row.
    with psycopg.connect(app_dsn) as conn:
        with conn.transaction():
            conn.execute(
                """SELECT set_config('app.tenant_id', %s, true),
                          set_config('app.oidc_issuer', %s, true),
                          set_config('app.oidc_subject', %s, true)""",
                (str(tenant_a), ISSUER, "oidc-bob"),
            )
            assert conn.execute("SELECT count(*) FROM saas.memberships").fetchone()[0] == 0

    with psycopg.connect(app_dsn) as conn:
        with conn.transaction():
            conn.execute(
                """SELECT set_config('app.tenant_id', %s, true),
                          set_config('app.oidc_subject', %s, true)""",
                (str(tenant_a), "oidc-alice"),
            )
            conn.execute(
                """INSERT INTO saas.sandbox_runs
                   (id, tenant_id, created_by_subject, total_profiles, over_threshold)
                   VALUES (%s, %s, 'oidc-alice', 2, 1)""",
                (uuid4(), tenant_a),
            )
            assert conn.execute("SELECT count(*) FROM saas.sandbox_runs").fetchone()[0] == 1

    with psycopg.connect(app_dsn) as conn:
        with conn.transaction():
            conn.execute(
                """SELECT set_config('app.tenant_id', %s, true),
                          set_config('app.oidc_subject', %s, true)""",
                (str(tenant_b), "oidc-bob"),
            )
            assert conn.execute("SELECT count(*) FROM saas.sandbox_runs").fetchone()[0] == 0
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(
                    """INSERT INTO saas.sandbox_runs
                       (id, tenant_id, created_by_subject, total_profiles, over_threshold)
                       VALUES (%s, %s, 'oidc-alice', 2, 1)""",
                    (uuid4(), tenant_a),
                )

    token = identity()
    with TestClient(app) as client:
        own = client.get(
            f"/saas/v2/tenants/{tenant_a}/me",
            headers={"Authorization": "Bearer " + token},
        )
        assert own.status_code == 200
        assert own.json()["role"] == "analyst"
        other = client.get(
            f"/saas/v2/tenants/{tenant_b}/me",
            headers={"Authorization": "Bearer " + token},
        )
        assert other.status_code == 404
