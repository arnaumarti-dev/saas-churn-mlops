"""Opt-in OIDC resource-server validation for a *private staging environment*.

This is not a login UI: a managed identity provider must perform authorization
code + PKCE itself. We verify its signed *access token* on the API, then use
server-side PostgreSQL membership; JWT claims never grant a tenant role.

No OIDC endpoint is enabled in the public portfolio deployment. The optional
PyJWT and psycopg packages are loaded only when this feature is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from urllib.parse import urlsplit

from fastapi import Header, HTTPException


FLAG = "CHURN_SAAS_OIDC_ENABLED"
ISSUER = "CHURN_SAAS_OIDC_ISSUER"
AUDIENCE = "CHURN_SAAS_OIDC_AUDIENCE"
JWKS = "CHURN_SAAS_OIDC_JWKS_URL"
DATABASE = "CHURN_SAAS_DATABASE_URL"


@dataclass(frozen=True)
class OIDCSettings:
    issuer: str
    audience: str
    jwks_url: str
    database_url: str


@dataclass(frozen=True)
class Principal:
    issuer: str
    subject: str


def settings() -> OIDCSettings:
    """Reject insecure/incomplete configuration before examining credentials."""
    if os.environ.get(FLAG, "0") == "0":
        raise HTTPException(status_code=404, detail="Not found")
    if os.environ.get(FLAG) != "1":
        raise HTTPException(status_code=503, detail="Identity service unavailable")
    issuer = os.environ.get(ISSUER, "")
    audience = os.environ.get(AUDIENCE, "")
    jwks_url = os.environ.get(JWKS, "")
    dsn = os.environ.get(DATABASE, "")
    i, j = urlsplit(issuer), urlsplit(jwks_url)
    if (not issuer or not audience or not dsn or len(audience) > 250
            or i.scheme != "https" or j.scheme != "https"
            or not i.hostname or i.hostname != j.hostname
            or i.username or i.password or i.query or i.fragment
            or j.username or j.password or j.fragment or not j.path
            or not j.path.startswith("/.well-known/")
            or i.port not in (None, 443) or j.port not in (None, 443)):
        raise HTTPException(status_code=503, detail="Identity service unavailable")
    return OIDCSettings(issuer, audience, jwks_url, dsn)


@lru_cache(maxsize=8)
def _trusted_jwks_client(jwks_url: str):
    # Only the operator-configured HTTPS URL is passed here, not a JWT header.
    import jwt
    return jwt.PyJWKClient(jwks_url, timeout=4, cache_jwk_set=True, lifespan=300)


def get_oidc_principal(
    authorization: str | None = Header(default=None),
) -> Principal:
    """Verify RS256 signature + registered claims; roles come from DB only."""
    cfg = settings()
    parts = authorization.split() if authorization else []
    if (len(parts) != 2 or parts[0].lower() != "bearer"
            or not 10 <= len(parts[1]) <= 8192):
        raise HTTPException(
            status_code=401, detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        import jwt
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Identity service unavailable") from exc

    try:
        header = jwt.get_unverified_header(parts[1])
        if (header.get("alg") != "RS256" or
                not isinstance(header.get("kid"), str) or
                not 1 <= len(header["kid"]) <= 128):
            raise jwt.InvalidTokenError("Unexpected token signing algorithm")
        jwk = _trusted_jwks_client(cfg.jwks_url).get_signing_key_from_jwt(parts[1])
        # Use the RSA public key, NOT a PyJWK object, to enforce the algorithm
        # allowlist explicitly (PyJWT versions before 2.13 had a bypass issue).
        claims = jwt.decode(
            parts[1], jwk.key, algorithms=["RS256"],
            issuer=cfg.issuer, audience=cfg.audience,
            options={"require": ["iss", "aud", "exp", "iat", "sub"]},
            leeway=10,
        )
        subject = claims["sub"]
        if not isinstance(subject, str) or not 1 <= len(subject) <= 255:
            raise jwt.InvalidTokenError("Invalid subject")
        return Principal(issuer=cfg.issuer, subject=subject)
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=401, detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        # JWKS infrastructure errors must fail closed; never leak keys or URLs.
        raise HTTPException(status_code=503, detail="Identity service unavailable") from exc


def membership_role(tenant_id, principal: Principal) -> str:
    """Resolve membership on a tenant-scoped, transaction-local RLS connection."""
    cfg = settings()
    try:
        import psycopg
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Tenant database unavailable") from exc

    try:
        with psycopg.connect(cfg.database_url, connect_timeout=4) as conn:
            with conn.transaction():
                conn.execute(
                    "SELECT set_config('app.tenant_id', %s, true)",
                    (str(tenant_id),),
                )
                row = conn.execute(
                    """SELECT role FROM saas.memberships
                       WHERE tenant_id = %s AND oidc_issuer = %s
                         AND oidc_subject = %s AND revoked_at IS NULL""",
                    (tenant_id, principal.issuer, principal.subject),
                ).fetchone()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Tenant database unavailable") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Not found")
    return row[0]
