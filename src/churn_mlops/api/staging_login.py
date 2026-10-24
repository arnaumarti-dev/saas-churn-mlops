"""Opt-in staging-only browser OIDC Code + PKCE login and organization selector.

No passwords, customer records or OAuth tokens are persisted. Login transaction
state and short-lived session are sealed in Secure HttpOnly cookies using
Fernet. Staging only: this lacks central logout/revocation, durable sessions,
production CSRF infrastructure, rate-limiting at the edge and a real IdP.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request as URLRequest, build_opener
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from churn_mlops.api.oidc_postgres import Principal, _trusted_jwks_client, settings


router = APIRouter(prefix="/saas/v3")
STATE_COOKIE = "__Host-churn_oidc_tx"
SESSION_COOKIE = "__Host-churn_saas_session"
TX_LIFETIME = 300
SESSION_LIFETIME = 900


@dataclass(frozen=True)
class LoginConfig:
    issuer: str
    client_id: str
    authorization_url: str
    token_url: str
    redirect_uri: str
    jwks_url: str
    key: str
    database_url: str


def _login_config() -> LoginConfig:
    if os.environ.get("CHURN_SAAS_LOGIN_ENABLED", "0") == "0":
        raise HTTPException(status_code=404, detail="Not found")
    if os.environ.get("CHURN_SAAS_LOGIN_ENABLED") != "1":
        raise HTTPException(status_code=503, detail="Login unavailable")
    oidc = settings()  # OIDC must independently be enabled in private staging.
    env = os.environ
    auth = env.get("CHURN_SAAS_AUTHORIZATION_URL", "")
    token = env.get("CHURN_SAAS_TOKEN_URL", "")
    redirect = env.get("CHURN_SAAS_REDIRECT_URI", "")
    client_id = env.get("CHURN_SAAS_OIDC_CLIENT_ID", "")
    key = env.get("CHURN_SAAS_SESSION_FERNET_KEY", "")
    issuer_parts = urlsplit(oidc.issuer)
    for url in (auth, token):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.hostname != issuer_parts.hostname
                or parsed.port not in (None, 443) or not parsed.path.startswith("/")
                or parsed.username or parsed.password or parsed.fragment or parsed.query):
            raise HTTPException(status_code=503, detail="Login unavailable")
    cb = urlsplit(redirect)
    if (cb.scheme != "https" or not cb.hostname or cb.port not in (None, 443)
            or cb.username or cb.password or cb.fragment or cb.query
            or cb.path != "/saas/v3/auth/callback"
            or not client_id or len(client_id) > 200 or client_id == oidc.audience):
        raise HTTPException(status_code=503, detail="Login unavailable")
    try:
        from cryptography.fernet import Fernet
        Fernet(key.encode("ascii"))
    except (ImportError, ValueError, TypeError, UnicodeError) as exc:
        raise HTTPException(status_code=503, detail="Login unavailable") from exc
    return LoginConfig(
        issuer=oidc.issuer, client_id=client_id,
        authorization_url=auth, token_url=token, redirect_uri=redirect,
        jwks_url=oidc.jwks_url, key=key, database_url=oidc.database_url,
    )


def _box(config: LoginConfig):
    from cryptography.fernet import Fernet
    return Fernet(config.key.encode("ascii"))


def _read_cookie(config: LoginConfig, value: str | None, ttl: int) -> dict:
    from cryptography.fernet import InvalidToken
    try:
        if not value or len(value) > 4096:
            raise InvalidToken()
        decoded = _box(config).decrypt(value.encode("ascii"), ttl=ttl)
        data = json.loads(decoded)
        if not isinstance(data, dict):
            raise InvalidToken()
        return data
    except (InvalidToken, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session") from exc


def _seal(config: LoginConfig, data: dict) -> str:
    return _box(config).encrypt(json.dumps(data, separators=(",", ":")).encode()).decode()


def _page(title: str, body: str) -> HTMLResponse:
    # All dynamic data in 'body' must be HTML-escaped by the caller.
    return HTMLResponse(
        "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>" + html.escape(title) + " · Churn Lab</title></head>"
        "<body><main><h1>" + html.escape(title) + "</h1>" + body
        + "</main></body></html>"
    )


@router.get("/login", response_class=HTMLResponse)
def login_page():
    _login_config()
    return _page("Acceso de pruebas", (
        "<p>Autenticación privada mediante proveedor OIDC. "
        "No utilices datos reales de clientes.</p>"
        "<p><a href='/saas/v3/auth/start'>Iniciar sesión con mi proveedor</a></p>"
    ))


@router.get("/auth/start")
def auth_start():
    cfg = _login_config()
    state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    url = cfg.authorization_url + "?" + urlencode({
        "response_type": "code", "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri, "scope": "openid",
        "code_challenge": challenge, "code_challenge_method": "S256",
        "state": state, "nonce": nonce,
    })
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(
        STATE_COOKIE, _seal(cfg, {"state": state, "nonce": nonce, "verifier": verifier}),
        max_age=TX_LIFETIME, secure=True, httponly=True, samesite="lax", path="/",
    )
    return response


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _exchange_code(cfg: LoginConfig, code: str, verifier: str) -> dict:
    data = urlencode({
        "grant_type": "authorization_code", "client_id": cfg.client_id,
        "code": code, "redirect_uri": cfg.redirect_uri, "code_verifier": verifier,
    }).encode()
    request = URLRequest(
        cfg.token_url, data=data, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    try:
        with build_opener(_NoRedirects).open(request, timeout=5) as result:
            if result.status != 200 or "application/json" not in result.headers.get("Content-Type", ""):
                raise ValueError("Unexpected token response")
            payload = result.read(16_385)
            if len(payload) > 16_384:
                raise ValueError("Oversized token response")
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid token response")
            return parsed
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail="Identity provider unavailable") from exc


def _verify_id_token(cfg: LoginConfig, token: str, nonce: str) -> tuple[Principal, int]:
    import jwt
    try:
        if not isinstance(token, str) or not 10 <= len(token) <= 8192:
            raise jwt.InvalidTokenError("Missing ID token")
        header = jwt.get_unverified_header(token)
        if (header.get("alg") != "RS256"
                or not isinstance(header.get("kid"), str)
                or not 1 <= len(header["kid"]) <= 128):
            raise jwt.InvalidTokenError("Unexpected ID token signing algorithm")
        jwk = _trusted_jwks_client(cfg.jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token, jwk.key, algorithms=["RS256"], issuer=cfg.issuer,
            audience=cfg.client_id,
            options={"require": ["iss", "aud", "iat", "exp", "sub", "nonce"]},
            leeway=10,
        )
        audience = claims["aud"]
        if isinstance(audience, list) and len(audience) > 1 and claims.get("azp") != cfg.client_id:
            raise jwt.InvalidTokenError("Invalid authorized party")
        subject = claims["sub"]
        if (not isinstance(subject, str) or not 1 <= len(subject) <= 255
                or not isinstance(claims["nonce"], str)
                or not hmac.compare_digest(claims["nonce"], nonce)):
            raise jwt.InvalidTokenError("Invalid subject or nonce")
        return Principal(issuer=cfg.issuer, subject=subject), int(claims["exp"])
    except jwt.PyJWKClientError as exc:
        raise HTTPException(status_code=503, detail="Identity provider unavailable") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid identity token") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="Identity provider unavailable") from exc


@router.get("/auth/callback")
def auth_callback(request: Request, code: str = "", state: str = ""):
    cfg = _login_config()
    tx = _read_cookie(cfg, request.cookies.get(STATE_COOKIE), TX_LIFETIME)
    if (not state or len(state) > 150 or not isinstance(tx.get("state"), str)
            or not hmac.compare_digest(tx["state"], state)
            or not isinstance(tx.get("nonce"), str)
            or not isinstance(tx.get("verifier"), str)
            or not 1 <= len(code) <= 2048):
        raise HTTPException(status_code=401, detail="Login state mismatch")
    tokens = _exchange_code(cfg, code, tx["verifier"])
    principal, id_expiry = _verify_id_token(cfg, tokens.get("id_token"), tx["nonce"])
    lifetime = min(SESSION_LIFETIME, id_expiry - int(time.time()))
    if lifetime <= 0:
        raise HTTPException(status_code=401, detail="Expired identity token")
    response = RedirectResponse("/saas/v3/organizations", status_code=303)
    response.delete_cookie(STATE_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    response.set_cookie(
        SESSION_COOKIE,
        _seal(cfg, {"iss": principal.issuer, "sub": principal.subject, "exp": int(time.time()) + lifetime}),
        secure=True, httponly=True, samesite="lax", path="/", max_age=lifetime,
    )
    return response


def get_browser_principal(request: Request) -> Principal:
    cfg = _login_config()
    data = _read_cookie(cfg, request.cookies.get(SESSION_COOKIE), SESSION_LIFETIME)
    if (data.get("iss") != cfg.issuer or not isinstance(data.get("sub"), str)
            or not 1 <= len(data["sub"]) <= 255
            or type(data.get("exp")) is not int or data["exp"] <= int(time.time())):
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return Principal(issuer=cfg.issuer, subject=data["sub"])


def _my_orgs(principal: Principal) -> list[dict[str, str]]:
    """Return only the user's active memberships, limited to 50 organizations."""
    cfg = _login_config()
    try:
        import psycopg
        with psycopg.connect(cfg.database_url, connect_timeout=4) as conn:
            with conn.transaction():
                conn.execute(
                    """SELECT set_config('app.oidc_issuer', %s, true),
                              set_config('app.oidc_subject', %s, true)""",
                    (principal.issuer, principal.subject),
                )
                memberships = conn.execute(
                    """SELECT tenant_id, role FROM saas.memberships
                       WHERE oidc_issuer = %s AND oidc_subject = %s
                         AND revoked_at IS NULL ORDER BY tenant_id LIMIT 50""",
                    (principal.issuer, principal.subject),
                ).fetchall()
                result = []
                for tenant_id, role in memberships:
                    conn.execute(
                        "SELECT set_config('app.tenant_id', %s, true)",
                        (str(tenant_id),),
                    )
                    name = conn.execute(
                        "SELECT display_name FROM saas.organizations WHERE id = %s",
                        (tenant_id,),
                    ).fetchone()
                    if name is not None:
                        result.append({
                            "tenant_id": str(tenant_id), "name": name[0], "role": role
                        })
                return result
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Tenant database unavailable") from exc


@router.get("/organizations", response_class=HTMLResponse)
def organizations(principal: Principal = Depends(get_browser_principal)):
    organizations = _my_orgs(principal)
    items = "".join(
        "<li><strong>" + html.escape(item["name"]) + "</strong>"
        " — " + html.escape(item["role"]) + "</li>"
        for item in organizations
    )
    if not items:
        items = "<li>Aún no tienes ninguna empresa asignada.</li>"
    return _page(
        "Mis empresas", "<p>Entorno privado de pruebas. No cargues datos reales.</p>"
        "<ul>" + items + "</ul>"
        "<form method='post' action='/saas/v3/logout'>"
        "<button type='submit'>Cerrar sesión</button></form>",
    )


@router.post("/logout")
def logout(request: Request, principal: Principal = Depends(get_browser_principal)):
    # Cookie Lax plus a strict Origin check for state-changing navigation.
    cfg = _login_config()
    origin = request.headers.get("origin", "")
    host = request.headers.get("host", "")
    if origin != "https://" + host:
        raise HTTPException(status_code=403, detail="Invalid origin")
    response = RedirectResponse("/saas/v3/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    response.delete_cookie(STATE_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    return response
