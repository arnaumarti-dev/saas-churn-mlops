"""STAGING-ONLY tenant-management JSON API. Disabled by default.

Only an authenticated short-lived browser session may call these endpoints.
Mutations check the trusted IdP staging origin. PostgreSQL SECURITY DEFINER
functions make all owner authorization and last-owner checks atomic, while
the app database role retains no table write permission.
"""
from __future__ import annotations

import hashlib
import os
import secrets
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from churn_mlops.api.oidc_postgres import Principal
from churn_mlops.api.staging_login import _login_config, get_browser_principal


router = APIRouter(prefix="/saas/v3")


def _v18_principal(request: Request) -> Principal:
    if os.getenv("CHURN_SAAS_MANAGEMENT_ENABLED", "0") == "0":
        raise HTTPException(status_code=404, detail="Not found")
    if os.getenv("CHURN_SAAS_MANAGEMENT_ENABLED") != "1":
        raise HTTPException(status_code=503, detail="Management unavailable")
    return get_browser_principal(request)


def _check_origin(request: Request, *, json_body: bool = True) -> None:
    from urllib.parse import urlsplit
    cfg = _login_config()
    callback = urlsplit(cfg.redirect_uri)
    trusted = callback.scheme + "://" + callback.netloc
    if request.headers.get("origin") != trusted:
        raise HTTPException(status_code=403, detail="Invalid request origin")
    if json_body and request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="JSON required")


def _sql(principal: Principal, statement: str, args: tuple):
    cfg = _login_config()
    try:
        import psycopg
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    try:
        with psycopg.connect(cfg.database_url, connect_timeout=4) as conn:
            with conn.transaction():
                conn.execute(
                    """SELECT set_config('app.oidc_issuer', %s, true),
                              set_config('app.oidc_subject', %s, true)""",
                    (principal.issuer, principal.subject),
                )
                # All SQL statements are fixed developer-owned literals.
                # User inputs enter psycopg placeholders only.
                return conn.execute(statement, args).fetchall()
    except psycopg.errors.RaiseException as exc:
        raise HTTPException(status_code=403, detail="Operation not allowed") from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc


class CreateOrganization(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class CreateInvite(BaseModel):
    role: str = Field(pattern="^(analyst|viewer)$")


class RedeemInvite(BaseModel):
    code: str = Field(pattern="^[A-Za-z0-9_-]{43}$", min_length=43, max_length=43)


class MemberRole(BaseModel):
    role: str = Field(pattern="^(owner|analyst|viewer)$")


@router.post("/organizations", status_code=201)
def create_organization(
    payload: CreateOrganization, request: Request,
    principal: Principal = Depends(_v18_principal),
):
    _check_origin(request)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Organization name required")
    tenant_id = _sql(principal, "SELECT saas.create_organization(%s)", (name,))[0][0]
    return {"tenant_id": str(tenant_id), "name": name, "role": "owner"}


@router.post("/organizations/{tenant_id}/invitations", status_code=201)
def create_invitation(
    tenant_id: UUID, payload: CreateInvite, request: Request,
    principal: Principal = Depends(_v18_principal),
):
    _check_origin(request)
    code = secrets.token_urlsafe(32)
    digest = hashlib.sha256(code.encode("ascii")).hexdigest()
    _sql(
        principal,
        "SELECT saas.create_invitation(%s, %s, %s)",
        (tenant_id, payload.role, digest),
    )
    # Return the secret once; PostgreSQL stores SHA-256 only. No email sent.
    return {"code": code, "role": payload.role, "expires_in_seconds": 86400}


@router.post("/invitations/accept")
def accept_invitation(
    payload: RedeemInvite, request: Request,
    principal: Principal = Depends(_v18_principal),
):
    _check_origin(request)
    digest = hashlib.sha256(payload.code.encode("ascii")).hexdigest()
    tenant_id = _sql(principal, "SELECT saas.accept_invitation(%s)", (digest,))[0][0]
    return {"tenant_id": str(tenant_id), "joined": True}


@router.get("/organizations/{tenant_id}/members")
def members(
    tenant_id: UUID, principal: Principal = Depends(_v18_principal),
):
    rows = _sql(
        principal,
        "SELECT member_subject, member_role FROM saas.list_organization_members(%s)",
        (tenant_id,),
    )
    return {"tenant_id": str(tenant_id),
            "members": [{"subject": subject, "role": role} for subject, role in rows]}


@router.patch("/organizations/{tenant_id}/members/{subject}")
def change_role(
    tenant_id: UUID, subject: str, payload: MemberRole, request: Request,
    principal: Principal = Depends(_v18_principal),
):
    _check_origin(request)
    if not 1 <= len(subject) <= 255:
        raise HTTPException(status_code=422, detail="Invalid member")
    _sql(
        principal,
        "SELECT saas.change_member_role(%s, %s, %s)",
        (tenant_id, subject, payload.role),
    )
    return {"updated": True}


@router.delete("/organizations/{tenant_id}/members/{subject}")
def revoke_member(
    tenant_id: UUID, subject: str, request: Request,
    principal: Principal = Depends(_v18_principal),
):
    _check_origin(request, json_body=False)
    if not 1 <= len(subject) <= 255:
        raise HTTPException(status_code=422, detail="Invalid member")
    _sql(
        principal,
        "SELECT saas.revoke_organization_member(%s, %s)",
        (tenant_id, subject),
    )
    return {"revoked": True}
