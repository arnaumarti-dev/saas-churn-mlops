"""Opt-in, test-only tenant authentication boundary.

This module intentionally does NOT implement customer login. For this learning
prototype, a server-provisioned SHA-256 digest of a random bearer token maps to
one tenant. Real SaaS deployment must replace this with OIDC + persistent,
auditable organization memberships before accepting actual customer data.

The feature is disabled by default and cannot be enabled without explicit
server-side configuration. No raw tokens or personal data are persisted.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from dataclasses import dataclass
from uuid import UUID

from fastapi import Header, HTTPException


_ENABLED_FLAG = "CHURN_SAAS_PROTOTYPE_ENABLED"
_TOKEN_CONFIG = "CHURN_SAAS_PROTOTYPE_TOKENS_JSON"
_HASH_PATTERN = re.compile(r"[a-f0-9]{64}\Z")
_ROLES = frozenset({"owner", "analyst", "viewer"})
_MAX_ACCOUNTS = 64


@dataclass(frozen=True)
class ServiceIdentity:
    tenant_id: UUID
    actor_id: UUID
    role: str
    expires_at: int
    digest: bytes


def _enabled() -> bool:
    flag = os.environ.get(_ENABLED_FLAG, "0")
    if flag == "0":
        return False
    if flag == "1":
        return True
    # Fail closed on typos like 'true', not silently enabling access.
    raise HTTPException(status_code=503, detail="Prototype authentication unavailable.")


def _load_identities() -> list[ServiceIdentity]:
    """Validate operator-supplied identities; invalid config fails closed."""
    raw = os.environ.get(_TOKEN_CONFIG, "")
    try:
        entries = json.loads(raw)
        if not isinstance(entries, list) or not 1 <= len(entries) <= _MAX_ACCOUNTS:
            raise ValueError("Unexpected account count")
        identities = []
        hashes: set[str] = set()
        for row in entries:
            if not isinstance(row, dict) or set(row) != {
                "token_sha256", "tenant_id", "actor_id", "role", "expires_at"
            }:
                raise ValueError("Unexpected identity fields")
            value = row["token_sha256"]
            if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
                raise ValueError("Invalid SHA256 digest")
            if value in hashes:
                raise ValueError("Duplicate token hash")
            hashes.add(value)
            tenant_id = UUID(row["tenant_id"])
            actor_id = UUID(row["actor_id"])
            if not isinstance(row["role"], str) or row["role"] not in _ROLES:
                raise ValueError("Unknown membership role")
            expiry = row["expires_at"]
            if isinstance(expiry, bool) or not isinstance(expiry, int) or expiry <= 0:
                raise ValueError("Invalid expiration")
            identities.append(ServiceIdentity(
                tenant_id=tenant_id, actor_id=actor_id, role=row["role"],
                expires_at=expiry, digest=bytes.fromhex(value)
            ))
        return identities
    except (ValueError, TypeError, KeyError) as exc:
        # Do not expose operator configuration or secrets in HTTP responses.
        raise HTTPException(
            status_code=503, detail="Prototype authentication unavailable."
        ) from exc


def get_service_identity(
    authorization: str | None = Header(default=None),
) -> ServiceIdentity:
    """Verify a short-lived, randomly generated, operator-provisioned token."""
    if not _enabled():
        raise HTTPException(status_code=404, detail="Not found")
    identities = _load_identities()
    parts = authorization.split(" ") if authorization else []
    if (len(parts) != 2 or parts[0].lower() != "bearer"
            or not re.fullmatch(r"[A-Za-z0-9_-]{43,96}", parts[1])):
        raise HTTPException(
            status_code=401, detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    candidate = hashlib.sha256(parts[1].encode("ascii")).digest()
    now = int(time.time())
    # Do not exit early based on the first matching hash.
    identity = None
    for entry in identities:
        match = hmac.compare_digest(candidate, entry.digest)
        if match and entry.expires_at > now:
            identity = entry
    if identity is None:
        raise HTTPException(
            status_code=401, detail="Invalid or expired credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return identity


def check_tenant_access(
    tenant_id: UUID, identity: ServiceIdentity, *, can_predict: bool = False
) -> None:
    """Authorization is based ONLY on the trusted token mapping.

    Unknown tenants and cross-tenant access produce the same 404.
    Caller-supplied tenant headers or body fields cannot grant membership.
    """
    if tenant_id != identity.tenant_id:
        raise HTTPException(status_code=404, detail="Not found")
    if can_predict and identity.role not in {"owner", "analyst"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions.")
