# SaaS foundation — identity, tenant isolation and release gates

**Status: prototype only, disabled by default. Not a production customer
login system. Do not upload real customer profiles.**

This increment adds a small, explicitly gated `/saas/v1/` namespace, with
tenant-scoped in-memory batch prediction and service identities meant for
local development/testing. The existing public demo stays unchanged.

## Current experimental endpoints

| Route | Required role | Behavior |
|---|---|---|
| `GET /saas/v1/tenants/{tenant_uuid}/me` | viewer, analyst, owner | Returns actor/tenant/role and `mode: prototype-only` |
| `POST /saas/v1/tenants/{tenant_uuid}/predict/batch` | analyst, owner | Runs the existing model on up to 200 profiles; no storage |

Both return 404 while `CHURN_SAAS_PROTOTYPE_ENABLED` is absent or `0`.
No frontend uses these routes and Render does not enable them. This feature
does not add user registration, passwords, invitations, account sessions,
billing or business data persistence.

When explicitly enabled **in a private development environment**, identities
are provisioned on the server through `CHURN_SAAS_PROTOTYPE_TOKENS_JSON`,
a JSON list with up to 64 entries. Each entry has exactly:

```json
{
  "token_sha256": "<64-char lowercase sha256 of a random 32-byte bearer token>",
  "tenant_id": "<uuid>",
  "actor_id": "<uuid>",
  "role": "analyst",
  "expires_at": 1900000000
}
```

`expires_at` is a Unix epoch timestamp in seconds. Set it to a short future
time for private tests. Generate a token with
`secrets.token_urlsafe(32)` **outside version control**, record only its
SHA-256 digest in private environment configuration, and supply the raw token
to the **test-only** client via `Authorization: Bearer <token>`.
Never commit tokens or configure them in a public demo.

A credential belongs to exactly one tenant. The URL tenant must match the
trusted server-side mapping; adding an `X-Tenant-ID` header or changing a
request body cannot grant access. Missing or invalid credentials return 401;
cross-tenant requests return 404 to avoid confirming another company exists;
viewer role cannot score and returns 403. Expired tokens fail. Any malformed,
missing or duplicated operator token configuration fails closed with 503.
Tenant responses and errors use `Cache-Control: no-store`. Tenant batch
requests also reuse the public JSON content-type restriction, 256 KiB body
limit and best-effort single-process rate limiter.

## Architecture proposed for *real* SaaS

1. **Identity**: replace test-only shared service tokens with a hosted
   OIDC provider using Authorization Code + PKCE for browser login; verify
   signed ID/access tokens with issuer, audience, expiration, signature,
   key rotation and session protections. Enforce MFA for owners/admins.
2. **Tenant membership**: store organization membership, roles, pending
   invitations and revocations in a *server-controlled* relational schema.
   Always derive tenant identity from an authenticated membership.
   Never authorize based solely on a request-supplied company ID.
3. **Persistence**: PostgreSQL with every tenant-owned row carrying a
   non-null `tenant_id`; scoped repositories plus **RLS** policies,
   with automatic tests ensuring tenant A cannot read/update/delete rows
   belonging to tenant B. Add unique constraints combining tenant IDs and
   resource IDs where applicable.
4. **Privacy/data**: document controller/processor roles, legal basis,
   retention schedules, deletion/export procedures, encryption at rest
   and in transit, sensitive-data access policies, incident response,
   DPAs and EU GDPR obligations. Run a model quality and bias review
   before accepting or using any real profiles.
5. **Operational security**: CSRF protection for cookie sessions,
   shared/distributed quotas, audit events without raw inputs or tokens,
   secret rotation, key management, backups, tenant-aware monitoring,
   migrations, versioned APIs and vulnerability scans.
6. **Production gates**: security threat model and independent review,
   staged testing, incident/rollback plan, workload budgets, monitoring,
   documented tenant offboarding and test evidence for isolation.

### Explicit exclusions and cautions

This learning increment **does not satisfy** any production gate above.
The SHA-256 mapping is a convenient ephemeral test-only bearer-key system,
**not end-user authentication**; it has no password recovery, browser
session management, audit trail, centralized revocation or key rotation
workflow. A service-token map in an environment variable is not a
multi-tenant membership database. The ML model is a teaching prototype,
not a validated business decision system. No production deployment,
feature enablement or handling of actual customer data is authorized here.

Run the isolated tests locally with:

```bash
PYTHONPATH=src pytest -q tests/test_tenant_auth.py
```

The main CI tests the tenant security boundary while keeping the
feature disabled by default in Docker/Render.
