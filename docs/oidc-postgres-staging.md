# Churn Lab — staged OIDC and PostgreSQL integration

**NOT A PUBLIC LOGIN, NOT READY FOR CUSTOMER DATA.** This feature is an
optional API-only integration behind `CHURN_SAAS_OIDC_ENABLED=1`. It is disabled
on Render, not shown in the web UI, and has no account signup or billing.

## What was implemented

- A resource-server verifier of **signed OAuth/OIDC access tokens**. The
  browser login process itself must be hosted by a managed identity provider,
  using Authorization Code + PKCE, HTTPS, state/nonce where appropriate,
  short-lived access tokens and refresh-token lifecycle controls.
- Only **RS256** access tokens signed by the operator-configured identity
  provider are accepted. Claims `iss`, `aud`, `exp`, `iat` and `sub` are
  required and validated. The JWKS endpoint is pinned to HTTPS on the same
  hostname as the configured issuer. No token-supplied JKU/X5U URL is fetched.
  The key material is verified as an RSA public key, rather than passing a
  JWK object that could affect algorithm policy. Role/tenant claims in a JWT
  are **never** treated as authorization.
- PostgreSQL memberships are authoritative: `tenant_id` from the route,
  verified issuer and verified subject must all match a non-revoked
  membership row. App queries set `app.tenant_id`, `app.oidc_issuer`,
  `app.oidc_subject` for **one transaction only**. Their absence denies
  access under RLS; cross-tenant membership lookup returns 404.
- New API routes:
  - `GET /saas/v2/tenants/{tenant_uuid}/me`: returns tenant and member role.
  - `POST /saas/v2/tenants/{tenant_uuid}/predict/batch`: only owner/analyst.
  They reuse the existing in-memory synthetic demo model, strict JSON,
  256 KiB request limit, rate limiting and no-store headers.
- A SQL migration creates `saas.organizations`, `saas.memberships`, and
  `saas.sandbox_runs` **metadata only**. Every tenant-owned table has
  `ENABLE/FORCE ROW LEVEL SECURITY`, explicit read and insert policies,
  tenant indexes and constrained app permissions. The app's normal login
  must be neither a superuser, table owner, nor a `BYPASSRLS` role.

## Private staging setup (not configured in production)

1. Choose a trusted OIDC provider and create an **API audience/resource**
   configured to issue signed RS256 **access tokens**, not ID tokens.
   Verify discovery metadata, key rotation and token shape manually first.
2. Provision a *private* PostgreSQL 16+ database and a **separate admin
   migration role**. Create `churn_saas_app` with minimal login privileges,
   `NOSUPERUSER NOBYPASSRLS`. It must not own tables or run migrations.
3. Apply `db/migrations/0001_saas_tenants_rls.sql` as migration owner.
   Provision organizations/memberships through a separate trusted admin
   process; do not add public signup endpoints until invitations and role
   changes have an audited authorization flow.
4. Install `requirements.saas.txt` in the **private staging** runtime. It is
   intentionally not included in the public Docker image.
5. Set the following **private** environment configuration values; never
   commit client secrets, tokens or database passwords:
   - `CHURN_SAAS_OIDC_ENABLED=1`
   - `CHURN_SAAS_OIDC_ISSUER=https://idp.example.com/issuer`
   - `CHURN_SAAS_OIDC_AUDIENCE=your-api-audience`
   - `CHURN_SAAS_OIDC_JWKS_URL=https://idp.example.com/.well-known/jwks.json`
   - `CHURN_SAAS_DATABASE_URL=postgresql://churn_saas_app:<secret>@host/dbname?sslmode=require`
6. Check valid, tampered, expired and wrong-audience access tokens; expired/
   revoked membership, role downgrade and cross-tenant access. Exercise
   actual PostgreSQL RLS as the least-privileged app DB role.

**Never enable this flag in the public Render service.** Even after these
unit/integration tests pass, production readiness still requires independent
security review, an end-user session/token handling design, token type
separation based on the chosen provider, key/secret rotation, strong MFA,
invitation and membership revocation flows, rate-limit infrastructure,
CSRF defenses for cookie sessions, backups, audit logging and GDPR/DPA
review. No real customer profiles should be processed until those gates pass.

## Test commands

```bash
pip install -r requirements.txt -r requirements.saas.txt
PYTHONPATH=src pytest -q tests/test_oidc_postgres.py
```

The GitHub Actions `Optional SaaS security integration` workflow uses a
fresh temporary PostgreSQL service and random synthetic tenant IDs, then
discards the database. Normal Python, Docker, Render and public demo
dependencies are unchanged.
