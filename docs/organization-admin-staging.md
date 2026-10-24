# Churn Lab v1.8 — private staging organization administration

**Disabled by default. No actual users, IdP, email delivery, billable services
or customer data have been provisioned. Do not enable this on public Render.**

This is the next *backend* foundation after v1.7 browser OIDC login. A
configured private OIDC staging deployment can create example organizations,
share one-time invitation codes and manage active organization members.
The existing public demo continues to serve synthetic churn predictions.

## Gate and database

Configure v1.7 per [OIDC staging guide](oidc-browser-login-staging.md)
and independently set `CHURN_SAAS_MANAGEMENT_ENABLED=1` **only** on a
private HTTPS staging service. Run
`db/migrations/0003_organization_management.sql` as the trusted migration
owner after migrations 0001 and 0002.

Migration 0003 requires PostgreSQL >=13; it creates no accounts or
customer data. The migration role must be able to execute the
`SECURITY DEFINER` functions on FORCE RLS tables and must be separate from
the non-superuser `churn_saas_app` runtime role. In private integration CI,
the migration is executed by ephemeral database superuser `postgres`.
The runtime role retains SELECT-only access to pre-existing organization
and self-membership tables; it receives **only EXECUTE permissions** on
named privileged routines, not direct table writes or the invitation table.
Every privileged procedure validates the operator-supplied, verified OIDC
`issuer`/`sub` bound by the server's transaction and checks DB owner
membership. Privileged routines never read JWT roles.

**Security caveat:** the app role can set PostgreSQL custom settings via
SQL; the backend must be the sole holder of its credentials, use exclusively
parameterized SQL and never offer arbitrary SQL execution. This is an
experimental boundary and needs a formal security review before production.

## Authenticated JSON endpoints

All endpoints are under `/saas/v3`, require a valid fifteen-minute
encrypted browser login cookie, a trusted same-origin `Origin` header
on mutations, and strict bounded JSON request bodies. The feature
flag returns HTTP 404 if unset.

| Method | Path | Required role | Effect |
|---|---|---|---|
| POST | `/organizations` | Authenticated browser user | Creates an example organization and the caller as initial owner |
| GET | `/organizations/{uuid}/members` | owner | Lists active member OIDC subjects and their roles |
| POST | `/organizations/{uuid}/invitations` | owner | Generates a 24-hour, one-time code for analyst/viewer |
| POST | `/invitations/accept` | Authenticated browser user | Accepts a previously generated unused code |
| PATCH | `/organizations/{uuid}/members/{subject}` | owner | Assigns viewer/analyst/owner |
| DELETE | `/organizations/{uuid}/members/{subject}` | owner | Revokes a member |

Raw invitation codes are returned **once** to the creating owner; only
SHA-256 digests are stored in Postgres. There is **no email delivery**
and codes are not bound to an email: anyone signed into the **same configured
OIDC issuer** who possesses a valid code may redeem it. Share codes only
with authorized synthetic test accounts. Do not paste codes into logs,
analytics or support tickets. Each code lasts 24 hours and can be redeemed
once. Owners can generate up to 10 outstanding codes per organization.
Organization names are limited to 120 characters. Owner membership can be
transferred, but the last active owner cannot demote/revoke themself.
There is no real identity administration yet: no password, email verification,
registration, payment, invite-email service or audit trail.

## Operator tests

`Optional SaaS security integration` runs on ephemeral PostgreSQL 16 using
synthetic OIDC subjects. It covers registration, invite token hashing,
replay/expiry, role elevation/denial, last-owner guard, cross-tenant access,
revocation, direct table grant denial, Origin and session checks.
The public Docker HTTP smoke also checks disabled v1.8 routes respond 404.

**Before production:** real OIDC provider end-to-end validation, account
onboarding, per-organization quotas, anti-abuse edge controls, secure
server-side revocable sessions, invite identity binding, delivery and
revocation, CSRF review for browser clients, migration-owner privilege
hardening, user-friendly management UI, token rotation, audit logs and
independent security/privacy review. Avoid real customer data.
