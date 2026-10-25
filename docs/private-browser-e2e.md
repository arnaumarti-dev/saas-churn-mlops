# Churn Lab v2.0 candidate: private Chromium integration QA

The **private staging console** has a real browser test that uses an ephemeral
PostgreSQL 16 service and a local HTTPS FastAPI process on the GitHub Actions
runner. No customer account, OIDC provider, external database, Render deployment,
third-party API, subscription, email or paid service is configured by this PR.

## What is actually verified

After running migrations `0001`, `0002` and `0003` against disposable
PostgreSQL, the `Optional SaaS security integration` CI job:

1. Enables `CHURN_SAAS_LOGIN_ENABLED=1`,
   `CHURN_SAAS_OIDC_ENABLED=1` and `CHURN_SAAS_MANAGEMENT_ENABLED=1`
   **only** for the CI subprocess, not for the deployed public demo.
2. Generates a fresh Fernet key and a self-signed TLS certificate in the
   ephemeral runner (never committed or uploaded). It uses HTTPS localhost
   with port 443 to retain the production Secure-cookie and strict callback
   URL requirements. Chromium ignores this **test-only** certificate.
3. Creates synthetic temporary browser sessions locally with the same
   signed/encrypted cookie format the real login would issue after OIDC.
   These test cookies *simulate* an authenticated principal; **they do not
   authenticate against a real IdP**.
4. Verifies anonymous users receive 401 on the console, JavaScript, CSS and
   organization JSON endpoints.
5. In desktop and mobile Chromium, creates a synthetic organization through
   the actual HTML/JavaScript, verifies owner controls and the last-owner
   safeguard, generates a one-time invitation, and redeems it as a distinct
   synthetic browser user.
6. Verifies that a viewer cannot list other members, while an owner can
   promote and revoke the viewer, and the viewer loses organization access.
7. Stores only synthetic screenshots **before generating the invitation**
   under `test-results/private-staging-qa/`, with seven-day retention.
   Invitation codes, Fernet keys and database connection strings are never
   part of screenshots.

The existing unit/integration suite already covers signature/nonce validation
using locally generated RSA test keys and an OIDC token exchange mock. The
new Chromium test checks the **separate browser and database path**, not the
real identity-provider redirect/token flow.

## Execution

GitHub Actions runs this test only in the `Optional SaaS security integration`
workflow, on eligible PRs involving staging backend files. For local manual
testing, provision a *throwaway* PostgreSQL 16 database with a nonprivileged
app role, apply migrations 0001-0003 with a separate migration owner, and set
`CHURN_TEST_DB_APP_URL` to the throwaway role DSN. Install the
`requirements.serve.txt`, `requirements.saas.txt`, `pytest`,
`httpx` and Python Playwright packages, plus Playwright Chromium. This test
requires binding HTTPS port 443; CI permits that only on its isolated
disposable runner. Do not relax the secure-cookie restrictions or run the
CI-only helper on a public server.

## Remaining production blockers

A real OIDC provider and end-to-end Authorization Code + PKCE login, private
HTTPS hosting, privately provisioned PostgreSQL, server-side revocable
sessions, secure invite/account recovery, user consent/MFA where relevant,
production CSRF hardening, audit logs, distributed rate limiting and a
security/privacy review remain **unimplemented and unverified**. Never use
real customer data or enable private routes on public Render.
