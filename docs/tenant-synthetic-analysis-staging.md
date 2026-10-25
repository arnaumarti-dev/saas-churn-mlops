# Churn Lab v2.2 — per-organization **synthetic** churn experiments

**Private staging only. No real customer or personal data.** This release
extends the confirmed local Keycloak + PostgreSQL prototype without enabling
new features in the public Render portfolio demo.

## What changed

Once signed in under a fictional organization, the private
`/saas/v3/console` contains an **Análisis de churn · ejemplos** panel.

- **Owner / Analyst:** can run one analysis of **three fixed synthetic**
  profiles included with the application. The server calls the same
  `predict_many`/trained XGBoost or logistic pipeline as the public demo.
  The response displays total profiles, predicted high/low-risk count and
  mean churn probability.
- **Viewer:** can view aggregate run history for organizations where they
  have active membership, but cannot initiate scoring.
- **Revoked users/non-members:** cannot access historical results or run
  an analysis for that tenant.
- **History:** reuses the existing `saas.sandbox_runs` table from
  migration `0001`; **only** UUID, tenant ID, synthetic actor subject,
  total profile count, high-risk count and creation timestamp are stored.
  No CSV, customer rows, emails, probability arrays or raw input payloads
  are stored. The UI exposes only timestamps and aggregate counts.

APIs:
- `GET /saas/v3/organizations/{tenant_id}/runs`
- `POST /saas/v3/organizations/{tenant_id}/runs/sample` with `{}`
  and a same-origin JSON request. No extra payload attributes are accepted.

Both routes require browser OIDC, the independent
`CHURN_SAAS_MANAGEMENT_ENABLED=1` staging feature flag and PostgreSQL
membership checks. Transaction-local tenant/identity settings enforce
Postgres RLS. Calls use parameterized SQL, strict origin checks for POST,
body size limits and the current in-process rate limiter.

## Real ML artifact in the local image

`dev/private-staging/Dockerfile` now trains from the same **public IBM
Telco sample dataset** at build time, using the existing checked-in
training scripts. It copies only `models/best_model.joblib` into the
runtime. The public Render demo and its Dockerfile are unchanged.
This build needs network access during training to fetch the public
dataset and enough local disk/RAM to train the models.

After **PR is merged** on GitHub, from Windows PowerShell in your own
cloned `saas-churn-mlops` repository directory:

```powershell
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
docker compose --env-file .env.staging.local -f compose.private-staging.yml ps
```

Then refresh `https://app.localhost/saas/v3/console`, log in again if
the short-lived cookie has expired, select the existing **Acme Demo**
company and choose **Analizar 3 perfiles de ejemplo**. This creates one
new historical aggregate row. Switch to a viewer session to confirm it
can read the run but cannot press the scoring button. Promoted analysts
can score; after revocation they cannot access the history.

**Never run `docker compose down --volumes` as part of this update**:
that permanently deletes the local PostgreSQL data, Keycloak test
users and local Caddy certificate authority. The normal `up -d --build
app` replaces only the app container and preserves those volumes.

## Security/operational limitations

- **Do not enter real customer data.** The sample endpoint accepts an
  empty JSON object only; the built-in fictional profiles are fixed.
- This is experimental staging, not production multi-tenancy.
- The short-lived session has no central revocation; authorization
  is re-evaluated from PostgreSQL at every API request. The existing
  in-process request limiter is not distributed.
- The build trains from an external public dataset; reproducible
  production artifacts would require checksum/provenance pinning and
  controlled model registry.
- No raw profile history, analyst dashboard, member-friendly identities,
  auditing, payments, or lifecycle monitoring are offered yet.
- Postgres integration CI stubs the inference math to stress real RLS
  and roles; existing independent model training/prediction CI exercises
  real inference. The full new **local image training path** has not
  been verified on a user machine until they rebuild/test it.
