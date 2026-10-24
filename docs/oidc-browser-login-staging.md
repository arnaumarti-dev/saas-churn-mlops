# Browser SSO in private staging (v1.7 candidate)

**Explicitly disabled in the public demo. Not suitable for real customer data.**

The previous version implemented signed OIDC API access-token verification and
PostgreSQL tenant membership. This increment adds a *separate staging-only*
browser login using Authorization Code + PKCE and a private "Mis empresas"
page that lists only the organizations the authenticated user belongs to.

It is not an account registration system and does not create a company,
invite collaborators, take payments, collect profiles or set up an identity
provider. No database or hosting resources are provisioned.

## What is available after private setup

- `GET /saas/v3/login`: Spanish introduction with a sign-in link.
- `GET /saas/v3/auth/start`: creates random `state`, `nonce`, PKCE
  verifier/S256 challenge. Transaction data are encrypted in a five-minute,
  `Secure`, `HttpOnly`, `SameSite=Lax`, `__Host-` cookie. Redirects only
  to the operator-pinned HTTPS IdP authorization URL.
- `GET /saas/v3/auth/callback`: checks the state, exchanges a single-use
  code using PKCE at the pinned HTTPS IdP token endpoint (redirects
  forbidden), checks a signed RS256 OIDC **ID token** against issuer, client
  audience, exp, iat, sub and nonce, then establishes a sealed
  fifteen-minute browser cookie. The access token is **not** saved.
- `GET /saas/v3/organizations`: decrypts the session, looks up its verified
  issuer and subject in PostgreSQL and lists up to 50 **active memberships**.
  Organization names are HTML-escaped and never taken from JWT roles.
- `POST /saas/v3/logout`: verifies same-origin POST and deletes the browser
  session cookie. There is no cross-provider logout or server-side
  revocation in this proof of concept.

A migration `db/migrations/0002_member_org_selector.sql` allows users to
read **only their own membership rows** across multiple organizations.
Organization detail access still requires its own transaction-local
tenant ID under PostgreSQL RLS, and no membership/invitation writes are
exposed. Roles, membership revocation and company names are always read
from the database on each organizations request.

## Required staging-only configuration

The existing `CHURN_SAAS_OIDC_*` values, JWKS and private PostgreSQL DB from
`docs/oidc-postgres-staging.md` must be configured. Add:

```dotenv
CHURN_SAAS_LOGIN_ENABLED=1
CHURN_SAAS_OIDC_ENABLED=1
CHURN_SAAS_OIDC_CLIENT_ID=<public browser OIDC client id>
CHURN_SAAS_AUTHORIZATION_URL=https://idp.example.org/authorize
CHURN_SAAS_TOKEN_URL=https://idp.example.org/oauth/token
CHURN_SAAS_REDIRECT_URI=https://private-staging.example.org/saas/v3/auth/callback
CHURN_SAAS_SESSION_FERNET_KEY=<random 32-byte Fernet key>
```

**Use real issuer-host URLs provided by the chosen IdP, not the example
values.** Authorization/token/JWKS HTTPS endpoints must share the trusted
issuer hostname, and the registered callback must match exactly. Register a
public Authorization Code + PKCE client that returns a signed RS256 ID token
with the requested nonce. The browser client ID must differ from the API
access-token audience. Some providers require a confidential client secret;
this staging implementation intentionally does not support that workflow.

Generate the session key **locally**, never in chat or Git:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Set secrets exclusively using the protected environment settings of the
private staging host. Run migrations with an admin-only role; the app role
must be `NOSUPERUSER NOBYPASSRLS`, not the table owner. Optional packages
remain outside `requirements.serve.txt` and the public Docker image.
Set the stage on an isolated private HTTPS hostname and restrict access.

## Tests and production blockers

The `Optional SaaS security integration` CI test uses a locally generated
RSA key, a mocked IdP code exchange and an ephemeral real PostgreSQL 16
instance. It checks PKCE, state/nonce tampering, ID token signature,
encrypted secure cookies, logout CSRF, HTML escaping, cross-org reads,
revoked membership and no database context leakage. The normal Docker smoke
test confirms that all browser staging endpoints respond 404 when disabled.

**This is not a live provider test and must not be advertised as a finished
public login.** Before production: choose/configure an actual OIDC IdP,
verify end-to-end browser login with that IdP, server-side revocable sessions,
secure key rotation, MFA, approved OAuth token storage policies, invitations,
role changes, organization lifecycle, CSRF protections for future mutations,
distributed abuse limiting, recovery, audit trails, privacy/GDPR review,
independent security review and operational monitoring. Never use real
customer data until those gates are met.
