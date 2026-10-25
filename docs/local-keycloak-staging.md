# Churn Lab v2.1 — local-only Keycloak and PostgreSQL staging

**An isolated disposable developer environment, NOT a cloud deployment or
a production authentication system.** No real customers or personal data.
This setup deliberately uses Keycloak's **development mode**, which must
NEVER be used on public infrastructure. It does not create any external
accounts or deploy to Render.

This is a concrete local provider option for validating the REAL browser
OIDC Authorization Code + PKCE redirect/token/JWKS flow, which previous
synthetic-cookie tests do not cover. The updated backend recognizes a
Keycloak realm's exact, operator-pinned HTTPS JWKS URL while continuing
to reject foreign hosts and wrong realms.

## Requirements

- Docker Desktop / Docker Compose v2, working locally on Windows, macOS
  or Linux; at least 3–4 GiB free for containers and model dependencies.
- A private local machine where TCP port **443** is free. The ONLY
  published container port is bound to `127.0.0.1:443`.
- Python 3 available locally to generate high-entropy random values.
- An explicit decision to trust a **generated local Caddy root CA** on
  *this test machine only*. Trust **only the exported root certificate**,
  NEVER the private CA key. Remove the root after testing.

Local hostnames `idp.localhost` and `app.localhost` should resolve to
127.0.0.1. If your OS/browser does not resolve them, add these exact entries
to your operating-system hosts file after verifying local network policy:

```text
127.0.0.1 idp.localhost app.localhost
```

### 1. Set up credentials locally

From the repository root:

```powershell
Copy-Item .env.staging.example .env.staging.local
python -c "import secrets; print(secrets.token_hex(24))"
python -c "import secrets,base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

Edit `.env.staging.local` manually. Generate a **different 48-character
hexadecimal password** for each of `LOCAL_DB_ADMIN_PASSWORD`,
`CHURN_DB_APP_PASSWORD`, `KEYCLOAK_DB_PASSWORD`,
`KEYCLOAK_ADMIN_PASSWORD`. Use the second command's 44-character
Fernet key for `CHURN_SAAS_SESSION_FERNET_KEY`. Set a local admin username.
Only lowercase/uppercase hex passwords are supported by the Compose
PostgreSQL DSN string, because arbitrary reserved URL characters must
otherwise be percent-encoded.

Do **not** send these values through ChatGPT, GitHub Issues, PRs or commits.
Do not use real passwords or reuse them elsewhere. The file and exported
`local-trust/root.crt` are ignored by Git.

### 2. Create disposable local DB and Keycloak, then TLS reverse proxy

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d db keycloak proxy
```

On first startup, `db` creates *two separate local databases*:
`keycloak_staging` (for Keycloak) and `churn_saas_staging` (for Churn
Lab). PostgreSQL migration scripts 0001–0003 run with the **migration
owner**; the Churn app's `churn_saas_app` role is non-superuser,
`NOBYPASSRLS` and does not own these tables.

The `proxy` service terminates HTTPS for both hostnames and forwards
to Keycloak and the API. Neither PostgreSQL nor the Keycloak HTTP port
is exposed to the host. The Caddy root CA private key remains ONLY in
a private Docker volume.

### 3. Export the Caddy *public* root certificate

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml cp proxy:/data/caddy/pki/authorities/local/root.crt ./dev/private-staging/local-trust/root.crt
```

If the file does not exist immediately, inspect `docker compose logs proxy`
and try again after the proxy has initialized.

Import **this certificate only** into your browser/OS's **current-user
Trusted Root Certification Authorities** trust store. On Windows, use
`certmgr.msc` → Current User → Trusted Root Certification Authorities →
Certificates → Import. Inspect the certificate and remove it after
finishing the local experiment. Do not install unknown CA files.

Caddy's root certificate is read-only mounted into the app as
`SSL_CERT_FILE=/run/private-staging-ca/root.crt`. This allows the Python
OIDC backchannel to validate TLS to Keycloak without disabling
verification or sharing Caddy's private CA key.

### 4. Launch local Churn Lab

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
docker compose --env-file .env.staging.local -f compose.private-staging.yml ps
```

Open the exact endpoints in your browser:

- Keycloak local admin: **https://idp.localhost/admin/**
- Private login: **https://app.localhost/saas/v3/login**
- Private management console after login:
  **https://app.localhost/saas/v3/console**

Keycloak boots a realm named `churn-staging` with a **public**
`churn-staging-browser` client, Authorization Code flow, mandatory
PKCE S256, one exact redirect URI and **registration disabled**.
Sign in to the Keycloak Admin Console using the local admin credentials
from your `.env.staging.local`. Select realm `churn-staging`, create two
**fictional** test users under Users, give each a throwaway password under
Credentials, and test each in separate browser sessions. Do not share
real names or use actual email addresses.

The private console lets an owner create a synthetic organization and
issue a viewer/analyst invite to the other account. The second account
redeems the one-time code and cannot manage memberships; the owner can
promote or revoke. PostgreSQL RLS is still enforced on actual membership
reads/writes via the existing API.

The staging API may log "Model artifact not found": model training is not
included in this lightweight **identity-only** image, and prediction
endpoints return 503. That does NOT affect testing login or team management.

### 5. Verify and troubleshoot

Verify OIDC discovery at:
`https://idp.localhost/realms/churn-staging/.well-known/openid-configuration`.
Its **issuer must be exactly**
`https://idp.localhost/realms/churn-staging`, and its
`jwks_uri` must equal
`https://idp.localhost/realms/churn-staging/protocol/openid-connect/certs`.
The application must use that pinned endpoint and verify signed RS256 ID
tokens.

Troubleshooting:
- **TLS warning**: root CA not trusted in the browser/OS. Don't use
  `verify=False` or disable HTTPS; import the correct local CA or stop.
- **App 503 at login**: check the `app` logs, local CA file, and environment
  configuration; the API must be able to reach Keycloak through `idp.localhost`.
- **Connection refused on :443**: another local process has port 443.
- **Keycloak unavailable**: `docker compose ... logs db keycloak proxy`.
  Keycloak can take longer to start than the API.
- **Changed DB credentials but login fails**: PostgreSQL initializes roles
  only on **first volume creation**. Do not blindly reuse a volume with
  mismatched passwords.

### 6. Shut down and clean up

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml down
```

For a full clean reset of **this local staging stack** (destroys only
the test databases, created users and locally issued certificates):

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml down --volumes
```

Afterward, REMOVE the previously imported **Caddy Local Authority**
certificate from the current user's trust store; delete the generated
`dev/private-staging/local-trust/root.crt` and your local
`.env.staging.local` if no longer needed. Otherwise a stale trusted CA
could remain without a managed private key.

## CI and limitations

GitHub CI runs schema, pinning, realm-client and Compose **static**
validation. It does **not** start the full local Keycloak provider, create
users, confirm interactive login in Chrome or verify host certificate trust
on a real Windows machine. The user-run browser verification above is still
required; do not advertise live provider integration as tested until done.

This experiment lacks production key rotation, centrally revocable user
sessions, MFA, audited invitations, rate-limiting shared across replicas,
provider high-availability, GDPR process, recoverability and external
security review. Never copy `start-dev`, local root CA or plaintext local
environment passwords into the public Render service.

Keycloak official references:
- https://www.keycloak.org/server/containers
- https://www.keycloak.org/server/importExport
- https://www.keycloak.org/server/hostname

Caddy reference:
- https://caddyserver.com/docs/caddyfile/directives/tls


## Extensión v2.3: alias ficticios de equipo

La versión 2.3 necesita `db/migrations/0004_member_aliases.sql` para añadir alias personales por empresa. En bases de datos persistentes los scripts `initdb` no se aplican otra vez: sigue el procedimiento de migración **no destructiva** de [`private-tenant-overview-v23.md`](private-tenant-overview-v23.md) antes de reiniciar la imagen nueva. No borres los volúmenes de PostgreSQL, Keycloak ni Caddy.
