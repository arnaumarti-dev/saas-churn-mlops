-- Applied ONLY to a private PostgreSQL staging database, after provisioning
-- a login role churn_saas_app with NOSUPERUSER NOBYPASSRLS and no CREATEDB.
-- Run as a separate migration owner, never using application credentials.
-- No real customer profiles or PII are stored by this migration.
BEGIN;

CREATE SCHEMA IF NOT EXISTS saas;

CREATE TABLE IF NOT EXISTS saas.organizations (
    id uuid PRIMARY KEY,
    display_name text NOT NULL CHECK (length(display_name) BETWEEN 1 AND 120),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS saas.memberships (
    tenant_id uuid NOT NULL REFERENCES saas.organizations(id) ON DELETE CASCADE,
    oidc_issuer text NOT NULL CHECK (length(oidc_issuer) BETWEEN 8 AND 500),
    oidc_subject text NOT NULL CHECK (length(oidc_subject) BETWEEN 1 AND 255),
    role text NOT NULL CHECK (role IN ('owner', 'analyst', 'viewer')),
    revoked_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, oidc_issuer, oidc_subject)
);
CREATE INDEX IF NOT EXISTS saas_membership_issuer_subject
    ON saas.memberships(oidc_issuer, oidc_subject, tenant_id)
    WHERE revoked_at IS NULL;

-- Demonstration-only metadata, NOT input profiles or user identifiers.
CREATE TABLE IF NOT EXISTS saas.sandbox_runs (
    id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL REFERENCES saas.organizations(id) ON DELETE CASCADE,
    created_by_subject text NOT NULL CHECK (length(created_by_subject) BETWEEN 1 AND 255),
    total_profiles integer NOT NULL CHECK (total_profiles BETWEEN 1 AND 200),
    over_threshold integer NOT NULL CHECK (over_threshold >= 0 AND over_threshold <= total_profiles),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS saas_sandbox_runs_tenant
    ON saas.sandbox_runs(tenant_id, created_at DESC);

ALTER TABLE saas.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas.organizations FORCE ROW LEVEL SECURITY;
ALTER TABLE saas.memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas.memberships FORCE ROW LEVEL SECURITY;
ALTER TABLE saas.sandbox_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas.sandbox_runs FORCE ROW LEVEL SECURITY;

-- A missing transaction-local GUC returns NULL and matches no tenant.
CREATE POLICY organizations_read_tenant ON saas.organizations
    FOR SELECT TO churn_saas_app
    USING (id = nullif(current_setting('app.tenant_id', true), '')::uuid);

CREATE POLICY memberships_read_self ON saas.memberships
    FOR SELECT TO churn_saas_app
    USING (
        tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid
        AND oidc_issuer = nullif(current_setting('app.oidc_issuer', true), '')
        AND oidc_subject = nullif(current_setting('app.oidc_subject', true), '')
    );

CREATE POLICY sandbox_runs_read_tenant ON saas.sandbox_runs
    FOR SELECT TO churn_saas_app
    USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

CREATE POLICY sandbox_runs_insert_self ON saas.sandbox_runs
    FOR INSERT TO churn_saas_app
    WITH CHECK (
        tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid
        AND created_by_subject = nullif(current_setting('app.oidc_subject', true), '')
    );

REVOKE ALL ON SCHEMA saas FROM PUBLIC;
GRANT USAGE ON SCHEMA saas TO churn_saas_app;
REVOKE ALL ON ALL TABLES IN SCHEMA saas FROM PUBLIC;
GRANT SELECT ON saas.organizations, saas.memberships TO churn_saas_app;
GRANT SELECT, INSERT ON saas.sandbox_runs TO churn_saas_app;

COMMIT;
