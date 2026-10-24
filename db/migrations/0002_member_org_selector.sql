-- Private staging only. Run as the migration owner (never churn_saas_app).
-- This changes only the SELECT visibility of the caller's OWN membership rows,
-- so a signed-in user can select among their organizations.
-- The RLS policy cannot expose other users and has no write permissions.
BEGIN;

DROP POLICY IF EXISTS memberships_read_self ON saas.memberships;
CREATE POLICY memberships_read_identity ON saas.memberships
    FOR SELECT TO churn_saas_app
    USING (
        oidc_issuer = nullif(current_setting('app.oidc_issuer', true), '')
        AND oidc_subject = nullif(current_setting('app.oidc_subject', true), '')
    );

-- Other tenant-owned tables still require a transaction-local tenant ID,
-- and the app role keeps SELECT-only access to memberships.
ALTER TABLE saas.memberships FORCE ROW LEVEL SECURITY;
COMMIT;
