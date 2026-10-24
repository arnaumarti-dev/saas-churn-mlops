-- STAGING ONLY. Apply as an audited migration OWNER with authority to
-- execute SECURITY DEFINER functions against FORCE RLS tables. The app role
-- receives EXECUTE ONLY; never give it owner or BYPASSRLS privileges.
-- Requires PostgreSQL >= 13 for built-in gen_random_uuid().
BEGIN;

CREATE TABLE saas.invitations (
    token_hash text PRIMARY KEY CHECK (token_hash ~ '^[0-9a-f]{64}$'),
    tenant_id uuid NOT NULL REFERENCES saas.organizations(id) ON DELETE CASCADE,
    oidc_issuer text NOT NULL,
    role text NOT NULL CHECK (role IN ('analyst','viewer')),
    issued_by_subject text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    consumed_at timestamptz,
    accepted_by_subject text
);
CREATE INDEX saas_invites_by_tenant ON saas.invitations(tenant_id, expires_at);
ALTER TABLE saas.invitations ENABLE ROW LEVEL SECURITY;
ALTER TABLE saas.invitations FORCE ROW LEVEL SECURITY;
REVOKE ALL ON saas.invitations FROM PUBLIC;

-- Use SECURITY DEFINER only for six narrow audited business operations.
-- Never create a generic arbitrary-SQL function or give direct table writes.
CREATE FUNCTION saas._v18_identity() RETURNS text
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE
    v_issuer text := nullif(current_setting('app.oidc_issuer', true), '');
    v_subject text := nullif(current_setting('app.oidc_subject', true), '');
BEGIN
    IF v_issuer IS NULL OR length(v_issuer) NOT BETWEEN 8 AND 500
       OR v_subject IS NULL OR length(v_subject) NOT BETWEEN 1 AND 255 THEN
        RAISE EXCEPTION 'Not authorized' USING ERRCODE = 'P0001';
    END IF;
    RETURN v_subject;
END
$fn$;

CREATE FUNCTION saas._v18_require_owner(p_tenant uuid) RETURNS void
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM saas.memberships m
        WHERE m.tenant_id = p_tenant
          AND m.oidc_issuer = nullif(current_setting('app.oidc_issuer', true), '')
          AND m.oidc_subject = saas._v18_identity()
          AND m.role = 'owner' AND m.revoked_at IS NULL
    ) THEN
        RAISE EXCEPTION 'Not authorized' USING ERRCODE = 'P0001';
    END IF;
END
$fn$;

CREATE FUNCTION saas.create_organization(p_name text) RETURNS uuid
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE
    v_tenant uuid := gen_random_uuid();
    v_name text := btrim(p_name);
    v_subject text := saas._v18_identity();
BEGIN
    IF v_name IS NULL OR length(v_name) NOT BETWEEN 1 AND 120 THEN
        RAISE EXCEPTION 'Invalid organization name' USING ERRCODE = 'P0001';
    END IF;
    IF (SELECT count(*) FROM saas.memberships m
        WHERE m.oidc_issuer = current_setting('app.oidc_issuer')
          AND m.oidc_subject = v_subject
          AND m.role = 'owner' AND m.revoked_at IS NULL) >= 10 THEN
        RAISE EXCEPTION 'Organization limit reached' USING ERRCODE = 'P0001';
    END IF;
    INSERT INTO saas.organizations(id, display_name) VALUES (v_tenant, v_name);
    INSERT INTO saas.memberships(tenant_id, oidc_issuer, oidc_subject, role)
      VALUES (v_tenant, current_setting('app.oidc_issuer'), v_subject, 'owner');
    RETURN v_tenant;
END
$fn$;

CREATE FUNCTION saas.create_invitation(p_tenant uuid, p_role text, p_hash text)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
BEGIN
    PERFORM saas._v18_identity();
    -- Serialize owner operations and re-check the role under the lock.
    PERFORM 1 FROM saas.organizations WHERE id = p_tenant FOR UPDATE;
    PERFORM saas._v18_require_owner(p_tenant);
    IF p_role NOT IN ('analyst','viewer') OR p_role IS NULL
       OR p_hash IS NULL OR p_hash !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'Invalid invitation' USING ERRCODE = 'P0001';
    END IF;
    IF (SELECT count(*) FROM saas.invitations i
        WHERE i.tenant_id = p_tenant AND i.consumed_at IS NULL
          AND i.expires_at > now()) >= 10 THEN
        RAISE EXCEPTION 'Invitation limit reached' USING ERRCODE = 'P0001';
    END IF;
    INSERT INTO saas.invitations(
        token_hash, tenant_id, oidc_issuer, role, issued_by_subject, expires_at
    ) VALUES (
        p_hash, p_tenant, current_setting('app.oidc_issuer'), p_role,
        saas._v18_identity(), now() + interval '24 hours'
    );
END
$fn$;

CREATE FUNCTION saas.accept_invitation(p_hash text) RETURNS uuid
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE
    v_subject text := saas._v18_identity();
    v_invite saas.invitations%ROWTYPE;
BEGIN
    IF p_hash IS NULL OR p_hash !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'Invalid invitation' USING ERRCODE = 'P0001';
    END IF;
    SELECT * INTO v_invite FROM saas.invitations
      WHERE token_hash = p_hash AND consumed_at IS NULL AND expires_at > now()
      FOR UPDATE;
    IF NOT FOUND OR v_invite.oidc_issuer <> current_setting('app.oidc_issuer') THEN
        RAISE EXCEPTION 'Invalid invitation' USING ERRCODE = 'P0001';
    END IF;
    PERFORM 1 FROM saas.organizations WHERE id = v_invite.tenant_id FOR UPDATE;
    IF EXISTS (
        SELECT 1 FROM saas.memberships m
        WHERE m.tenant_id = v_invite.tenant_id
          AND m.oidc_issuer = v_invite.oidc_issuer
          AND m.oidc_subject = v_subject AND m.revoked_at IS NULL
    ) THEN
        RAISE EXCEPTION 'Already a member' USING ERRCODE = 'P0001';
    END IF;
    INSERT INTO saas.memberships(tenant_id, oidc_issuer, oidc_subject, role)
    VALUES (v_invite.tenant_id, v_invite.oidc_issuer, v_subject, v_invite.role)
    ON CONFLICT (tenant_id, oidc_issuer, oidc_subject)
    DO UPDATE SET role = EXCLUDED.role, revoked_at = NULL, created_at = now();
    UPDATE saas.invitations SET consumed_at = now(), accepted_by_subject = v_subject
      WHERE token_hash = p_hash;
    RETURN v_invite.tenant_id;
END
$fn$;

CREATE FUNCTION saas.list_organization_members(p_tenant uuid)
RETURNS TABLE(member_subject text, member_role text, joined_at timestamptz)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
BEGIN
    PERFORM saas._v18_identity();
    PERFORM saas._v18_require_owner(p_tenant);
    RETURN QUERY SELECT m.oidc_subject, m.role, m.created_at
        FROM saas.memberships m
        WHERE m.tenant_id = p_tenant
          AND m.oidc_issuer = current_setting('app.oidc_issuer')
          AND m.revoked_at IS NULL
        ORDER BY m.created_at, m.oidc_subject LIMIT 100;
END
$fn$;

CREATE FUNCTION saas.change_member_role(
    p_tenant uuid, p_subject text, p_role text
) RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE v_old_role text;
BEGIN
    PERFORM saas._v18_identity();
    PERFORM 1 FROM saas.organizations WHERE id = p_tenant FOR UPDATE;
    PERFORM saas._v18_require_owner(p_tenant);
    IF p_role IS NULL OR p_role NOT IN ('owner','analyst','viewer')
       OR p_subject IS NULL OR length(p_subject) NOT BETWEEN 1 AND 255 THEN
        RAISE EXCEPTION 'Invalid membership' USING ERRCODE = 'P0001';
    END IF;
    SELECT m.role INTO v_old_role FROM saas.memberships m
      WHERE m.tenant_id = p_tenant
        AND m.oidc_issuer = current_setting('app.oidc_issuer')
        AND m.oidc_subject = p_subject AND m.revoked_at IS NULL;
    IF v_old_role IS NULL THEN
        RAISE EXCEPTION 'Member not found' USING ERRCODE = 'P0001';
    END IF;
    IF v_old_role = 'owner' AND p_role <> 'owner' AND (
        SELECT count(*) FROM saas.memberships m
        WHERE m.tenant_id = p_tenant AND m.role = 'owner' AND m.revoked_at IS NULL
    ) <= 1 THEN
        RAISE EXCEPTION 'Last owner cannot be removed' USING ERRCODE = 'P0001';
    END IF;
    UPDATE saas.memberships SET role = p_role
      WHERE tenant_id = p_tenant
        AND oidc_issuer = current_setting('app.oidc_issuer')
        AND oidc_subject = p_subject AND revoked_at IS NULL;
END
$fn$;

CREATE FUNCTION saas.revoke_organization_member(p_tenant uuid, p_subject text)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE v_old_role text;
BEGIN
    PERFORM saas._v18_identity();
    PERFORM 1 FROM saas.organizations WHERE id = p_tenant FOR UPDATE;
    PERFORM saas._v18_require_owner(p_tenant);
    SELECT m.role INTO v_old_role FROM saas.memberships m
      WHERE m.tenant_id = p_tenant AND m.oidc_issuer = current_setting('app.oidc_issuer')
        AND m.oidc_subject = p_subject AND m.revoked_at IS NULL;
    IF v_old_role IS NULL THEN
        RAISE EXCEPTION 'Member not found' USING ERRCODE = 'P0001';
    END IF;
    IF v_old_role = 'owner' AND (
        SELECT count(*) FROM saas.memberships m
        WHERE m.tenant_id = p_tenant AND m.role = 'owner' AND m.revoked_at IS NULL
    ) <= 1 THEN
        RAISE EXCEPTION 'Last owner cannot be removed' USING ERRCODE = 'P0001';
    END IF;
    UPDATE saas.memberships SET revoked_at = now()
      WHERE tenant_id = p_tenant
        AND oidc_issuer = current_setting('app.oidc_issuer')
        AND oidc_subject = p_subject AND revoked_at IS NULL;
END
$fn$;

-- PostgreSQL functions default to PUBLIC EXECUTE: close that door before commit.
REVOKE ALL ON FUNCTION saas._v18_identity() FROM PUBLIC;
REVOKE ALL ON FUNCTION saas._v18_require_owner(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION saas.create_organization(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION saas.create_invitation(uuid,text,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION saas.accept_invitation(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION saas.list_organization_members(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION saas.change_member_role(uuid,text,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION saas.revoke_organization_member(uuid,text) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION saas.create_organization(text) TO churn_saas_app;
GRANT EXECUTE ON FUNCTION saas.create_invitation(uuid,text,text) TO churn_saas_app;
GRANT EXECUTE ON FUNCTION saas.accept_invitation(text) TO churn_saas_app;
GRANT EXECUTE ON FUNCTION saas.list_organization_members(uuid) TO churn_saas_app;
GRANT EXECUTE ON FUNCTION saas.change_member_role(uuid,text,text) TO churn_saas_app;
GRANT EXECUTE ON FUNCTION saas.revoke_organization_member(uuid,text) TO churn_saas_app;
COMMIT;
