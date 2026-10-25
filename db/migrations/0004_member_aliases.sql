-- Private staging only. Apply with a migration-owner role before rebuilding the app.
-- Add self-managed fictional display aliases without copying Keycloak attributes.
-- Never expose write access to saas.memberships to the application DB role.
BEGIN;

ALTER TABLE saas.memberships
    ADD COLUMN display_alias text
    CHECK (display_alias IS NULL OR (
        length(display_alias) BETWEEN 2 AND 40
        AND display_alias = btrim(display_alias)
        AND display_alias !~ '[[:cntrl:]<>@]'
    ));

-- Duplicate active aliases within one organization would make role changes ambiguous.
CREATE UNIQUE INDEX saas_member_alias_unique_active
    ON saas.memberships (tenant_id, lower(display_alias))
    WHERE display_alias IS NOT NULL AND revoked_at IS NULL;

CREATE FUNCTION saas.set_my_display_alias(p_tenant uuid, p_alias text)
RETURNS text LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE
    v_issuer text := nullif(current_setting('app.oidc_issuer', true), '');
    v_subject text := saas._v18_identity();
    v_alias text := btrim(p_alias);
BEGIN
    IF v_alias IS NULL OR length(v_alias) NOT BETWEEN 2 AND 40
       OR v_alias ~ '[[:cntrl:]<>@]' THEN
        RAISE EXCEPTION 'Invalid display alias' USING ERRCODE = 'P0001';
    END IF;
    UPDATE saas.memberships
       SET display_alias = v_alias
     WHERE tenant_id = p_tenant
       AND oidc_issuer = v_issuer
       AND oidc_subject = v_subject
       AND revoked_at IS NULL;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Not authorized' USING ERRCODE = 'P0001';
    END IF;
    RETURN v_alias;
END
$fn$;

-- Owners may read only active membership aliases for their own organization.
-- The function is fixed-purpose; callers cannot enumerate foreign tenants.
CREATE FUNCTION saas.list_organization_member_aliases(p_tenant uuid)
RETURNS TABLE(member_subject text, member_role text, member_alias text)
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, pg_temp
AS $fn$
BEGIN
    PERFORM saas._v18_identity();
    PERFORM saas._v18_require_owner(p_tenant);
    RETURN QUERY SELECT m.oidc_subject, m.role, m.display_alias
      FROM saas.memberships m
      WHERE m.tenant_id = p_tenant
        AND m.oidc_issuer = current_setting('app.oidc_issuer')
        AND m.revoked_at IS NULL
      ORDER BY m.created_at, m.oidc_subject LIMIT 100;
END
$fn$;

REVOKE ALL ON FUNCTION saas.set_my_display_alias(uuid,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION saas.list_organization_member_aliases(uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION saas.set_my_display_alias(uuid,text) TO churn_saas_app;
GRANT EXECUTE ON FUNCTION saas.list_organization_member_aliases(uuid) TO churn_saas_app;
COMMIT;
