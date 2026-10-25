#!/bin/sh
# Executed by the disposable Postgres container only on first initialization.
set -eu
: "${CHURN_DB_APP_PASSWORD:?}"
: "${KEYCLOAK_DB_PASSWORD:?}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
    -v app_pw="$CHURN_DB_APP_PASSWORD" -v kc_pw="$KEYCLOAK_DB_PASSWORD" <<'SQL'
SELECT format(
  'CREATE ROLE churn_saas_app LOGIN PASSWORD %L NOSUPERUSER NOBYPASSRLS',
  :'app_pw'
) \gexec
SELECT format(
  'CREATE ROLE churn_keycloak LOGIN PASSWORD %L NOSUPERUSER NOBYPASSRLS',
  :'kc_pw'
) \gexec
CREATE DATABASE churn_saas_staging OWNER postgres;
CREATE DATABASE keycloak_staging OWNER churn_keycloak;
SQL

for migration in /churn-migrations/0001_saas_tenants_rls.sql \
                 /churn-migrations/0002_member_org_selector.sql \
                 /churn-migrations/0003_organization_management.sql \
                 /churn-migrations/0004_member_aliases.sql \
                 /churn-migrations/0005_synthetic_scenarios.sql; do
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" \
      --dbname churn_saas_staging -f "$migration"
done
