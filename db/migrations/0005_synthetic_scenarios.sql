-- Churn Lab v2.5: store ONLY a server-defined scenario label on aggregate
-- fictional test runs. Keep the existing 0001-0004 data and RLS policies.
-- Apply ONCE as the local migration owner; never with the app DB role.
BEGIN;
ALTER TABLE saas.sandbox_runs
  ADD COLUMN scenario_key text NOT NULL DEFAULT 'baseline'
  CHECK (scenario_key IN ('baseline','at_risk','loyal'));
COMMENT ON COLUMN saas.sandbox_runs.scenario_key IS
  'Server-curated synthetic scenario key, not a customer segment or identity.';
COMMIT;
