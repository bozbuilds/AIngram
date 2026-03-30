-- Run once in Supabase SQL Editor (same project as the waitlist).
-- Serverless inserts use SUPABASE_SERVICE_ROLE_KEY (bypasses RLS).

CREATE TABLE IF NOT EXISTS public.cli_telemetry (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  schema_version INTEGER NOT NULL,
  kind TEXT NOT NULL,
  install_id UUID NOT NULL,
  command TEXT NOT NULL,
  aingram_version TEXT NOT NULL
);

COMMENT ON TABLE public.cli_telemetry IS 'Anonymous aingram Lite CLI usage (no memory content).';

CREATE INDEX IF NOT EXISTS idx_cli_telemetry_created_at ON public.cli_telemetry (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cli_telemetry_install_id ON public.cli_telemetry (install_id);

ALTER TABLE public.cli_telemetry ENABLE ROW LEVEL SECURITY;

-- Lock down direct client access; Edge Function / Vercel uses service_role.
REVOKE ALL ON public.cli_telemetry FROM PUBLIC;
GRANT INSERT ON public.cli_telemetry TO service_role;
GRANT SELECT ON public.cli_telemetry TO service_role;
GRANT USAGE, SELECT ON SEQUENCE public.cli_telemetry_id_seq TO service_role;
