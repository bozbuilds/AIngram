/**
 * Anonymous CLI telemetry ingest (aingram Lite) → Supabase `cli_telemetry`.
 * Public URL: POST https://api.aingram.dev/v1/telemetry (see vercel.json rewrite).
 *
 * Requires `landing_page/supabase/cli_telemetry.sql` applied once.
 * Env: same as waitlist — SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY (or anon if policy allows; service_role recommended).
 */
function getSupabaseConfig() {
  const supabaseUrl = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey =
    process.env.SUPABASE_SERVICE_ROLE_KEY ||
    process.env.SUPABASE_ANON_KEY ||
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  return { supabaseUrl, supabaseKey };
}

function isUuid(s) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    String(s),
  );
}

function validatePayload(body) {
  if (!body || typeof body !== 'object') {
    return { ok: false, error: 'Invalid JSON body' };
  }
  const { schema_version, kind, install_id, command, aingram_version } = body;
  if (schema_version !== 1 || typeof schema_version !== 'number') {
    return { ok: false, error: 'Invalid schema_version' };
  }
  if (kind !== 'cli_invocation' || typeof kind !== 'string') {
    return { ok: false, error: 'Invalid kind' };
  }
  if (!install_id || typeof install_id !== 'string' || !isUuid(install_id)) {
    return { ok: false, error: 'Invalid install_id' };
  }
  if (!command || typeof command !== 'string' || command.length > 128) {
    return { ok: false, error: 'Invalid command' };
  }
  if (!aingram_version || typeof aingram_version !== 'string' || aingram_version.length > 64) {
    return { ok: false, error: 'Invalid aingram_version' };
  }
  return {
    ok: true,
    row: {
      schema_version,
      kind,
      install_id,
      command,
      aingram_version,
    },
  };
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).end('Method Not Allowed');
  }

  const { supabaseUrl, supabaseKey } = getSupabaseConfig();
  if (!supabaseUrl || !supabaseKey) {
    console.error('telemetry: missing Supabase env');
    return res.status(500).json({ error: 'Database configuration error' });
  }

  const parsed = validatePayload(req.body);
  if (!parsed.ok) {
    return res.status(400).json({ error: parsed.error });
  }

  try {
    const response = await fetch(`${supabaseUrl}/rest/v1/cli_telemetry`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        apikey: supabaseKey,
        Authorization: `Bearer ${supabaseKey}`,
        Prefer: 'return=minimal',
      },
      body: JSON.stringify(parsed.row),
    });

    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.json();
      } catch (_e) {
        errorData = { message: response.statusText };
      }
      console.error('telemetry Supabase error:', response.status, errorData);
      return res.status(500).json({ error: 'Failed to store telemetry' });
    }

    return res.status(204).end();
  } catch (err) {
    console.error('telemetry fetch error:', err);
    return res.status(500).json({ error: 'Internal Server Error' });
  }
}
