/**
 * Anonymous CLI telemetry ingest (aingram Lite).
 * Public URL: POST https://api.aingram.dev/v1/telemetry (see vercel.json rewrite).
 */
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).end('Method Not Allowed');
  }
  return res.status(204).end();
}
