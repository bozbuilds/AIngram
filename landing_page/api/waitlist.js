export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { email } = req.body;
  
  if (!email || !email.includes('@')) {
    return res.status(400).json({ error: 'Invalid email address' });
  }

  // Vercel auto-injects these when you link Supabase. Prefer SERVICE_ROLE_KEY so
  // waitlist inserts still work with RLS enabled on `public.waitlist` (service role bypasses RLS).
  const supabaseUrl = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    console.error("Missing Supabase Environment Variables");
    return res.status(500).json({ error: 'Database configuration error in Vercel. Have you linked Supabase?' });
  }

  try {
    // Calling the Supabase REST API directly via fetch to avoid needing a package.json
    // We assume the table is named "waitlist"
    const response = await fetch(`${supabaseUrl}/rest/v1/waitlist`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'apikey': supabaseKey,
        'Authorization': `Bearer ${supabaseKey}`,
        'Prefer': 'return=representation'
      },
      body: JSON.stringify({ email })
    });

    if (!response.ok) {
        let errorData;
        try { errorData = await response.json(); } catch(e) { errorData = { message: response.statusText } }
        
        const msg = errorData.message || JSON.stringify(errorData);
        // Supabase Postgres unique violation error code is "23505"
        if (msg.includes('duplicate key') || errorData.code === '23505') {
             return res.status(400).json({ error: 'Email already registered' });
        }
        console.error("Supabase Error Data:", errorData);
        return res.status(500).json({ error: 'Failed to add to waitlist database' });
    }

    const data = await response.json();
    return res.status(200).json({ message: 'Success', result: data });

  } catch (error) {
    console.error("Fetch Error:", error);
    return res.status(500).json({ error: 'Internal Server Error' });
  }
}
