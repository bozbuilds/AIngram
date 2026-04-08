#!/usr/bin/env node
// build.js — Bakes dev.to posts into blog.html at Vercel deploy time.
// Uses only Node.js built-ins (no npm dependencies).
// If the API is unreachable, exits cleanly so the dynamic JS fallback remains.

const https = require('https')
const fs = require('fs')
const path = require('path')

const DEV_TO_USERNAME = 'bozbuilds'
const PER_PAGE = 12
const BLOG_HTML = path.join(__dirname, 'blog.html')

const TAG_PALETTE = [
  { bg: 'rgba(59,130,246,0.12)',  border: 'rgba(59,130,246,0.25)',  text: '#60a5fa' },
  { bg: 'rgba(139,92,246,0.12)', border: 'rgba(139,92,246,0.25)',  text: '#a78bfa' },
  { bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.25)',  text: '#34d399' },
  { bg: 'rgba(251,191,36,0.10)', border: 'rgba(251,191,36,0.22)',  text: '#fbbf24' },
  { bg: 'rgba(239,68,68,0.10)',  border: 'rgba(239,68,68,0.22)',   text: '#f87171' },
  { bg: 'rgba(6,182,212,0.10)',  border: 'rgba(6,182,212,0.22)',   text: '#22d3ee' },
]

function tagStyle(tag) {
  let hash = 0
  for (let i = 0; i < tag.length; i++) hash = (hash * 31 + tag.charCodeAt(i)) % TAG_PALETTE.length
  return TAG_PALETTE[Math.abs(hash)]
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

function tagPill(tag) {
  const s = tagStyle(tag)
  return `<span class="blog-tag" style="background:${s.bg};border-color:${s.border};color:${s.text}">#${tag}</span>`
}

function coverFallback(post) {
  const s = tagStyle(post.title.slice(0, 4))
  const s2 = tagStyle(post.title.slice(-4))
  return `<div class="blog-cover-fallback" style="background:linear-gradient(135deg,${s.bg.replace('0.12', '0.4')} 0%,${s2.bg.replace('0.12', '0.2')} 100%)">
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="${s.text}" stroke-width="1.2" opacity="0.6"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
  </div>`
}

function renderFeatured(post) {
  const tags = (post.tag_list || []).slice(0, 4).map(tagPill).join('')
  const cover = post.cover_image
    ? `<img class="blog-featured-img" src="${escHtml(post.cover_image)}" alt="${escHtml(post.title)}" loading="eager">`
    : coverFallback(post)
  return `
    <a class="blog-featured-card" href="${escHtml(post.url)}" target="_blank" rel="noopener">
      <div class="blog-featured-media">${cover}</div>
      <div class="blog-featured-body">
        <div class="blog-featured-meta">
          <span class="blog-featured-label">Featured post</span>
          <span class="blog-meta-sep">·</span>
          <span class="blog-meta-date">${formatDate(post.published_at)}</span>
          <span class="blog-meta-sep">·</span>
          <span class="blog-meta-read">${post.reading_time_minutes} min read</span>
        </div>
        <h2 class="blog-featured-title">${escHtml(post.title)}</h2>
        <p class="blog-featured-desc">${escHtml(post.description || '')}</p>
        <div class="blog-featured-footer">
          <div class="blog-tags">${tags}</div>
          <div class="blog-reactions">
            <span class="blog-reaction-item">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>
              ${post.positive_reactions_count}
            </span>
            <span class="blog-reaction-item">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              ${post.comments_count}
            </span>
          </div>
        </div>
      </div>
    </a>`
}

function renderCard(post) {
  const tags = (post.tag_list || []).slice(0, 3).map(tagPill).join('')
  const cover = post.cover_image
    ? `<img class="blog-card-img" src="${escHtml(post.cover_image)}" alt="${escHtml(post.title)}" loading="lazy">`
    : coverFallback(post)
  return `
    <a class="blog-card" href="${escHtml(post.url)}" target="_blank" rel="noopener">
      <div class="blog-card-media">${cover}</div>
      <div class="blog-card-body">
        <div class="blog-tags">${tags}</div>
        <h3 class="blog-card-title">${escHtml(post.title)}</h3>
        <p class="blog-card-desc">${escHtml((post.description || '').slice(0, 120))}${(post.description || '').length > 120 ? '\u2026' : ''}</p>
        <div class="blog-card-footer">
          <span class="blog-meta-date">${formatDate(post.published_at)}</span>
          <span class="blog-meta-sep">\u00b7</span>
          <span class="blog-meta-read">${post.reading_time_minutes} min read</span>
        </div>
      </div>
    </a>`
}

function fetchPosts() {
  return new Promise((resolve, reject) => {
    const url = `https://dev.to/api/articles?username=${DEV_TO_USERNAME}&per_page=${PER_PAGE}&state=published`
    const req = https.get(url, { headers: { 'User-Agent': 'aingram-build/1.0', 'Accept': 'application/json' } }, res => {
      let data = ''
      res.on('data', chunk => { data += chunk })
      res.on('end', () => {
        if (res.statusCode !== 200) { reject(new Error(`HTTP ${res.statusCode}`)); return }
        try { resolve(JSON.parse(data)) } catch (e) { reject(e) }
      })
    })
    req.on('error', reject)
    req.setTimeout(10000, () => { req.destroy(); reject(new Error('Request timed out')) })
  })
}

function buildArticleSchema(post) {
  return {
    '@type': 'BlogPosting',
    headline: post.title,
    description: post.description || '',
    url: post.url,
    datePublished: post.published_at,
    dateModified: post.edited_at || post.published_at,
    author: {
      '@type': 'Person',
      name: post.user?.name || DEV_TO_USERNAME,
      url: `https://dev.to/${DEV_TO_USERNAME}`
    },
    ...(post.cover_image ? { image: post.cover_image } : {}),
    keywords: (post.tag_list || []).join(', ')
  }
}

async function main() {
  console.log(`[build] Fetching posts from dev.to for @${DEV_TO_USERNAME}...`)

  let posts
  try {
    posts = await fetchPosts()
  } catch (err) {
    console.warn(`[build] dev.to fetch failed (${err.message}). Keeping dynamic blog.html.`)
    process.exit(0)
  }

  if (!Array.isArray(posts) || !posts.length) {
    console.log('[build] No posts found. Keeping dynamic blog.html.')
    process.exit(0)
  }

  console.log(`[build] Got ${posts.length} post(s). Generating static HTML...`)

  let html = fs.readFileSync(BLOG_HTML, 'utf8')

  const [featured, ...rest] = posts

  // 1. Remove loading skeleton section
  html = html.replace(
    /\s*<!-- Loading skeletons -->[\s\S]*?<\/div>\s*\n(\s*<!-- Error state -->)/,
    '\n\n        $1'
  )

  // 2. Populate and un-hide the featured post container
  html = html.replace(
    /<div id="blog-featured" hidden><\/div>/,
    `<div id="blog-featured">${renderFeatured(featured)}\n        </div>`
  )

  // 3. Populate and un-hide the grid section
  if (rest.length) {
    html = html.replace(
      /<div id="blog-grid-section" hidden>/,
      '<div id="blog-grid-section">'
    )
    html = html.replace(
      /<div id="blog-grid" class="blog-grid-wrap"><\/div>/,
      `<div id="blog-grid" class="blog-grid-wrap">\n            ${rest.map(renderCard).join('\n            ')}\n            </div>`
    )
  }

  // 4. Inject JSON-LD BlogPosting schema
  const schema = {
    '@context': 'https://schema.org',
    '@graph': [{
      '@type': 'Blog',
      name: 'The AIngram Blog',
      url: 'https://aingram.dev/blog',
      description: 'Technical deep-dives on memory systems, agent architectures, and local-first AI.',
      blogPost: posts.map(buildArticleSchema)
    }]
  }

  // Remove any previously injected schema (re-runs during local dev)
  html = html.replace(/<script type="application\/ld\+json">[\s\S]*?<\/script>\n/, '')
  html = html.replace(
    '</head>',
    `    <script type="application/ld+json">\n${JSON.stringify(schema, null, 2)}\n    </script>\n</head>`
  )

  fs.writeFileSync(BLOG_HTML, html, 'utf8')
  console.log(`[build] blog.html updated with ${posts.length} static post(s). Deploy ready.`)
}

main().catch(err => {
  console.error('[build] Unexpected error:', err.message)
  process.exit(0) // Never fail the Vercel build
})
