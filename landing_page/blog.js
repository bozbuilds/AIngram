// ─── Configuration ────────────────────────────────────────────────────────────
// Change this to your dev.to username.
const DEV_TO_USERNAME = 'bozbuilds';
const PER_PAGE = 12;
// ──────────────────────────────────────────────────────────────────────────────

const API_URL = `https://dev.to/api/articles?username=${DEV_TO_USERNAME}&per_page=${PER_PAGE}&state=published`;

// Deterministic tag → color from a palette that complements the site theme.
const TAG_PALETTE = [
  { bg: 'rgba(59,130,246,0.12)',  border: 'rgba(59,130,246,0.25)',  text: '#60a5fa' },
  { bg: 'rgba(139,92,246,0.12)', border: 'rgba(139,92,246,0.25)',  text: '#a78bfa' },
  { bg: 'rgba(16,185,129,0.12)', border: 'rgba(16,185,129,0.25)',  text: '#34d399' },
  { bg: 'rgba(251,191,36,0.10)', border: 'rgba(251,191,36,0.22)',  text: '#fbbf24' },
  { bg: 'rgba(239,68,68,0.10)',  border: 'rgba(239,68,68,0.22)',   text: '#f87171' },
  { bg: 'rgba(6,182,212,0.10)',  border: 'rgba(6,182,212,0.22)',   text: '#22d3ee' },
];

function tagStyle(tag) {
  let hash = 0;
  for (let i = 0; i < tag.length; i++) hash = (hash * 31 + tag.charCodeAt(i)) % TAG_PALETTE.length;
  return TAG_PALETTE[Math.abs(hash)];
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function tagPill(tag) {
  const s = tagStyle(tag);
  return `<span class="blog-tag" style="background:${s.bg};border-color:${s.border};color:${s.text}">#${tag}</span>`;
}

function coverFallback(post) {
  // A gradient placeholder when no cover image is set — uses the tag palette
  // seeded by the post title so each post gets a unique gradient.
  const s = tagStyle(post.title.slice(0, 4));
  const s2 = tagStyle(post.title.slice(-4));
  return `<div class="blog-cover-fallback" style="background:linear-gradient(135deg,${s.bg.replace('0.12', '0.4')} 0%,${s2.bg.replace('0.12', '0.2')} 100%)">
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="${s.text}" stroke-width="1.2" opacity="0.6"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
  </div>`;
}

function renderFeatured(post) {
  const tags = (post.tag_list || []).slice(0, 4).map(tagPill).join('');
  const cover = post.cover_image
    ? `<img class="blog-featured-img" src="${escHtml(post.cover_image)}" alt="${escHtml(post.title)}" loading="lazy">`
    : coverFallback(post);

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
    </a>`;
}

function renderCard(post) {
  const tags = (post.tag_list || []).slice(0, 3).map(tagPill).join('');
  const cover = post.cover_image
    ? `<img class="blog-card-img" src="${escHtml(post.cover_image)}" alt="${escHtml(post.title)}" loading="lazy">`
    : coverFallback(post);

  return `
    <a class="blog-card" href="${escHtml(post.url)}" target="_blank" rel="noopener">
      <div class="blog-card-media">${cover}</div>
      <div class="blog-card-body">
        <div class="blog-tags">${tags}</div>
        <h3 class="blog-card-title">${escHtml(post.title)}</h3>
        <p class="blog-card-desc">${escHtml((post.description || '').slice(0, 120))}${(post.description || '').length > 120 ? '…' : ''}</p>
        <div class="blog-card-footer">
          <span class="blog-meta-date">${formatDate(post.published_at)}</span>
          <span class="blog-meta-sep">·</span>
          <span class="blog-meta-read">${post.reading_time_minutes} min read</span>
        </div>
      </div>
    </a>`;
}

// Minimal HTML escaping — prevents XSS from API-sourced strings.
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

async function init() {
  const loadingEl   = document.getElementById('blog-loading');
  const errorEl     = document.getElementById('blog-error');
  const errorMsgEl  = document.getElementById('blog-error-msg');
  const featuredEl  = document.getElementById('blog-featured');
  const gridSection = document.getElementById('blog-grid-section');
  const gridEl      = document.getElementById('blog-grid');
  const emptyEl     = document.getElementById('blog-empty');

  // Static content was baked in at build time — just run animations and exit.
  if (featuredEl && !featuredEl.hidden) {
    if (loadingEl) loadingEl.hidden = true;
    requestAnimationFrame(() => {
      document.querySelectorAll('.blog-card').forEach((card, i) => {
        card.style.animationDelay = `${i * 60}ms`;
        card.classList.add('blog-card-animate');
      });
    });
    return;
  }

  try {
    const res = await fetch(API_URL);
    if (!res.ok) throw new Error(`dev.to API returned ${res.status}`);
    const posts = await res.json();

    loadingEl.hidden = true;

    if (!posts.length) {
      emptyEl.hidden = false;
      return;
    }

    const [featured, ...rest] = posts;

    featuredEl.innerHTML = renderFeatured(featured);
    featuredEl.hidden = false;

    if (rest.length) {
      gridEl.innerHTML = rest.map(renderCard).join('');
      gridSection.hidden = false;
    }

    // Stagger-animate the cards in
    requestAnimationFrame(() => {
      document.querySelectorAll('.blog-card').forEach((card, i) => {
        card.style.animationDelay = `${i * 60}ms`;
        card.classList.add('blog-card-animate');
      });
    });

  } catch (err) {
    loadingEl.hidden = true;
    if (errorMsgEl) errorMsgEl.textContent = `Could not load posts from dev.to. ${err.message}`;
    errorEl.hidden = false;
  }
}

document.addEventListener('DOMContentLoaded', init);
