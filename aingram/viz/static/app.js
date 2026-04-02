/* global d3 */

(function () {
  const main = document.getElementById('main-panel');
  const inspector = document.getElementById('inspector');
  const statsEl = document.getElementById('stats-summary');
  const loadingEl = document.getElementById('loading');

  function typeClass(t) {
    const k = (t || 'unknown').toLowerCase();
    return 'type-' + k;
  }

  function extractText(content) {
    try {
      const o = JSON.parse(content);
      if (o && typeof o === 'object' && o.text != null) return String(o.text);
    } catch (_) {}
    return content || '';
  }

  async function fetchStats() {
    const r = await fetch('/api/stats');
    const d = await r.json();
    statsEl.textContent =
      'Entries: ' + d.entry_count + ' · Entities: ' + d.entity_count + ' · Chains: ' + d.chain_count;
  }

  async function fetchEntry(id) {
    const r = await fetch('/api/entry?id=' + encodeURIComponent(id));
    if (!r.ok) {
      inspector.textContent = 'Not found';
      return;
    }
    const e = await r.json();
    const text = extractText(e.content);
    inspector.innerHTML =
      '<div class="' +
      typeClass(e.entry_type) +
      '"><strong>' +
      escapeHtml(e.entry_type) +
      '</strong></div>' +
      '<p>ID: <code>' +
      escapeHtml(e.entry_id) +
      '</code></p>' +
      '<p>Confidence: ' +
      (e.confidence != null ? e.confidence : '—') +
      '</p>' +
      '<pre>' +
      escapeHtml(text) +
      '</pre>';
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  let sim;

  function renderGraph(data) {
    main.innerHTML = '';
    const w = main.clientWidth || 600;
    const h = Math.max(400, window.innerHeight * 0.65);
    const svg = d3
      .select(main)
      .append('svg')
      .attr('id', 'graph-svg')
      .attr('viewBox', [0, 0, w, h]);

    const nodes = (data.nodes || []).map((d) => Object.assign({}, d));
    const links = (data.edges || []).map((d) => ({
      source: d.source,
      target: d.target,
      weight: d.weight,
    }));

    const idToNode = new Map(nodes.map((n) => [n.id, n]));
    links.forEach((l) => {
      if (typeof l.source === 'string') l.source = idToNode.get(l.source) || l.source;
      if (typeof l.target === 'string') l.target = idToNode.get(l.target) || l.target;
    });

    const color = d3.scaleOrdinal(d3.schemeTableau10);

    if (sim) sim.stop();
    sim = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink(links)
          .id((d) => d.id)
          .distance(80)
          .strength(0.7)
      )
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(w / 2, h / 2));

    const link = svg
      .append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('class', 'graph-link')
      .attr('stroke-width', (d)         => Math.sqrt((d.weight || 1)) * 0.8);

    const node = svg
      .append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', 'graph-node')
      .call(
        d3
          .drag()
          .on('start', (ev, d) => {
            if (!ev.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (ev, d) => {
            d.fx = ev.x;
            d.fy = ev.y;
          })
          .on('end', (ev, d) => {
            if (!ev.active) sim.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      )
      .on('click', (_, d) => fetchEntry(d.id));

    node
      .append('circle')
      .attr('r', 8)
      .attr('fill', (d) => color(d.type || 'x'));

    node
      .append('text')
      .text((d) => (d.name || d.id).slice(0, 24))
      .attr('x', 12)
      .attr('y', 4)
      .attr('fill', '#e6edf3')
      .style('font-size', '11px');

    sim.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);
      node.attr('transform', (d) => 'translate(' + d.x + ',' + d.y + ')');
    });
  }

  function renderChains(chains) {
    main.innerHTML = '<div class="chain-timeline"></div>';
    const box = main.querySelector('.chain-timeline');
    (chains || []).forEach((ch) => {
      const card = document.createElement('div');
      card.className = 'chain-card';
      card.innerHTML =
        '<strong>' +
        escapeHtml(ch.title || ch.chain_id) +
        '</strong> <span class="type-meta">(' +
        escapeHtml(ch.status || '') +
        ')</span>';
      (ch.entries || []).forEach((en) => {
        const row = document.createElement('div');
        row.className = 'chain-entry ' + typeClass(en.type);
        row.textContent =
          (en.type || '') + ': ' + extractText(en.content).slice(0, 160);
        row.addEventListener('click', () => fetchEntry(en.entry_id));
        card.appendChild(row);
      });
      box.appendChild(card);
    });
  }

  async function fetchEntities() {
    loadingEl.hidden = false;
    try {
      const r = await fetch('/api/entities');
      const d = await r.json();
      renderGraph(d);
    } finally {
      loadingEl.hidden = true;
    }
  }

  async function fetchChains() {
    loadingEl.hidden = false;
    try {
      const r = await fetch('/api/chains');
      const d = await r.json();
      renderChains(d);
    } finally {
      loadingEl.hidden = true;
    }
  }

  document.querySelectorAll('.tab').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((b) => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const tab = btn.getAttribute('data-tab');
      if (tab === 'graph') fetchEntities();
      else fetchChains();
    });
  });

  fetchStats().catch(() => {
    statsEl.textContent = 'Stats unavailable';
  });
  fetchEntities();

  window.addEventListener('resize', () => {
    const active = document.querySelector('.tab.active');
    if (active && active.getAttribute('data-tab') === 'graph') fetchEntities();
  });
})();
