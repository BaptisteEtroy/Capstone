/**
 * features.js — Feature Browser panel
 * Paginated grid of labeled SAE features with search, filter, and detail drawer.
 */

import { api } from './app.js';

// ── State ──────────────────────────────────────────────────────────────────────
let currentPage   = 0;
let currentSearch = '';
let currentConf   = '';
let totalFeatures = 0;
let hasLoaded     = false;   // true once a successful load completes
let isLoading     = false;   // prevent concurrent fetches
const PAGE_SIZE   = 48;
let searchTimeout = null;

// ── Init ───────────────────────────────────────────────────────────────────────
export function initFeatures(appState) {
  const searchEl = document.getElementById('feat-search');
  const grid     = document.getElementById('features-grid');

  searchEl?.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      currentSearch = searchEl.value.trim();
      currentPage   = 0;
      hasLoaded     = false;
      loadFeatures();
    }, 350);
  });

  // Confidence filter
  document.querySelectorAll('.filter-pill[data-conf]').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.filter-pill').forEach(p => {
        p.classList.toggle('active', p === pill);
        p.setAttribute('aria-pressed', String(p === pill));
      });
      currentConf = pill.dataset.conf;
      currentPage = 0;
      loadFeatures();
    });
  });

  // Feature drawer close
  document.getElementById('feat-drawer-close')?.addEventListener('click', closeDrawer);
  document.getElementById('feat-drawer-backdrop')?.addEventListener('click', closeDrawer);

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeDrawer();
  });

  // Load when the features tab becomes active (fired by app.js switchTab)
  window.addEventListener('sae:tab', ({ detail }) => {
    if (detail.tab === 'features' && !hasLoaded) {
      loadFeatures();
    }
  });
}

// ── Load features ──────────────────────────────────────────────────────────────
async function loadFeatures() {
  if (isLoading) return;
  const grid = document.getElementById('features-grid');
  if (!grid) return;

  isLoading = true;
  grid.innerHTML = '<div style="grid-column:1/-1;padding:24px;text-align:center;color:var(--text-3);font-size:12px">Loading…</div>';

  try {
    const params = new URLSearchParams({
      page:       currentPage,
      limit:      PAGE_SIZE,
      search:     currentSearch,
      confidence: currentConf,
    });

    const data = await api(`/api/features?${params}`);
    totalFeatures = data.total;
    hasLoaded = true;

    const totalEl = document.getElementById('features-total');
    if (totalEl) totalEl.textContent = data.total.toLocaleString();

    renderGrid(data.features, grid);
    renderPagination(data.total, data.page);

  } catch (err) {
    hasLoaded = false;  // allow retry on next tab visit
    grid.innerHTML = `<div style="grid-column:1/-1;padding:24px;color:var(--text-2);font-size:12px">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    isLoading = false;
  }
}

// ── Render grid ────────────────────────────────────────────────────────────────
function renderGrid(features, grid) {
  grid.innerHTML = '';

  if (!features.length) {
    grid.innerHTML = '<div style="grid-column:1/-1;padding:32px;text-align:center;color:var(--text-3);font-size:13px">No features match your search.</div>';
    return;
  }

  features.forEach((feat, i) => {
    const card = document.createElement('div');
    card.className = 'feat-grid-card';
    card.setAttribute('role', 'listitem');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', `Feature ${feat.index}: ${feat.label}`);

    const topToks = (feat.top_tokens || []).slice(0, 3)
      .filter(Boolean)
      .map(t => `<span class="token-pill">${escapeHtml(t)}</span>`)
      .join('');

    const vocabToks = (feat.vocab_proj || []).slice(0, 3)
      .filter(Boolean)
      .map(t => `<span class="token-pill" style="border-color:rgba(255,255,255,0.08);color:var(--text-2)">${escapeHtml(t)}</span>`)
      .join('');

    card.innerHTML = `
      <div class="feat-grid-top">
        <span class="feat-grid-label">${escapeHtml(feat.label)}</span>
        <span class="feat-grid-idx">#${feat.index}</span>
      </div>
      <div class="feat-grid-stats">
        <span class="conf-badge ${feat.confidence}">${feat.confidence}</span>
        <span class="feat-stat">
          freq <span class="feat-stat-val">${feat.frequency.toFixed(1)}%</span>
        </span>
        <span class="feat-stat">
          max <span class="feat-stat-val">${feat.max_activation.toFixed(1)}</span>
        </span>
      </div>
      ${topToks ? `<div class="feat-grid-tokens">${topToks}</div>` : ''}
      ${vocabToks ? `<div class="feat-grid-tokens">${vocabToks}</div>` : ''}
    `;

    // Staggered entrance
    card.style.opacity = '0';
    card.style.transform = 'translateY(8px)';
    setTimeout(() => {
      card.style.transition = 'opacity 200ms, transform 200ms, background 150ms, border-color 150ms, transform 150ms';
      card.style.opacity = '1';
      card.style.transform = 'none';
    }, i * 20);

    card.addEventListener('click', () => openFeatureDrawer(feat.index));
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter') openFeatureDrawer(feat.index);
    });

    grid.appendChild(card);
  });
}

// ── Pagination ─────────────────────────────────────────────────────────────────
function renderPagination(total, page) {
  const el = document.getElementById('features-pagination');
  if (!el) return;

  const totalPages = Math.ceil(total / PAGE_SIZE);
  if (totalPages <= 1) { el.innerHTML = ''; return; }

  const pages = getPaginationPages(page, totalPages);
  el.innerHTML = '';

  // Prev
  const prevBtn = document.createElement('button');
  prevBtn.className = 'page-btn';
  prevBtn.textContent = '←';
  prevBtn.disabled = page === 0;
  prevBtn.setAttribute('aria-label', 'Previous page');
  prevBtn.addEventListener('click', () => { currentPage = page - 1; loadFeatures(); });
  el.appendChild(prevBtn);

  pages.forEach(p => {
    const btn = document.createElement('button');
    btn.className = `page-btn ${p === page ? 'active' : ''}`;
    btn.textContent = p === '…' ? '…' : p + 1;
    btn.disabled = p === '…';
    if (p !== '…') {
      btn.setAttribute('aria-label', `Page ${p + 1}`);
      btn.addEventListener('click', () => { currentPage = p; loadFeatures(); });
    }
    el.appendChild(btn);
  });

  // Next
  const nextBtn = document.createElement('button');
  nextBtn.className = 'page-btn';
  nextBtn.textContent = '→';
  nextBtn.disabled = page >= totalPages - 1;
  nextBtn.setAttribute('aria-label', 'Next page');
  nextBtn.addEventListener('click', () => { currentPage = page + 1; loadFeatures(); });
  el.appendChild(nextBtn);
}

function getPaginationPages(current, total) {
  const pages = [];
  if (total <= 7) {
    for (let i = 0; i < total; i++) pages.push(i);
    return pages;
  }
  pages.push(0);
  if (current > 2) pages.push('…');
  for (let i = Math.max(1, current - 1); i <= Math.min(total - 2, current + 1); i++) pages.push(i);
  if (current < total - 3) pages.push('…');
  pages.push(total - 1);
  return pages;
}

// ── Feature detail drawer ──────────────────────────────────────────────────────
async function openFeatureDrawer(idx) {
  const drawer    = document.getElementById('feat-drawer');
  const titleEl   = document.getElementById('feat-drawer-title');
  const indexEl   = document.getElementById('feat-drawer-index');
  const confEl    = document.getElementById('feat-drawer-conf');
  const bodyEl    = document.getElementById('feat-drawer-body');
  const panelEl   = drawer?.querySelector('.feat-drawer-panel');

  if (!drawer) return;

  titleEl.textContent = 'Loading…';
  indexEl.textContent = `#${idx}`;
  confEl.textContent  = '';
  bodyEl.innerHTML    = `<div style="color:var(--text-3);font-size:12px">Loading feature data…</div>`;
  drawer.hidden = false;
  panelEl?.focus();

  try {
    const feat = await api(`/api/feature/${idx}`);

    titleEl.textContent = feat.label;
    indexEl.textContent = `#${feat.index}`;
    confEl.className    = `feat-drawer-conf conf-badge ${feat.confidence}`;
    confEl.textContent  = feat.confidence;

    bodyEl.innerHTML = buildDrawerBody(feat);

  } catch (err) {
    bodyEl.innerHTML = `<p style="color:var(--text-2);font-size:12px">Error: ${escapeHtml(err.message)}</p>`;
  }
}

function buildDrawerBody(feat) {
  // Stats
  const stats = `
    <div class="feat-drawer-section">
      <h3>Statistics</h3>
      <div class="feat-stat-row">
        <div class="feat-stat-box">
          <span class="feat-stat-box-val">${feat.frequency.toFixed(2)}%</span>
          <span class="feat-stat-box-label">Frequency</span>
        </div>
        <div class="feat-stat-box">
          <span class="feat-stat-box-val">${feat.mean_activation.toFixed(3)}</span>
          <span class="feat-stat-box-label">Mean Act.</span>
        </div>
        <div class="feat-stat-box">
          <span class="feat-stat-box-val">${feat.max_activation.toFixed(3)}</span>
          <span class="feat-stat-box-label">Max Act.</span>
        </div>
      </div>
    </div>
  `;

  // Reasoning
  const reasoning = feat.reasoning ? `
    <div class="feat-drawer-section">
      <h3>Reasoning</h3>
      <p style="font-size:13px;color:var(--text-2);line-height:1.7">${escapeHtml(feat.reasoning)}</p>
    </div>
  ` : '';

  // Vocab projections
  const vocabPills = (feat.vocab_proj || []).map((tok, i) => {
    const logit = feat.vocab_proj_logits?.[i];
    return `<span class="vocab-chip">${escapeHtml(tok)}${logit !== undefined ? ` <span style="color:var(--text-3)">${logit.toFixed(2)}</span>` : ''}</span>`;
  }).join('');

  const vocab = vocabPills ? `
    <div class="feat-drawer-section">
      <h3>Vocab Projection (what this feature promotes)</h3>
      <div class="feat-vocab-chips">${vocabPills}</div>
    </div>
  ` : '';

  // Max activating examples
  const examples = (feat.examples || []).slice(0, 6).map(ex => {
    const ctx = ex.context
      ? ex.context.replace(/\[([^\]]+)\]/g, '<mark style="background:rgba(255,255,255,0.06);color:var(--text-1);border-radius:3px;padding:0 2px">$1</mark>')
      : '';
    return `
      <div class="feat-evidence-item">
        <div class="feat-evidence-token">"${escapeHtml(ex.token)}" <span style="color:var(--text-3);font-weight:400">act: ${ex.activation}</span></div>
        ${ctx ? `<div class="feat-evidence-ctx">${ctx}</div>` : ''}
        ${ex.source_id ? `<div style="font-size:9px;color:var(--text-3);margin-top:4px;font-family:var(--font-mono)">${escapeHtml(ex.source_id)}</div>` : ''}
      </div>
    `;
  }).join('');

  const evidence = examples ? `
    <div class="feat-drawer-section">
      <h3>Max-Activating Training Examples</h3>
      ${examples}
    </div>
  ` : '';

  return stats + reasoning + vocab + evidence;
}

function closeDrawer() {
  const drawer = document.getElementById('feat-drawer');
  if (drawer) drawer.hidden = true;
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
