/**
 * steering.js — Feature Steering panel
 * Search features, set strengths, run steered + baseline generation.
 */

import { api, apiPost } from './app.js';

// ── State ──────────────────────────────────────────────────────────────────────
let activeFeatures = new Map();   // idx → strength
let searchDebounce = null;

// ── Init ───────────────────────────────────────────────────────────────────────
export function initSteering(appState) {
  const promptEl    = document.getElementById('steer-prompt');
  const searchEl    = document.getElementById('steer-search');
  const resultsEl   = document.getElementById('steer-search-results');
  const runBtn      = document.getElementById('steer-run-btn');
  const tokensEl    = document.getElementById('steer-tokens');
  const tokensValEl = document.getElementById('steer-tokens-val');

  // Tokens slider
  tokensEl?.addEventListener('input', () => {
    if (tokensValEl) tokensValEl.textContent = tokensEl.value;
  });

  // Search features
  searchEl?.addEventListener('input', () => {
    clearTimeout(searchDebounce);
    const q = searchEl.value.trim();
    if (!q) {
      hideResults();
      return;
    }
    searchDebounce = setTimeout(() => searchFeatures(q, resultsEl), 300);
  });

  // Close results on outside click
  document.addEventListener('click', e => {
    if (!searchEl?.contains(e.target) && !resultsEl?.contains(e.target)) {
      hideResults();
    }
  });

  // Run
  runBtn?.addEventListener('click', async () => {
    const prompt = promptEl?.value.trim();
    if (!prompt) {
      promptEl?.focus();
      return;
    }

    runBtn.disabled = true;
    runBtn.textContent = 'Generating…';

    const baselineEl = document.getElementById('steer-baseline');
    const steeredEl  = document.getElementById('steer-steered');
    const badgeEl    = document.getElementById('steer-badge');

    if (baselineEl) baselineEl.textContent = '…';
    if (steeredEl)  steeredEl.textContent  = '…';

    try {
      const featureMap = {};
      activeFeatures.forEach((strength, idx) => { featureMap[idx] = strength; });

      const data = await apiPost('/api/steer', {
        prompt,
        features: featureMap,
        max_tokens: parseInt(tokensEl?.value || '150', 10),
      });

      if (baselineEl) baselineEl.textContent = stripPrompt(data.baseline, prompt);
      if (steeredEl)  steeredEl.textContent  = stripPrompt(data.steered, prompt);

      if (badgeEl && data.applied_features?.length) {
        badgeEl.textContent = data.applied_features.map(f => f.label).join(' · ');
        badgeEl.hidden = false;
      }

    } catch (err) {
      if (baselineEl) baselineEl.textContent = `Error: ${err.message}`;
      if (steeredEl)  steeredEl.textContent  = `Error: ${err.message}`;
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = `Generate
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <polygon points="4 3 16 10 4 17"/>
        </svg>`;
    }
  });
}

// ── Feature search ─────────────────────────────────────────────────────────────
async function searchFeatures(query, resultsEl) {
  if (!resultsEl) return;
  try {
    const data = await api(`/api/features?search=${encodeURIComponent(query)}&limit=10`);
    renderResults(data.features, resultsEl);
  } catch {
    hideResults();
  }
}

function renderResults(features, resultsEl) {
  if (!resultsEl) return;
  resultsEl.innerHTML = '';

  if (!features.length) {
    resultsEl.innerHTML = '<div style="padding:10px 14px;font-size:11px;color:var(--text-3)">No features found</div>';
    resultsEl.hidden = false;
    return;
  }

  features.forEach(feat => {
    const item = document.createElement('div');
    item.className = 'search-result-item';
    item.setAttribute('role', 'option');
    item.setAttribute('tabindex', '0');
    item.innerHTML = `
      <div class="search-result-label">${escapeHtml(feat.label)}</div>
      <div class="search-result-idx">#${feat.index}</div>
      <span class="conf-badge ${feat.confidence}">${feat.confidence}</span>
    `;

    item.addEventListener('click', () => addFeature(feat));
    item.addEventListener('keydown', e => { if (e.key === 'Enter') addFeature(feat); });
    resultsEl.appendChild(item);
  });

  resultsEl.hidden = false;
}

function hideResults() {
  const el = document.getElementById('steer-search-results');
  if (el) el.hidden = true;
}

// ── Active feature management ──────────────────────────────────────────────────
function addFeature(feat) {
  hideResults();
  const searchEl = document.getElementById('steer-search');
  if (searchEl) searchEl.value = '';

  if (activeFeatures.has(feat.index)) return; // already added
  activeFeatures.set(feat.index, 10);          // default strength

  renderActiveFeatures();
}

function removeFeature(idx) {
  activeFeatures.delete(idx);
  renderActiveFeatures();
}

function renderActiveFeatures() {
  const container = document.getElementById('steer-active-features');
  if (!container) return;
  container.innerHTML = '';

  if (!activeFeatures.size) {
    container.innerHTML = `
      <div class="steer-empty-hint">
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
          <line x1="10" y1="4" x2="10" y2="16"/><line x1="4" y1="10" x2="16" y2="10"/>
        </svg>
        Search features below to add
      </div>`;
    return;
  }

  activeFeatures.forEach((strength, idx) => {
    // Get label from DOM or use index
    const row = document.createElement('div');
    row.className = 'steer-feature-row';

    row.innerHTML = `
      <span class="steer-feature-label">Feature #${idx}</span>
      <input
        type="number"
        class="steer-strength-input"
        value="${strength}"
        min="-50"
        max="50"
        step="1"
        aria-label="Steering strength for feature ${idx}"
      />
      <button class="steer-remove-btn" aria-label="Remove feature ${idx}">
        <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="5" y1="5" x2="15" y2="15"/><line x1="15" y1="5" x2="5" y2="15"/>
        </svg>
      </button>
    `;

    row.querySelector('.steer-strength-input').addEventListener('input', e => {
      activeFeatures.set(idx, parseFloat(e.target.value) || 0);
    });

    row.querySelector('.steer-remove-btn').addEventListener('click', () => removeFeature(idx));

    // Fetch label async and update
    fetchFeatureLabel(idx).then(label => {
      const labelEl = row.querySelector('.steer-feature-label');
      if (labelEl) labelEl.textContent = label;
    });

    container.appendChild(row);
  });
}

async function fetchFeatureLabel(idx) {
  try {
    const data = await api(`/api/feature/${idx}`);
    return `${data.label} (#${idx})`;
  } catch {
    return `Feature #${idx}`;
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────
function stripPrompt(text, prompt) {
  // Remove the echoed prompt prefix from decoded output
  if (text.startsWith(prompt)) {
    return text.slice(prompt.length).trim();
  }
  // Sometimes the model prepends "User: ..." — strip it
  const idx = text.indexOf('Assistant:');
  if (idx !== -1) {
    return text.slice(idx + 10).trim();
  }
  return text.trim();
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
