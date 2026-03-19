/**
 * explore.js — Circuit Explorer panel
 * Tokenise input text, show activated features, render a D3 circuit.
 */

import { apiPost } from './app.js';

const SANKEY_FEATS = 8;   // features shown in diagram
const TABLE_FEATS  = 20;  // features shown in table

// ── Init ───────────────────────────────────────────────────────────────────────
export function initExplore(_appState) {
  const textEl  = document.getElementById('explore-text');
  const runBtn  = document.getElementById('explore-run-btn');
  const results = document.getElementById('explore-results');

  runBtn?.addEventListener('click', async () => {
    const text = textEl?.value.trim();
    if (!text) { textEl?.focus(); return; }

    runBtn.disabled = true;
    runBtn.textContent = 'Analysing…';
    if (results) results.hidden = true;
    clearDetail();

    try {
      const data = await apiPost('/api/circuit', { text });
      renderResults(data);
      if (results) results.hidden = false;
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
        <circle cx="9" cy="9" r="6"/><line x1="14" y1="14" x2="18" y2="18"/>
      </svg> Analyse`;
    }
  });

  textEl?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runBtn?.click(); }
  });

  document.querySelectorAll('.explore-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      if (textEl) textEl.value = chip.dataset.prompt;
      runBtn?.click();
    });
  });
}

// ── Render results ─────────────────────────────────────────────────────────────
function renderResults({ tokens, features, links }) {
  renderTokens(tokens, links);
  renderSankey(tokens, features, links);
  renderFeatureTable(features);
}

// ── Token display ──────────────────────────────────────────────────────────────
function renderTokens(tokens, links) {
  const el = document.getElementById('explore-tokens');
  if (!el) return;
  el.innerHTML = '';

  tokens.forEach((tok, i) => {
    const badge = document.createElement('span');
    badge.className = 'token-badge';
    badge.textContent = tok;
    badge.title = `Token ${i}`;
    badge.style.cursor = 'pointer';

    const strength = links.filter(l => l.source === i).reduce((s, l) => s + l.value, 0);
    if (strength > 1) {
      badge.style.borderColor = 'rgba(255,255,255,0.18)';
      badge.style.color = 'var(--text-1)';
    }

    badge.addEventListener('click', () => highlightSVGLinks('source', i));
    badge.addEventListener('mouseenter', () => highlightLinksForToken(i, links));
    badge.addEventListener('mouseleave', () => resetHighlights());
    el.appendChild(badge);
  });
}

function highlightLinksForToken(tokenIdx, links) {
  const active = new Set(links.filter(l => l.source === tokenIdx).map(l => l.target));
  document.querySelectorAll('.explore-feat-row').forEach(row => {
    row.classList.toggle('highlighted', active.has(parseInt(row.dataset.featIdx)));
  });
}

function resetHighlights() {
  document.querySelectorAll('.explore-feat-row').forEach(r => r.classList.remove('highlighted'));
}

// ── SVG link highlighting ───────────────────────────────────────────────────────
function highlightSVGLinks(type, idx) {
  const paths = document.querySelectorAll('#explore-circuit .circuit-link');
  paths.forEach(path => {
    const src = parseInt(path.dataset.src);
    const tgt = parseInt(path.dataset.tgt);
    const match = (type === 'source' && src === idx) || (type === 'target' && tgt === idx);
    path.style.stroke = match ? 'rgba(180,180,220,0.9)' : 'rgba(152,152,180,0.05)';
    path.style.strokeWidth = match ? '2' : '0.5';
  });
}

function resetSVGHighlights() {
  const paths = document.querySelectorAll('#explore-circuit .circuit-link');
  paths.forEach(path => {
    path.style.stroke = '';
    path.style.strokeWidth = '';
  });
}

// ── Sankey diagram ─────────────────────────────────────────────────────────────
function renderSankey(tokens, features, links) {
  const container = document.getElementById('explore-circuit');
  if (!container) return;
  container.innerHTML = '';
  if (!features.length) return;

  const topFeatures = features.slice(0, SANKEY_FEATS);
  // All links pointing to displayed features, from any token
  const visibleLinks = links.filter(l =>
    l.source < tokens.length && l.target < topFeatures.length
  );

  const W           = container.offsetWidth || 700;
  const TOK_NODE_W  = 160;
  const TOK_H       = 24;
  const FEAT_H      = 26;
  const GAP         = 5;
  const PAD         = 14;
  // Measure feature label widths: IBM Plex Mono 9px ≈ 5.5px/char
  const CHAR_W      = 5.5;
  const FEAT_TEXT_PAD = 20;  // left(7) + right(13) padding inside node
  const maxLabelPx  = Math.max(...topFeatures.map(f => f.label.length * CHAR_W));
  const FEAT_NODE_W = Math.max(140, Math.ceil(maxLabelPx) + FEAT_TEXT_PAD);

  const numTok  = tokens.length;
  const numFeat = topFeatures.length;

  const H = Math.max(
    PAD + numTok  * (TOK_H  + GAP) - GAP + PAD,
    PAD + numFeat * (FEAT_H + GAP) - GAP + PAD
  );

  const tokenY = tokens.map((_, i) => PAD + i * (TOK_H  + GAP));
  const featY  = topFeatures.map((_, i) => PAD + i * (FEAT_H + GAP));

  const leftX  = 0;
  const rightX = W - FEAT_NODE_W;
  const maxVal = Math.max(...visibleLinks.map(l => l.value), 1);

  const svg = d3.create('svg')
    .attr('width', W).attr('height', H)
    .attr('viewBox', `0 0 ${W} ${H}`);

  // Background click resets highlights
  svg.on('click', () => resetSVGHighlights());

  // Links — thin bezier curves, width scaled by value
  visibleLinks.forEach((link, idx) => {
    const si = link.source, ti = link.target;
    const x0 = leftX + TOK_NODE_W, x1 = rightX;
    const y0 = tokenY[si] + TOK_H  / 2;
    const y1 = featY[ti]  + FEAT_H / 2;
    const mx = x0 + (x1 - x0) * 0.5;
    const sw = Math.max(0.8, (link.value / maxVal) * 2.5);

    svg.append('path')
      .attr('class', 'circuit-link')
      .attr('data-src', si)
      .attr('data-tgt', ti)
      .attr('d', `M ${x0} ${y0} C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1}`)
      .attr('fill', 'none')
      .attr('stroke', 'rgba(152,152,180,0.30)')
      .attr('stroke-width', sw)
      .attr('opacity', 0)
      .transition().duration(280).delay(idx * 15)
      .attr('opacity', 1);
  });

  // Token nodes — all input tokens shown
  tokens.forEach((tok, i) => {
    const g = svg.append('g').attr('class', 'node-clickable');
    g.append('rect')
      .attr('x', leftX).attr('y', tokenY[i])
      .attr('width', TOK_NODE_W).attr('height', TOK_H).attr('rx', 3)
      .attr('fill', 'var(--bg-3)').attr('stroke', 'var(--border-2)').attr('stroke-width', 0.5);

    g.append('title').text(tok);

    g.append('text')
      .attr('x', leftX + 8).attr('y', tokenY[i] + TOK_H / 2 + 4)
      .attr('fill', 'var(--text-2)')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 10)
      .text(tok.length > 20 ? tok.slice(0, 20) + '…' : tok);

    g.on('click', (event) => {
      event.stopPropagation();
      highlightSVGLinks('source', i);
    });
  });

  // Feature nodes — 2-line labels so full text is visible
  topFeatures.forEach((feat, i) => {
    const g = svg.append('g').attr('class', 'node-clickable');
    const confColor = CONF_COLORS[feat.confidence] || CONF_COLORS.unknown;
    g.append('rect')
      .attr('x', rightX).attr('y', featY[i])
      .attr('width', FEAT_NODE_W).attr('height', FEAT_H).attr('rx', 3)
      .attr('fill', 'var(--bg-3)').attr('stroke', confColor)
      .attr('stroke-width', 0.8).attr('stroke-opacity', 0.55);

    g.append('title').text(feat.label);

    g.append('text')
      .attr('x', rightX + 7).attr('y', featY[i] + FEAT_H / 2 + 4)
      .attr('fill', 'var(--text-1)')
      .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 9)
      .text(feat.label);

    g.on('click', (event) => {
      event.stopPropagation();
      highlightSVGLinks('target', i);
    });
  });

  container.appendChild(svg.node());
}

// ── Detail panel ───────────────────────────────────────────────────────────────
function clearDetail() {
  const el = document.getElementById('explore-detail');
  if (el) { el.innerHTML = ''; el.hidden = true; }
}

// ── Feature table ──────────────────────────────────────────────────────────────
function renderFeatureTable(features) {
  const container = document.getElementById('explore-feature-table');
  if (!container) return;

  if (!features.length) {
    container.innerHTML = '<p style="padding:16px;font-size:12px;color:var(--text-3)">No features activated above threshold.</p>';
    return;
  }

  const rows = features.slice(0, TABLE_FEATS).map((feat, i) => `
    <tr class="explore-feat-row" data-feat-idx="${i}" style="cursor:pointer">
      <td class="mono-cell">${i + 1}</td>
      <td class="mono-cell" style="color:var(--text-3)">#${feat.index}</td>
      <td class="feat-label-cell">${escapeHtml(feat.label)}</td>
      <td><span class="conf-badge ${feat.confidence}">${feat.confidence}</span></td>
      <td class="mono-cell" style="color:var(--text-1)">${feat.activation.toFixed(3)}</td>
    </tr>
  `).join('');

  container.innerHTML = `
    <table class="explore-table">
      <thead>
        <tr><th>#</th><th>Idx</th><th>Label</th><th>Conf</th><th>Activation</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  // Click table row to highlight SVG links for that feature
  container.querySelectorAll('.explore-feat-row').forEach((row, i) => {
    row.addEventListener('click', () => highlightSVGLinks('target', i));
  });
}

// ── Constants ──────────────────────────────────────────────────────────────────
const CONF_COLORS = {
  high:    '#9898a8',
  medium:  '#6e6e78',
  low:     '#4a4a52',
  unknown: '#38383e',
};

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
