/**
 * explore.js — Circuit Explorer panel
 * Tokenise input text, show activated features, render a D3 Sankey circuit.
 */

import { apiPost } from './app.js';

// ── Init ───────────────────────────────────────────────────────────────────────
export function initExplore(appState) {
  const textEl  = document.getElementById('explore-text');
  const runBtn  = document.getElementById('explore-run-btn');
  const results = document.getElementById('explore-results');

  runBtn?.addEventListener('click', async () => {
    const text = textEl?.value.trim();
    if (!text) { textEl?.focus(); return; }

    runBtn.disabled = true;
    runBtn.textContent = 'Analysing…';
    if (results) results.hidden = true;

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

  // Enter key
  textEl?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runBtn?.click();
    }
  });
}

// ── Render results ─────────────────────────────────────────────────────────────
function renderResults({ tokens, features, links }) {
  renderTokens(tokens, features, links);
  renderCircuit(tokens, features, links);
  renderFeatureTable(features);
}

// ── Token display ──────────────────────────────────────────────────────────────
function renderTokens(tokens, features, links) {
  const el = document.getElementById('explore-tokens');
  if (!el) return;
  el.innerHTML = '';

  tokens.forEach((tok, i) => {
    const badge = document.createElement('span');
    badge.className = 'token-badge';
    badge.textContent = tok;
    badge.title = `Token ${i}`;

    // Highlight tokens that have strong links
    const strength = links.filter(l => l.source === i).reduce((s, l) => s + l.value, 0);
    if (strength > 1) {
      badge.style.borderColor = 'rgba(255,255,255,0.18)';
      badge.style.color = 'var(--text-1)';
    }

    badge.addEventListener('mouseenter', () => highlightLinksForToken(i, links));
    badge.addEventListener('mouseleave', () => resetHighlights());

    el.appendChild(badge);
  });
}

function highlightLinksForToken(tokenIdx, links) {
  const active = new Set(links.filter(l => l.source === tokenIdx).map(l => l.target));
  document.querySelectorAll('.explore-feat-row').forEach(row => {
    const featI = parseInt(row.dataset.featIdx);
    row.classList.toggle('highlighted', active.has(featI));
  });
}

function resetHighlights() {
  document.querySelectorAll('.explore-feat-row').forEach(r => r.classList.remove('highlighted'));
}

// ── Circuit SVG ────────────────────────────────────────────────────────────────
function renderCircuit(tokens, features, links) {
  const container = document.getElementById('explore-circuit');
  if (!container) return;
  container.innerHTML = '';

  if (!features.length) return;

  const W     = container.offsetWidth || 600;
  const NODE_H = 32;
  const GAP    = 6;
  const COL_W  = 130;

  const topFeatures = features.slice(0, 15);
  const topTokens   = tokens.slice(0, 20);

  const totalRows = Math.max(topTokens.length, topFeatures.length);
  const H = totalRows * (NODE_H + GAP) + 50;

  const inputX   = 0;
  const outputX  = W - COL_W;
  const midX     = W / 2;

  const svg = d3.create('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('width', W)
    .attr('height', H);

  // Headers
  const hStyle = 'font-family:IBM Plex Mono,monospace;font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;';
  const headerY = 12;

  svg.append('text').attr('x', inputX + COL_W / 2).attr('y', headerY)
    .attr('text-anchor', 'middle').attr('fill', 'var(--text-3)').attr('style', hStyle).text('TOKENS');

  svg.append('text').attr('x', outputX + COL_W / 2).attr('y', headerY)
    .attr('text-anchor', 'middle').attr('fill', 'var(--text-3)').attr('style', hStyle).text('FEATURES');

  const startY = 24;

  // Token y centres
  const tokenY = topTokens.map((_, i) => startY + i * (NODE_H + GAP) + NODE_H / 2);
  const featY  = topFeatures.map((_, i) => startY + i * (NODE_H + GAP) + NODE_H / 2);

  // Find max activation for scaling
  const maxAct = Math.max(...topFeatures.map(f => f.activation), 1);

  // Draw links
  const validLinks = links.filter(l => l.source < topTokens.length && l.target < topFeatures.length);
  const maxVal = Math.max(...validLinks.map(l => l.value), 1);

  validLinks.forEach(link => {
    const x1 = inputX + COL_W + 8;
    const x2 = outputX - 8;
    const y1 = tokenY[link.source];
    const y2 = featY[link.target];
    const opacity = 0.08 + (link.value / maxVal) * 0.5;
    const w = 0.5 + (link.value / maxVal) * 2;

    svg.append('path')
      .attr('d', `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`)
      .attr('fill', 'none')
      .attr('stroke', '#6e6e78')
      .attr('stroke-width', w)
      .attr('stroke-opacity', 0)
      .transition().delay((link.source + link.target) * 20).duration(200)
      .attr('stroke-opacity', opacity);
  });

  // Draw token nodes
  topTokens.forEach((tok, i) => {
    const y = startY + i * (NODE_H + GAP);
    const g = svg.append('g').attr('transform', `translate(${inputX},${y})`);

    g.append('rect')
      .attr('width', COL_W).attr('height', NODE_H).attr('rx', 4)
      .attr('fill', 'var(--bg-3)').attr('stroke', 'var(--border-default)').attr('stroke-width', 0.6);

    g.append('text')
      .attr('x', 8).attr('y', NODE_H / 2 + 4)
      .attr('fill', 'var(--text-2)')
      .attr('font-family', 'IBM Plex Mono, monospace')
      .attr('font-size', 10)
      .text(tok.length > 14 ? tok.slice(0, 14) + '…' : tok);
  });

  // Draw feature nodes
  topFeatures.forEach((feat, i) => {
    const y = startY + i * (NODE_H + GAP);
    const actRatio = feat.activation / maxAct;
    const confColor = CONF_COLORS[feat.confidence] || CONF_COLORS.unknown;

    const g = svg.append('g').attr('transform', `translate(${outputX},${y})`);

    g.append('rect')
      .attr('width', COL_W).attr('height', NODE_H).attr('rx', 4)
      .attr('fill', 'var(--bg-3)')
      .attr('stroke', confColor)
      .attr('stroke-width', 0.6)
      .attr('stroke-opacity', 0.4);

    // Activation bar
    g.append('rect')
      .attr('x', 4).attr('y', NODE_H - 3)
      .attr('width', (COL_W - 8) * actRatio).attr('height', 2).attr('rx', 1)
      .attr('fill', confColor).attr('opacity', 0.6);

    g.append('text')
      .attr('x', 6).attr('y', NODE_H / 2 + 4)
      .attr('fill', 'var(--text-1)')
      .attr('font-family', 'IBM Plex Mono, monospace')
      .attr('font-size', 9.5)
      .text(feat.label.length > 13 ? feat.label.slice(0, 13) + '…' : feat.label);
  });

  container.appendChild(svg.node());
}

// ── Feature table ──────────────────────────────────────────────────────────────
function renderFeatureTable(features) {
  const container = document.getElementById('explore-feature-table');
  if (!container) return;

  if (!features.length) {
    container.innerHTML = '<p style="padding:16px;font-size:12px;color:var(--text-3)">No features activated above threshold.</p>';
    return;
  }

  const rows = features.slice(0, 20).map((feat, i) => `
    <tr class="explore-feat-row" data-feat-idx="${i}">
      <td class="mono-cell">${i + 1}</td>
      <td class="mono-cell" style="color:var(--text-3)">#${feat.index}</td>
      <td class="feat-label-cell">${escapeHtml(feat.label)}</td>
      <td><span class="conf-badge ${feat.confidence}">${feat.confidence}</span></td>
      <td class="mono-cell" style="color:var(--text-1)">${feat.activation.toFixed(3)}</td>
      <td class="mono-cell" style="color:var(--text-3)">${(feat.vocab_proj || []).slice(0,3).join(' · ')}</td>
    </tr>
  `).join('');

  container.innerHTML = `
    <table class="explore-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Idx</th>
          <th>Label</th>
          <th>Conf</th>
          <th>Activation</th>
          <th>Promotes</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
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
