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

  // Example chips
  document.querySelectorAll('.explore-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      if (textEl) textEl.value = chip.dataset.prompt;
      runBtn?.click();
    });
  });
}

// ── Render results ─────────────────────────────────────────────────────────────
function renderResults({ tokens, features, links }) {
  renderTokens(tokens, features, links);
  renderSankey(tokens, features, links);
  renderFeatureTable(features);
}

// ── Token display ──────────────────────────────────────────────────────────────
function renderTokens(tokens, _features, links) {
  const el = document.getElementById('explore-tokens');
  if (!el) return;
  el.innerHTML = '';

  tokens.forEach((tok, i) => {
    const badge = document.createElement('span');
    badge.className = 'token-badge';
    badge.textContent = tok;
    badge.title = `Token ${i}`;

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
    row.classList.toggle('highlighted', active.has(parseInt(row.dataset.featIdx)));
  });
}

function resetHighlights() {
  document.querySelectorAll('.explore-feat-row').forEach(r => r.classList.remove('highlighted'));
}

// ── Sankey diagram ─────────────────────────────────────────────────────────────
function renderSankey(tokens, features, links) {
  const container = document.getElementById('explore-circuit');
  if (!container) return;
  container.innerHTML = '';
  if (!features.length) return;

  const topTokens   = tokens.slice(0, 15);
  const topFeatures = features.slice(0, 15);
  const validLinks  = links.filter(l => l.source < topTokens.length && l.target < topFeatures.length);

  const W       = container.offsetWidth || 700;
  const NODE_W  = 148;
  const NODE_GAP = 5;
  const MIN_H   = 24;
  const PAD_T   = 16;

  // Total outgoing flow per token
  const tokenFlow = new Array(topTokens.length).fill(0);
  validLinks.forEach(l => { tokenFlow[l.source] += l.value; });

  // Feature height proportional to activation
  const maxAct = Math.max(...topFeatures.map(f => f.activation), 1);
  const featFlow = topFeatures.map(f => f.activation / maxAct);

  function buildHeights(values, count) {
    const total = values.reduce((a, b) => a + b, 0) || 1;
    const base = count * (MIN_H + NODE_GAP);
    return values.map(v => Math.max(MIN_H, (v / total) * base * 1.4));
  }

  const tokenH = buildHeights(tokenFlow, topTokens.length);
  const featH  = buildHeights(featFlow, topFeatures.length);

  const H = Math.max(
    PAD_T + tokenH.reduce((a, b) => a + b, 0) + NODE_GAP * (topTokens.length - 1) + PAD_T,
    PAD_T + featH.reduce((a, b) => a + b, 0) + NODE_GAP * (topFeatures.length - 1) + PAD_T
  );

  // Y positions
  const tokenY = [], featY = [];
  let y = PAD_T;
  tokenH.forEach(h => { tokenY.push(y); y += h + NODE_GAP; });
  y = PAD_T;
  featH.forEach(h => { featY.push(y); y += h + NODE_GAP; });

  const leftX  = 0;
  const rightX = W - NODE_W;

  const svg = d3.create('svg')
    .attr('width', W).attr('height', H)
    .attr('viewBox', `0 0 ${W} ${H}`);

  // Link stacking offsets within each node
  const tokOff  = new Array(topTokens.length).fill(0);
  const featOff = new Array(topFeatures.length).fill(0);

  // Sort links by value descending for cleaner layering
  [...validLinks].sort((a, b) => b.value - a.value).forEach(link => {
    const si = link.source, ti = link.target;
    const tokTotal  = tokenFlow[si] || 1;
    const featTotal = featFlow[ti] || 1;
    const lh_tok  = Math.max(1.5, (link.value / tokTotal)  * tokenH[si]);
    const lh_feat = Math.max(1.5, (link.value / featTotal) * featH[ti]);

    const x0 = leftX + NODE_W;
    const x1 = rightX;
    const y0 = tokenY[si] + tokOff[si];
    const y1 = featY[ti]  + featOff[ti];
    tokOff[si]  += lh_tok;
    featOff[ti] += lh_feat;

    const mx = x0 + (x1 - x0) * 0.55;
    const d = [
      `M ${x0} ${y0}`,
      `C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1}`,
      `L ${x1} ${y1 + lh_feat}`,
      `C ${mx} ${y1 + lh_feat}, ${mx} ${y0 + lh_tok}, ${x0} ${y0 + lh_tok}`,
      'Z',
    ].join(' ');

    svg.append('path')
      .attr('d', d)
      .attr('fill', 'rgba(152,152,180,0.18)')
      .attr('stroke', 'rgba(152,152,180,0.08)')
      .attr('stroke-width', 0.5)
      .attr('opacity', 0)
      .transition().duration(280).delay((si + ti) * 22)
      .attr('opacity', 1);
  });

  // Token nodes
  topTokens.forEach((tok, i) => {
    const g = svg.append('g');
    g.append('rect')
      .attr('x', leftX).attr('y', tokenY[i])
      .attr('width', NODE_W).attr('height', tokenH[i]).attr('rx', 3)
      .attr('fill', 'var(--bg-3)').attr('stroke', 'var(--border-2)').attr('stroke-width', 0.5);

    if (tokenH[i] >= 16) {
      g.append('text')
        .attr('x', leftX + 8).attr('y', tokenY[i] + tokenH[i] / 2 + 4)
        .attr('fill', 'var(--text-2)')
        .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 10)
        .text(tok.length > 18 ? tok.slice(0, 18) + '…' : tok);
    }
  });

  // Feature nodes
  topFeatures.forEach((feat, i) => {
    const g = svg.append('g');
    const confColor = CONF_COLORS[feat.confidence] || CONF_COLORS.unknown;
    g.append('rect')
      .attr('x', rightX).attr('y', featY[i])
      .attr('width', NODE_W).attr('height', featH[i]).attr('rx', 3)
      .attr('fill', 'var(--bg-3)').attr('stroke', confColor)
      .attr('stroke-width', 0.8).attr('stroke-opacity', 0.55);

    if (featH[i] >= 16) {
      const label = feat.label.length > 15 ? feat.label.slice(0, 15) + '…' : feat.label;
      g.append('text')
        .attr('x', rightX + 6).attr('y', featY[i] + featH[i] / 2 + 4)
        .attr('fill', 'var(--text-1)')
        .attr('font-family', 'IBM Plex Mono, monospace').attr('font-size', 9)
        .text(label);
    }
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
