/**
 * attribution.js — Feature attribution visualisation
 *
 * Renders two views:
 *  1. Graph view — SVG Sankey-style with animated bezier paths (D3 for paths)
 *  2. List view  — Feature cards with activation bars
 *
 * Data shape:
 *   { inputFeatures: [...], outputFeatures: [...], responseTokens: [...] }
 *
 * Feature shape:
 *   { index, label, confidence, activation, evidence, vocab_proj }
 */

// ── Monochrome weight map (confidence → opacity/brightness) ───────────────────
const CONF_OPACITY = { high: 1.0, medium: 0.65, low: 0.4, unknown: 0.3 };
const NODE_COLOR   = '#32323a';   // bg-4
const NODE_STROKE  = '#3e3e48';   // border-3
const PATH_COLOR   = '#888896';   // text-2

// ── Public API ─────────────────────────────────────────────────────────────────
export function renderAttribution({ inputFeatures, outputFeatures, responseTokens }) {
  const emptyEl    = document.getElementById('attr-empty');
  const graphView  = document.getElementById('attr-graph-view');
  const listView   = document.getElementById('attr-list-view');

  const hasData = inputFeatures.length > 0 || outputFeatures.length > 0;

  if (emptyEl)   emptyEl.style.display   = hasData ? 'none' : '';
  if (graphView) graphView.hidden         = !hasData;
  if (listView)  listView.hidden          = !hasData;

  if (!hasData) return;

  renderGraph({ inputFeatures, outputFeatures, responseTokens });
  renderList({ inputFeatures, outputFeatures });
}

// ── Graph view ─────────────────────────────────────────────────────────────────
function renderGraph({ inputFeatures, outputFeatures }) {
  const container = document.getElementById('attr-graph');
  if (!container) return;
  container.innerHTML = '';

  const panelWidth = container.parentElement?.offsetWidth || 320;
  const W = panelWidth - 32; // minus padding

  // Layout constants
  const FEAT_H   = 38;
  const GAP      = 8;
  const COL_W    = 120;
  const CONN_GAP = 16;

  const inputCount  = Math.min(inputFeatures.length, 8);
  const outputCount = Math.min(outputFeatures.length, 8);
  const rows        = Math.max(inputCount, outputCount);

  const TOTAL_H = rows * (FEAT_H + GAP) + 48; // +header

  // Column x positions
  const inputX  = 0;
  const midX    = W / 2;
  const outputX = W - COL_W;

  const svg = d3.create('svg')
    .attr('viewBox', `0 0 ${W} ${TOTAL_H}`)
    .attr('width', W)
    .attr('height', TOTAL_H)
    .attr('role', 'img')
    .attr('aria-label', 'Feature attribution graph');

  // ── Column headers ───────────────────────────────────────────────────────────
  const headerY = 14;
  const headerStyle = 'font-family: IBM Plex Mono, monospace; font-size: 9px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;';

  svg.append('text')
    .attr('x', inputX + COL_W / 2)
    .attr('y', headerY)
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--text-3)')
    .attr('style', headerStyle)
    .text('INPUT');

  svg.append('text')
    .attr('x', outputX + COL_W / 2)
    .attr('y', headerY)
    .attr('text-anchor', 'middle')
    .attr('fill', 'var(--text-3)')
    .attr('style', headerStyle)
    .text('OUTPUT');

  const startY = 30;

  // ── Compute node centres ─────────────────────────────────────────────────────
  const inputCentres  = inputFeatures.slice(0, inputCount).map((_, i) =>
    startY + i * (FEAT_H + GAP) + FEAT_H / 2
  );
  const outputCentres = outputFeatures.slice(0, outputCount).map((_, i) =>
    startY + i * (FEAT_H + GAP) + FEAT_H / 2
  );

  // ── Draw connections ─────────────────────────────────────────────────────────
  const maxAct = Math.max(
    ...inputFeatures.map(f => f.activation),
    ...outputFeatures.map(f => f.activation),
    1,
  );

  // ── Draw connections — always connect top features, weight by activation ──────
  // Connect each input to each output; opacity = geometric mean of ranks
  inputFeatures.slice(0, inputCount).forEach((inFeat, ii) => {
    outputFeatures.slice(0, outputCount).forEach((outFeat, oi) => {
      // Decay by rank position so top-ranked features have most prominent paths
      const inRank  = 1 - ii / inputCount;
      const outRank = 1 - oi / outputCount;
      const baseOpacity = Math.sqrt(inRank * outRank) * 0.22;
      const w = Math.max(0.4, Math.sqrt(inRank * outRank) * 1.2);

      const x1 = inputX + COL_W + CONN_GAP;
      const x2 = outputX - CONN_GAP;
      const y1 = inputCentres[ii];
      const y2 = outputCentres[oi];

      const path = svg.append('path')
        .attr('class', 'attr-path')
        .attr('data-input', ii)
        .attr('data-output', oi)
        .attr('data-base-opacity', baseOpacity)
        .attr('d', `M${x1},${y1} C${midX},${y1} ${midX},${y2} ${x2},${y2}`)
        .attr('fill', 'none')
        .attr('stroke', PATH_COLOR)
        .attr('stroke-width', w)
        .attr('stroke-opacity', 0);

      // Staggered entrance
      setTimeout(() => {
        path.transition().duration(300).attr('stroke-opacity', baseOpacity);
      }, (ii + oi) * 35);
    });
  });

  // ── Draw feature nodes ───────────────────────────────────────────────────────
  function drawFeatureNodes(features, colX, side) {
    const count = side === 'input' ? inputCount : outputCount;
    features.slice(0, count).forEach((feat, i) => {
      const y = startY + i * (FEAT_H + GAP);
      const confOpacity = CONF_OPACITY[feat.confidence] ?? 0.3;
      const actRatio = feat.activation / maxAct;

      const g = svg.append('g')
        .attr('class', `attr-node attr-node-${side}`)
        .attr('data-idx', i)
        .attr('transform', `translate(${colX}, ${y})`)
        .attr('cursor', 'pointer')
        .attr('aria-label', `Feature ${feat.index}: ${feat.label}`);

      // Background rect — flat, no colour
      g.append('rect')
        .attr('width', COL_W)
        .attr('height', FEAT_H)
        .attr('rx', 3)
        .attr('fill', NODE_COLOR)
        .attr('stroke', NODE_STROKE)
        .attr('stroke-width', 0.6);

      // Activation bar (bottom edge)
      if (actRatio > 0) {
        g.append('rect')
          .attr('x', 0)
          .attr('y', FEAT_H - 1)
          .attr('width', COL_W * actRatio)
          .attr('height', 1)
          .attr('fill', '#6e6e78')
          .attr('opacity', confOpacity);
      }

      // Label
      const maxChars = 13;
      const labelText = feat.label.length > maxChars
        ? feat.label.slice(0, maxChars) + '…' : feat.label;

      g.append('text')
        .attr('x', 7).attr('y', 14)
        .attr('fill', '#c8c8cc')
        .attr('font-family', 'IBM Plex Mono, monospace')
        .attr('font-size', 9.5)
        .attr('font-weight', 400)
        .text(labelText);

      // Activation value
      g.append('text')
        .attr('x', COL_W - 6).attr('y', 14)
        .attr('text-anchor', 'end')
        .attr('fill', '#4a4a52')
        .attr('font-family', 'IBM Plex Mono, monospace')
        .attr('font-size', 8.5)
        .text(feat.activation.toFixed(2));

      // Index
      g.append('text')
        .attr('x', 7).attr('y', 26)
        .attr('fill', '#3a3a40')
        .attr('font-family', 'IBM Plex Mono, monospace')
        .attr('font-size', 8)
        .text(`#${feat.index}`);

      // Hover
      g.node().addEventListener('mouseenter', () => highlightConnections(svg, i, side));
      g.node().addEventListener('mouseleave', () => resetConnections(svg));

      // Entrance
      g.attr('opacity', 0);
      setTimeout(() => g.transition().duration(200).attr('opacity', 1),
        i * 50 + (side === 'output' ? 100 : 0));
    });
  }

  drawFeatureNodes(inputFeatures, inputX, 'input');
  drawFeatureNodes(outputFeatures, outputX, 'output');
  container.appendChild(svg.node());
}

function highlightConnections(svg, nodeIdx, side) {
  svg.selectAll('.attr-path').each(function() {
    const el = d3.select(this);
    const ii = +el.attr('data-input');
    const oi = +el.attr('data-output');
    const isConnected = side === 'input' ? ii === nodeIdx : oi === nodeIdx;
    el.transition().duration(100)
      .attr('stroke-opacity', isConnected ? 0.75 : 0.03)
      .attr('stroke-width',   isConnected ? 1.5  : 0.4);
  });
}

function resetConnections(svg) {
  svg.selectAll('.attr-path').each(function() {
    const el = d3.select(this);
    const base = +el.attr('data-base-opacity') || 0.15;
    el.transition().duration(150).attr('stroke-opacity', base);
  });
}

// ── List view ──────────────────────────────────────────────────────────────────
function renderList({ inputFeatures, outputFeatures }) {
  renderFeatureList('input-features', inputFeatures, 'input');
  renderFeatureList('output-features', outputFeatures, 'output');

  const inputCount  = document.getElementById('input-count');
  const outputCount = document.getElementById('output-count');
  if (inputCount)  inputCount.textContent  = inputFeatures.length;
  if (outputCount) outputCount.textContent = outputFeatures.length;
}

function renderFeatureList(containerId, features, side) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  if (!features.length) {
    container.innerHTML = '<p style="font-size:11px;color:var(--text-3);padding:8px 4px;">No features detected</p>';
    return;
  }

  const maxAct = Math.max(...features.map(f => f.activation), 1);

  features.slice(0, 8).forEach((feat, i) => {
    const card = document.createElement('div');
    card.className = 'feature-card';
    card.setAttribute('role', 'listitem');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', `Feature ${feat.index}: ${feat.label}`);

    const actPct = Math.round((feat.activation / maxAct) * 100);
    const evidencePills = (feat.evidence || []).slice(0, 4)
      .map(t => `<span class="token-pill evidence">${escapeHtml(t)}</span>`)
      .join('');

    card.innerHTML = `
      <div class="feature-card-top">
        <span class="feature-card-label">${escapeHtml(feat.label)}</span>
        <span class="feature-card-idx">#${feat.index}</span>
      </div>
      <div class="feature-card-meta">
        <span class="conf-badge ${feat.confidence || 'unknown'}">${feat.confidence || '?'}</span>
        <div class="activation-bar-wrap">
          <div class="activation-bar" style="width:${actPct}%"></div>
        </div>
        <span class="activation-val">${feat.activation.toFixed(2)}</span>
      </div>
      ${evidencePills ? `<div class="feature-card-tokens">${evidencePills}</div>` : ''}
    `;

    // Animate in
    card.style.opacity = '0';
    card.style.transform = 'translateY(6px)';
    setTimeout(() => {
      card.style.transition = 'opacity 200ms, transform 200ms';
      card.style.opacity = '1';
      card.style.transform = 'none';
    }, i * 50 + (side === 'output' ? 100 : 0));

    container.appendChild(card);
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
