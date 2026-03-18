/**
 * attribution.js — Feature attribution panels + token highlighting
 *
 * Exports:
 *   renderInputPanel(panelEl, { inputTokens, inputFeatures })
 *   renderOutputPanel(panelEl, { responseTokens, outputFeatures })
 *   renderHighlightedTokens(tokenStrs, features, palette) → HTML string
 */

// ── Colour palette (8 slots, one per feature rank) ─────────────────────────
export const PALETTE = [
  '#6ea8f5',  // blue
  '#5dcea0',  // teal
  '#f5a44c',  // amber
  '#e87878',  // coral
  '#b07ae8',  // purple
  '#e8c840',  // yellow
  '#52c8e8',  // cyan
  '#e880b4',  // pink
];

// Dataset source display names + CSS class key
const SOURCE_META = {
  medmcqa:    { label: 'MCQ',        cls: 'mcq'     },
  pubmed_qa:  { label: 'PubMed QA',  cls: 'qa'      },
  pubmed_abs: { label: 'Abstracts',  cls: 'abs'     },
};

// ── Token highlighting ──────────────────────────────────────────────────────
/**
 * Returns an HTML string where each token in tokenStrs is wrapped in a
 * coloured <span> if its index appears in any feature's token_indices.
 * The first feature that references a token index wins the colour.
 */
export function renderHighlightedTokens(tokenStrs, features, palette = PALETTE) {
  const colorMap = {};   // token_index → { color, label }
  features.forEach((feat, fi) => {
    const color = palette[fi % palette.length];
    (feat.token_indices || []).forEach(ti => {
      if (!(ti in colorMap)) colorMap[ti] = { color, label: feat.label };
    });
  });

  return tokenStrs.map((tok, i) => {
    if (!tok) return '';
    const escaped = escapeHtml(tok).replace(/\n/g, '<br>');
    if (colorMap[i]) {
      const c = colorMap[i].color;
      return `<span class="tok-hl" style="color:${c};border-bottom:1.5px solid ${c}55" title="${escapeHtml(colorMap[i].label)}">${escaped}</span>`;
    }
    return escaped;
  }).join('');
}

// ── Left panel: input features ──────────────────────────────────────────────
export function renderInputPanel(panelEl, { inputFeatures }) {
  if (!panelEl) return;
  if (!inputFeatures?.length) {
    panelEl.innerHTML = '<div class="attr-empty">No labeled features activated above threshold</div>';
    return;
  }

  panelEl.innerHTML = inputFeatures.map((feat, fi) => {
    const color = PALETTE[fi % PALETTE.length];
    const actPct = Math.min(100, Math.round((feat.activation / 1.5) * 100));
    const evidence = (feat.evidence || []).slice(0, 5)
      .map(t => `<span class="attr-tok-pill">${escapeHtml(t)}</span>`).join('');

    return `
      <div class="attr-feat-card" style="--fc:${color}">
        <div class="attr-feat-row">
          <div class="attr-feat-dot"></div>
          <div class="attr-feat-label">${escapeHtml(feat.label)}</div>
          <div class="attr-feat-act">${feat.activation.toFixed(2)}</div>
        </div>
        <div class="attr-feat-bar"><div class="attr-feat-fill" style="width:${actPct}%"></div></div>
        ${evidence ? `<div class="attr-feat-evidence">${evidence}</div>` : ''}
      </div>`;
  }).join('');
}

// ── Right panel: output features ─────────────────────────────────────────────
export function renderOutputPanel(panelEl, { outputFeatures }) {
  if (!panelEl) return;
  if (!outputFeatures?.length) {
    panelEl.innerHTML = '<div class="attr-empty">No labeled features activated above threshold</div>';
    return;
  }

  panelEl.innerHTML = outputFeatures.map((feat, fi) => {
    const color = PALETTE[fi % PALETTE.length];
    const actPct = Math.min(100, Math.round((feat.activation / 1.5) * 100));
    const id = `of-${feat.index}-${fi}`;

    // Source breakdown bar
    const sourceBreakdown = feat.source_breakdown || {};
    const total = Object.values(sourceBreakdown).reduce((a, b) => a + b, 0) || 1;
    const srcBar = Object.entries(sourceBreakdown).map(([src, n]) => {
      const pct = Math.round((n / total) * 100);
      const meta = SOURCE_META[src] || { label: src, cls: 'other' };
      return `<div class="src-seg src-${meta.cls}" style="width:${pct}%" title="${meta.label} · ${pct}%"></div>`;
    }).join('');

    // MaxAct examples (expandable)
    const examples = (feat.max_act_examples || []).map(ex => {
      const meta = SOURCE_META[ex.source] || { label: ex.source, cls: 'other' };
      // Context: highlight the [token] bracket pattern
      const ctx = ex.context
        ? escapeHtml(ex.context).replace(/\[([^\]]+)\]/g, '<mark class="ctx-mark">$1</mark>')
        : '';
      return `
        <div class="out-example">
          <div class="out-example-hd">
            <span class="out-example-tok">${escapeHtml(ex.token)}</span>
            <span class="src-badge src-badge-${meta.cls}">${meta.label}</span>
            <span class="out-example-act">${ex.activation.toFixed(2)}</span>
          </div>
          ${ctx ? `<div class="out-example-ctx">${ctx}</div>` : ''}
        </div>`;
    }).join('');

    return `
      <div class="out-feat-card" style="--fc:${color}">
        <button class="out-feat-hd" aria-expanded="false" data-target="${id}">
          <div class="attr-feat-dot"></div>
          <div class="attr-feat-label">${escapeHtml(feat.label)}</div>
          <div class="attr-feat-act">${feat.activation.toFixed(2)}</div>
          <svg class="out-feat-chevron" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
            <polyline points="2 3 5 7 8 3"/>
          </svg>
        </button>
        <div class="attr-feat-bar"><div class="attr-feat-fill" style="width:${actPct}%"></div></div>
        ${srcBar ? `<div class="src-bar">${srcBar}</div>` : ''}
        <div class="out-feat-examples" id="${id}" hidden>
          ${examples || '<div class="attr-empty">No examples available</div>'}
        </div>
      </div>`;
  }).join('');

  // Wire expand/collapse
  panelEl.querySelectorAll('.out-feat-hd').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.target);
      if (!target) return;
      const expanded = btn.getAttribute('aria-expanded') === 'true';
      btn.setAttribute('aria-expanded', String(!expanded));
      target.hidden = expanded;
    });
  });
}

// ── Shared util ────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
