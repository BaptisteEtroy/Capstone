/**
 * attribution.js — Feature attribution visualisation
 *
 * Exports:
 *  - renderInlineAttribution(messageEl, { inputFeatures, outputFeatures })
 *    Renders compact feature chips inline to the right of an assistant message.
 *
 * Feature shape:
 *   { index, label, confidence, activation, evidence, vocab_proj }
 */

// ── Inline attribution (chat sidebar per message) ──────────────────────────────
export function renderInlineAttribution(messageEl, { inputFeatures, outputFeatures }) {
  const attrEl = messageEl.querySelector('.message-attr');
  if (!attrEl) return;

  const hasInput  = inputFeatures.length  > 0;
  const hasOutput = outputFeatures.length > 0;
  if (!hasInput && !hasOutput) return;

  const content = document.createElement('div');
  content.className = 'message-attr-content';

  if (hasInput) {
    content.appendChild(buildSection('Input', inputFeatures.slice(0, 5)));
  }
  if (hasOutput) {
    content.appendChild(buildSection('Output', outputFeatures.slice(0, 5)));
  }

  attrEl.appendChild(content);
}

function buildSection(label, features) {
  const maxAct = Math.max(...features.map(f => f.activation), 1);

  const section = document.createElement('div');
  section.className = 'attr-inline-section';

  const heading = document.createElement('div');
  heading.className = 'attr-inline-label';
  heading.textContent = label;
  section.appendChild(heading);

  const chips = document.createElement('div');
  chips.className = 'attr-inline-chips';

  features.forEach((feat, i) => {
    const actPct = Math.round((feat.activation / maxAct) * 100);
    const maxChars = 22;
    const labelText = feat.label.length > maxChars
      ? feat.label.slice(0, maxChars) + '…'
      : feat.label;

    const chip = document.createElement('div');
    chip.className = 'attr-chip';
    chip.title = `${feat.label} (#${feat.index}) — activation: ${feat.activation.toFixed(3)}`;
    chip.innerHTML = `
      <div class="attr-chip-top">
        <span class="attr-chip-label">${escapeHtml(labelText)}</span>
        <span class="attr-chip-val">${feat.activation.toFixed(2)}</span>
      </div>
      <div class="attr-chip-bar">
        <div class="attr-chip-bar-fill" style="width:0%"></div>
      </div>
    `;

    // Animate bar in with stagger
    setTimeout(() => {
      const fill = chip.querySelector('.attr-chip-bar-fill');
      if (fill) fill.style.width = `${actPct}%`;
    }, i * 40 + 100);

    chips.appendChild(chip);
  });

  section.appendChild(chips);
  return section;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Legacy graph renderer (used by explore.js if needed) ──────────────────────
const NODE_COLOR = '#32323a';
const NODE_STROKE = '#3e3e48';
const PATH_COLOR  = '#888896';

export function renderAttribution({ inputFeatures, outputFeatures, responseTokens }) {
  // No-op: attribution is now rendered inline per-message via renderInlineAttribution
}
