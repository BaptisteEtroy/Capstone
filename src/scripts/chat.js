/**
 * chat.js — Chat panel logic
 * Handles message submission, display, conversation history, and wires up attribution.
 */

import { apiPost } from './app.js';
import { renderAttribution } from './attribution.js';

// ── State ──────────────────────────────────────────────────────────────────────
let history = [];     // [{role, content}]
let isLoading = false;

// ── DOM refs ───────────────────────────────────────────────────────────────────
let messagesEl, chatForm, inputEl, sendBtn, welcomeEl;

// ── Init ───────────────────────────────────────────────────────────────────────
export function initChat(appState) {
  messagesEl   = document.getElementById('messages');
  chatForm     = document.getElementById('chat-form');
  inputEl      = document.getElementById('chat-input');
  sendBtn      = document.getElementById('send-btn');
  welcomeEl    = document.getElementById('welcome');

  // Auto-resize textarea
  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
    sendBtn.disabled = !inputEl.value.trim();
  });

  // Submit on Enter (Shift+Enter = newline)
  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) chatForm.requestSubmit();
    }
  });

  chatForm.addEventListener('submit', e => {
    e.preventDefault();
    const message = inputEl.value.trim();
    if (!message || isLoading) return;
    sendMessage(message);
  });

  // Example chips
  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      inputEl.value = chip.dataset.prompt;
      inputEl.dispatchEvent(new Event('input'));
      inputEl.focus();
    });
  });

  // Attribution panel toggle
  const attrToggle = document.getElementById('attr-toggle');
  const chatLayout = document.querySelector('.chat-layout');
  attrToggle?.addEventListener('click', () => {
    const hidden = chatLayout.classList.toggle('attr-hidden');
    attrToggle.classList.toggle('active', !hidden);
    attrToggle.setAttribute('aria-pressed', String(!hidden));
  });

}

// ── Send message ───────────────────────────────────────────────────────────────
async function sendMessage(message) {
  if (isLoading) return;
  isLoading = true;

  // Hide welcome
  if (welcomeEl) welcomeEl.style.display = 'none';

  // Clear input
  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;
  sendBtn.classList.add('loading');

  // Append user message
  appendMessage('user', message);

  // Scroll to bottom
  scrollToBottom();

  // Append thinking indicator
  const thinkingId = appendThinking();

  // Push to history
  history.push({ role: 'user', content: message });

  try {
    const data = await apiPost('/api/chat', {
      message,
      history: history.slice(-6), // last 6 turns
    });

    // Remove thinking
    removeThinking(thinkingId);

    // Append assistant response
    appendMessage('assistant', data.response);

    // Update history
    history.push({ role: 'assistant', content: data.response });

    // Reveal attribution panel on first response
    const attrPanel = document.getElementById('attr-panel');
    const attrToggle = document.getElementById('attr-toggle');
    if (attrPanel) attrPanel.classList.remove('attr-panel-hidden');
    if (attrToggle) attrToggle.style.visibility = '';

    // Render attribution
    renderAttribution({
      inputFeatures: data.input_features || [],
      outputFeatures: data.output_features || [],
      responseTokens: data.response_tokens || [],
    });

  } catch (err) {
    removeThinking(thinkingId);
    appendError(err.message || 'Request failed. Is the model loaded?');
    history.pop(); // remove failed user message from history
  } finally {
    isLoading = false;
    sendBtn.classList.remove('loading');
    sendBtn.disabled = !inputEl.value.trim();
    scrollToBottom();
  }
}

// ── Message rendering ──────────────────────────────────────────────────────────
function appendMessage(role, content) {
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.setAttribute('role', 'listitem');

  el.innerHTML = `
    <div class="message-role ${role}-role">${role === 'user' ? 'You' : 'Assistant'}</div>
    <div class="message-body">${escapeHtml(content)}</div>
  `;

  messagesEl.appendChild(el);
  return el;
}

function appendThinking() {
  const id = `thinking-${Date.now()}`;
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.id = id;
  el.innerHTML = `
    <div class="message-role assistant-role">Assistant</div>
    <div class="thinking">
      <div class="thinking-dots">
        <span></span><span></span><span></span>
      </div>
      <span class="thinking-label">Thinking…</span>
    </div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
  return id;
}

function removeThinking(id) {
  document.getElementById(id)?.remove();
}

function appendError(msg) {
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.innerHTML = `
    <div class="message-role assistant-role">Error</div>
    <div class="message-body" style="border-color: rgba(255,255,255,0.1); color: var(--text-2);">${escapeHtml(msg)}</div>
  `;
  messagesEl.appendChild(el);
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\n/g, '<br>');
}


