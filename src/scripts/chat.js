/**
 * chat.js — Chat panel logic
 * Handles message submission, display, conversation history, and wires up attribution.
 */

import { apiPost } from './app.js';
import { renderAttribution } from './attribution.js';

// ── State ──────────────────────────────────────────────────────────────────────
let history = [];     // [{role, content}]
let sessions = [];    // [{id, title, history}]
let currentSessionId = null;
let isLoading = false;

// ── DOM refs ───────────────────────────────────────────────────────────────────
let messagesEl, chatForm, inputEl, sendBtn, welcomeEl, historyListEl;

// ── Init ───────────────────────────────────────────────────────────────────────
export function initChat(appState) {
  messagesEl   = document.getElementById('messages');
  chatForm     = document.getElementById('chat-form');
  inputEl      = document.getElementById('chat-input');
  sendBtn      = document.getElementById('send-btn');
  welcomeEl    = document.getElementById('welcome');
  historyListEl= document.getElementById('history-list');

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

  // New chat
  document.getElementById('new-chat-btn')?.addEventListener('click', startNewChat);

  // Attribution panel toggle
  const attrToggle = document.getElementById('attr-toggle');
  const chatLayout = document.querySelector('.chat-layout');
  attrToggle?.addEventListener('click', () => {
    const hidden = chatLayout.classList.toggle('attr-hidden');
    attrToggle.classList.toggle('active', !hidden);
    attrToggle.setAttribute('aria-pressed', String(!hidden));
  });

  // Attribution view tabs
  document.querySelectorAll('.attr-view-tab').forEach(tab => {
    tab.addEventListener('click', () => switchAttrView(tab.dataset.view));
  });

  loadSessions();
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

    // Update session
    saveSession(message, data.response);

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

// ── Attribution view tabs ──────────────────────────────────────────────────────
function switchAttrView(view) {
  document.querySelectorAll('.attr-view-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.view === view);
    t.setAttribute('aria-selected', String(t.dataset.view === view));
  });

  const graphView = document.getElementById('attr-graph-view');
  const listView  = document.getElementById('attr-list-view');

  if (view === 'graph') {
    graphView.hidden = false;
    listView.hidden  = true;
  } else {
    graphView.hidden = true;
    listView.hidden  = false;
  }
}

// ── Session management ─────────────────────────────────────────────────────────
function startNewChat() {
  history = [];
  currentSessionId = null;
  messagesEl.innerHTML = '';
  if (welcomeEl) {
    welcomeEl.style.display = '';
    messagesEl.appendChild(welcomeEl);
  }
  renderAttribution({ inputFeatures: [], outputFeatures: [], responseTokens: [] });
}

function saveSession(userMsg, assistantMsg) {
  if (!currentSessionId) {
    currentSessionId = `session-${Date.now()}`;
    const title = userMsg.length > 40 ? userMsg.slice(0, 40) + '…' : userMsg;
    sessions.unshift({ id: currentSessionId, title, history: [] });
    renderHistoryList();
  }
  const session = sessions.find(s => s.id === currentSessionId);
  if (session) {
    session.history = [...history];
    try {
      localStorage.setItem('sae_sessions', JSON.stringify(sessions.slice(0, 20)));
    } catch {}
  }
}

function loadSessions() {
  try {
    const stored = localStorage.getItem('sae_sessions');
    if (stored) sessions = JSON.parse(stored);
    renderHistoryList();
  } catch {}
}

function renderHistoryList() {
  if (!historyListEl) return;
  historyListEl.innerHTML = '';
  sessions.forEach(session => {
    const item = document.createElement('div');
    item.className = `history-item ${session.id === currentSessionId ? 'active' : ''}`;
    item.setAttribute('role', 'listitem');
    item.setAttribute('tabindex', '0');
    item.textContent = session.title;
    item.addEventListener('click', () => loadSession(session));
    item.addEventListener('keydown', e => { if (e.key === 'Enter') loadSession(session); });
    historyListEl.appendChild(item);
  });
}

function loadSession(session) {
  currentSessionId = session.id;
  history = [...session.history];

  messagesEl.innerHTML = '';

  // Rebuild messages (skip welcome)
  let pairs = [];
  for (let i = 0; i < history.length; i += 2) {
    if (history[i] && history[i + 1]) {
      pairs.push([history[i].content, history[i + 1].content]);
    }
  }

  pairs.forEach(([u, a]) => {
    appendMessage('user', u);
    appendMessage('assistant', a);
  });

  scrollToBottom();
  renderHistoryList();
}
