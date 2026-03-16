/**
 * chat.js — Chat panel logic
 * Handles message submission, display, conversation history, and inline attribution.
 */

import { renderInlineAttribution } from './attribution.js';

// ── State ──────────────────────────────────────────────────────────────────────
let history = [];
let isLoading = false;

// ── DOM refs ───────────────────────────────────────────────────────────────────
let messagesEl, chatForm, inputEl, sendBtn, welcomeEl;

// ── Init ───────────────────────────────────────────────────────────────────────
export function initChat(_appState) {
  messagesEl = document.getElementById('messages');
  chatForm   = document.getElementById('chat-form');
  inputEl    = document.getElementById('chat-input');
  sendBtn    = document.getElementById('send-btn');
  welcomeEl  = document.getElementById('welcome');

  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 160)}px`;
    sendBtn.disabled = !inputEl.value.trim();
  });

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

  document.querySelectorAll('.example-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      inputEl.value = chip.dataset.prompt;
      inputEl.dispatchEvent(new Event('input'));
      chatForm.requestSubmit();
    });
  });
}

// ── Send message ───────────────────────────────────────────────────────────────
async function sendMessage(message) {
  if (isLoading) return;
  isLoading = true;

  if (welcomeEl) welcomeEl.style.display = 'none';

  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;
  sendBtn.classList.add('loading');

  appendMessage('user', message);
  scrollToBottom();

  const thinkingId = appendThinking();
  history.push({ role: 'user', content: message });

  let msgEl = null;
  let responseText = '';

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: history.slice(-6) }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(6)); } catch { continue; }

        if (evt.type === 'log') {
          // Only show log stages while thinking (before tokens arrive)
          if (!msgEl) updateThinkingStage(thinkingId, evt.text);

        } else if (evt.type === 'token') {
          if (!msgEl) {
            removeThinking(thinkingId);
            msgEl = appendMessage('assistant', '');
          }
          responseText += evt.text;
          updateMessageText(msgEl, responseText);

        } else if (evt.type === 'result') {
          // Fallback: if no tokens streamed, render full response now
          if (!msgEl) {
            removeThinking(thinkingId);
            msgEl = appendMessage('assistant', evt.response || '');
            responseText = evt.response || '';
          }
          history.push({ role: 'assistant', content: responseText });
          if ((evt.input_features?.length || evt.output_features?.length) && msgEl) {
            renderInlineAttribution(msgEl, {
              inputFeatures: evt.input_features || [],
              outputFeatures: evt.output_features || [],
            });
          }

        } else if (evt.type === 'error') {
          removeThinking(thinkingId);
          appendError(evt.text || 'Generation failed');
          history.pop();
        }
      }
      scrollToBottom();
    }

  } catch (err) {
    removeThinking(thinkingId);
    if (!msgEl) appendError(err.message || 'Request failed. Is the model loaded?');
    history.pop();
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

  if (role === 'assistant') {
    el.innerHTML = `
      <div class="message-role assistant-role">Assistant</div>
      <div class="message-inner">
        <div class="message-main">
          <div class="message-body">${escapeHtml(content)}</div>
        </div>
        <div class="message-attr" id="msg-attr-${Date.now()}"></div>
      </div>
    `;
  } else {
    el.innerHTML = `
      <div class="message-role user-role">You</div>
      <div class="message-body">${escapeHtml(content)}</div>
    `;
  }

  messagesEl.appendChild(el);
  return el;
}

function updateMessageText(msgEl, text) {
  const body = msgEl.querySelector('.message-body');
  if (body) body.innerHTML = escapeHtml(text);
}

function appendThinking() {
  const id = `thinking-${Date.now()}`;
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.id = id;
  el.innerHTML = `
    <div class="message-role assistant-role">Assistant</div>
    <div class="message-inner">
      <div class="message-main">
        <div class="thinking-stages" id="${id}-stages"></div>
      </div>
    </div>
  `;
  messagesEl.appendChild(el);
  scrollToBottom();
  return id;
}

function updateThinkingStage(id, text) {
  const stagesEl = document.getElementById(`${id}-stages`);
  if (!stagesEl) return;
  const prev = stagesEl.querySelector('.thinking-stage:last-child');
  if (prev) prev.classList.add('done');
  const stage = document.createElement('div');
  stage.className = 'thinking-stage';
  stage.innerHTML = `<span class="thinking-stage-dot"></span><span class="thinking-stage-label">${text}</span>`;
  stagesEl.appendChild(stage);
  requestAnimationFrame(() => requestAnimationFrame(() => stage.classList.add('visible')));
  scrollToBottom();
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
