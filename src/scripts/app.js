/**
 * app.js — SAE Lab main entry point
 * Orchestrates tab switching, floating nav hub, and module initialisation.
 */

import { initChat } from './chat.js';
import { initSteering } from './steering.js';
import { initExplore } from './explore.js';
import { initFeatures } from './features.js';

// ── State ──────────────────────────────────────────────────────────────────────
export const appState = {
  currentTab: 'chat',
  modelReady: false,
  featuresCount: 0,
};

// ── API ────────────────────────────────────────────────────────────────────────
export async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function apiPost(path, data) {
  return api(path, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// ── Tabs ───────────────────────────────────────────────────────────────────────
function initTabs() {
  const navBtns = document.querySelectorAll('.hub-nav-btn[data-tab]');
  const panels  = document.querySelectorAll('.panel[id^="panel-"]');

  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      if (tab === appState.currentTab) return;
      switchTab(tab, navBtns, panels);
    });
  });
}

function switchTab(tab, navBtns, panels) {
  appState.currentTab = tab;

  navBtns.forEach(btn => {
    const isActive = btn.dataset.tab === tab;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', String(isActive));
  });

  panels.forEach(panel => {
    const isActive = panel.id === `panel-${tab}`;
    panel.classList.toggle('active', isActive);
    panel.hidden = !isActive;
  });

  window.dispatchEvent(new CustomEvent('sae:tab', { detail: { tab } }));
}

// ── Status ─────────────────────────────────────────────────────────────────────
async function pollStatus() {
  const indicator = document.getElementById('status-indicator');
  const featTotal = document.getElementById('features-total');

  try {
    const health = await api('/health');
    appState.modelReady = health.model_loaded;
    appState.featuresCount = health.features_count;

    if (health.error) {
      indicator.dataset.state = 'error';
    } else if (health.model_loaded) {
      indicator.dataset.state = 'ok';
    } else {
      indicator.dataset.state = 'loading';
    }

    if (featTotal && health.features_count) {
      featTotal.textContent = health.features_count.toLocaleString();
    }
  } catch {
    indicator.dataset.state = 'error';
  }
}

// ── Init ───────────────────────────────────────────────────────────────────────
async function init() {
  const overlay = document.getElementById('loading-overlay');
  overlay.classList.add('hidden');

  initTabs();

  initChat(appState);
  initSteering(appState);
  initExplore(appState);
  initFeatures(appState);

  await pollStatus();
  setInterval(pollStatus, 10_000);
}

document.addEventListener('DOMContentLoaded', init);
