/**
 * app.js — SAE Lab main entry point
 * Orchestrates tab switching, sidebar, and module initialisation.
 */

import { initChat } from './chat.js';
import { initSteering } from './steering.js';
import { initExplore } from './explore.js';
import { initFeatures } from './features.js';

// ── State ──────────────────────────────────────────────────────────────────────
export const appState = {
  currentTab: 'chat',
  sidebarCollapsed: true,
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

// ── Sidebar ────────────────────────────────────────────────────────────────────
function initSidebar() {
  const app = document.getElementById('app');
  const toggle = document.getElementById('sidebar-toggle');

  // Start collapsed
  app.classList.add('sidebar-collapsed');

  toggle.addEventListener('click', () => {
    appState.sidebarCollapsed = !appState.sidebarCollapsed;
    app.classList.toggle('sidebar-collapsed', appState.sidebarCollapsed);
    toggle.setAttribute('aria-expanded', String(!appState.sidebarCollapsed));
    toggle.setAttribute('aria-label', appState.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar');
  });
}

// ── Tabs ───────────────────────────────────────────────────────────────────────
function initTabs() {
  const navItems = document.querySelectorAll('.nav-item[data-tab]');
  const panels = document.querySelectorAll('.panel[id^="panel-"]');

  navItems.forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      if (tab === appState.currentTab) return;
      switchTab(tab, navItems, panels);
    });
  });
}

function switchTab(tab, navItems, panels) {
  appState.currentTab = tab;

  // Update nav
  navItems.forEach(btn => {
    const isActive = btn.dataset.tab === tab;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', String(isActive));
  });

  // Update panels
  panels.forEach(panel => {
    const isActive = panel.id === `panel-${tab}`;
    panel.classList.toggle('active', isActive);
    panel.hidden = !isActive;
  });

  // Notify modules that a tab became active
  window.dispatchEvent(new CustomEvent('sae:tab', { detail: { tab } }));
}

// ── Status ─────────────────────────────────────────────────────────────────────
async function pollStatus() {
  const indicator = document.getElementById('status-indicator');
  const modelChip = document.getElementById('model-chip');
  const featTotal = document.getElementById('features-total');

  try {
    const health = await api('/health');
    appState.modelReady = health.model_loaded;
    appState.featuresCount = health.features_count;

    if (health.error) {
      indicator.dataset.state = 'error';
    } else if (health.model_loaded) {
      indicator.dataset.state = 'ok';
      if (modelChip) {
        modelChip.textContent = `Llama 3.2 1B · SAE L8 · ${health.features_count} features`;
      }
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
  // Hide loading overlay
  const overlay = document.getElementById('loading-overlay');
  overlay.classList.add('hidden');

  initSidebar();
  initTabs();

  // Init modules
  initChat(appState);
  initSteering(appState);
  initExplore(appState);
  initFeatures(appState);

  // Start health polling
  await pollStatus();
  setInterval(pollStatus, 10_000);
}

// Boot
document.addEventListener('DOMContentLoaded', init);
