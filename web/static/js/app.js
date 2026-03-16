/* ============================================================
   SerieAI — Shared JavaScript Utilities
   ============================================================ */

// --- API Fetch Wrapper ---
async function apiFetch(url, options = {}) {
  try {
    const res = await fetch(url, options);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.error(`[API] ${url} failed:`, err);
    throw err;
  }
}

// --- Number Formatters ---
function pct(v) { return Math.round((v || 0) * 100); }
function pctStr(v) { return pct(v) + '%'; }
function xgFmt(v) { return (v || 0).toFixed(2); }
function oddsFmt(v) { return v ? v.toFixed(2) : '-'; }
function eurFmt(v) {
  if (v == null) return '-';
  return new Intl.NumberFormat('en-IE', { style: 'currency', currency: 'EUR', minimumFractionDigits: 2 }).format(v);
}
function signedPct(v) {
  if (v == null) return '-';
  const sign = v >= 0 ? '+' : '';
  return sign + v.toFixed(1) + '%';
}

// --- Date Formatters ---
// All time functions use the browser's local timezone automatically.
// Input should be a UTC ISO string (e.g. "2026-02-18T19:45:00Z").
// new Date(iso) parses UTC and converts to local for display.

function formatTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (isNaN(d)) return '';
    return d.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }) + ' ' +
           d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function formatKickoffTime(iso) {
  // Returns just the local kickoff time (e.g. "14:45" in Miami, "20:45" in Italy)
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (isNaN(d)) return '';
    return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (isNaN(d)) return '';
    return d.toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
  } catch { return iso; }
}

function formatDateShort(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (isNaN(d)) return '';
    return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
  } catch { return iso; }
}

function localDateKey(iso) {
  // Returns the local date string "YYYY-MM-DD" for grouping matches by the viewer's date
  if (!iso) return '';
  try {
    const d = new Date(iso);
    if (isNaN(d)) return '';
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
  } catch { return ''; }
}

function timeUntil(iso) {
  if (!iso) return '';
  try {
    const now = Date.now();
    const target = new Date(iso).getTime();
    if (isNaN(target)) return '';
    const diff = target - now;
    if (diff < -9000000) return 'FT';  // >2.5h past = completed
    if (diff < 0) return 'LIVE';
    const days = Math.floor(diff / 86400000);
    const hours = Math.floor((diff % 86400000) / 3600000);
    const mins = Math.floor((diff % 3600000) / 60000);
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  } catch { return ''; }
}

function relativeTime(iso) {
  if (!iso) return '';
  try {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  } catch { return ''; }
}

// --- Color Maps ---
const OUTCOME_COLORS = {
  HOME: 'var(--green)', DRAW: 'var(--amber)', AWAY: 'var(--red)',
  'Home Win': 'var(--green)', Draw: 'var(--amber)', 'Away Win': 'var(--red)'
};

const CONF_BADGE_CLASS = {
  'Very High': 'badge--conf-very-high',
  'High': 'badge--conf-high',
  'Medium-High': 'badge--conf-medium-high',
  'Medium': 'badge--conf-medium',
  'Low': 'badge--conf-low',
  'Very Low': 'badge--conf-very-low'
};

const WINDOW_BADGE_CLASS = {
  far: 'badge--neutral',
  approaching: 'badge--amber',
  imminent: 'badge--red',
  live: 'badge--green',
  completed: 'badge--neutral'
};

// --- Badge Builders ---
function confBadge(level) {
  if (!level) return '<span class="badge badge--neutral">N/A</span>';
  // Case-insensitive lookup (API returns UPPERCASE, CSS map uses Mixed Case)
  const key = Object.keys(CONF_BADGE_CLASS).find(k => k.toLowerCase() === level.toLowerCase());
  const cls = key ? CONF_BADGE_CLASS[key] : 'badge--neutral';
  return `<span class="badge ${cls}">${level}</span>`;
}

function windowBadge(w) {
  if (!w) return '';
  const cls = WINDOW_BADGE_CLASS[w] || 'badge--neutral';
  return `<span class="badge ${cls}">${w}</span>`;
}

function steamBadge(detected) {
  return detected ? '<span class="badge badge--red">STEAM</span>' : '';
}

// --- Probability Bar HTML ---
function probBarHTML(probs) {
  const h = pct(probs.home), d = pct(probs.draw), a = pct(probs.away);
  return `<div class="prob-bar">
    <span class="prob-bar__label prob-bar__label--home">${h}%</span>
    <div class="prob-bar__track">
      <div class="prob-bar__fill--home" style="width:${h}%"></div>
      <div class="prob-bar__fill--draw" style="width:${d}%"></div>
      <div class="prob-bar__fill--away" style="width:${a}%"></div>
    </div>
    <span class="prob-bar__label prob-bar__label--away">${a}%</span>
  </div>
  <div class="prob-bar__legend">
    <span>H</span><span>D ${d}%</span><span>A</span>
  </div>`;
}

// --- Mini Prob Bars (component methods) ---
function miniProbBarsHTML(methods) {
  if (!methods || typeof methods !== 'object') return '<span class="text-small text-tertiary">No component data</span>';
  const names = ['factor', 'xg', 'ml', 'player_xg', 'deep', 'market'];
  const labels = { factor: 'Factor', xg: 'xG', ml: 'ML', player_xg: 'Plyr xG', deep: 'Deep', market: 'Market' };
  let html = '';
  for (const n of names) {
    const m = methods[n];
    if (!m) continue;
    const h = pct(m.home || m.H || m.prob_H || 0);
    const d = pct(m.draw || m.D || m.prob_D || 0);
    const a = pct(m.away || m.A || m.prob_A || 0);
    html += `<div class="mini-prob-bar">
      <span class="mini-prob-bar__label">${labels[n]}</span>
      <div class="mini-prob-bar__track">
        <div class="prob-bar__fill--home" style="width:${h}%"></div>
        <div class="prob-bar__fill--draw" style="width:${d}%"></div>
        <div class="prob-bar__fill--away" style="width:${a}%"></div>
      </div>
      <span class="mini-prob-bar__values">${h}-${d}-${a}</span>
    </div>`;
  }
  return html || '<span class="text-small text-tertiary">No component data</span>';
}

// --- Arrow/Direction Icon ---
function arrowIcon(dir) {
  if (!dir) return '';
  const d = dir.toLowerCase();
  if (d.includes('home') || d.includes('up')) return '<span class="text-green">&#9650;</span>';
  if (d.includes('away') || d.includes('down')) return '<span class="text-red">&#9660;</span>';
  if (d.includes('draw')) return '<span class="text-amber">&#9654;</span>';
  return '<span class="text-tertiary">&#8226;</span>';
}

// --- Team Search ---
let _allTeams = [];

async function loadTeamsList() {
  try {
    const data = await apiFetch('/api/teams');
    _allTeams = data.teams || [];
  } catch {}
}

function initTeamSearch() {
  const input = document.getElementById('team-search');
  const dropdown = document.getElementById('search-dropdown');
  if (!input || !dropdown) return;

  input.addEventListener('input', () => {
    const q = input.value.toLowerCase().trim();
    if (q.length === 0) { dropdown.classList.add('hidden'); return; }
    const matches = _allTeams.filter(t => t.toLowerCase().includes(q));
    if (matches.length === 0) { dropdown.classList.add('hidden'); return; }
    dropdown.innerHTML = matches.map(t =>
      `<a href="/team/${encodeURIComponent(t)}" class="block px-3 py-2 text-small text-secondary" style="transition:all 0.15s" onmouseover="this.style.background='var(--bg-tertiary)';this.style.color='var(--text-primary)'" onmouseout="this.style.background='';this.style.color=''">${t}</a>`
    ).join('');
    dropdown.classList.remove('hidden');
  });

  input.addEventListener('blur', () => setTimeout(() => dropdown.classList.add('hidden'), 200));
  input.addEventListener('focus', () => { if (input.value.trim()) input.dispatchEvent(new Event('input')); });
}

// --- Toast Notifications ---
function showToast(title, lines, color) {
  const toast = document.getElementById('refresh-toast');
  if (!toast) return;
  document.getElementById('toast-title').textContent = title;
  const colorMap = { green: 'toast--green', red: 'toast--red', amber: 'toast--amber' };
  toast.className = `toast ${colorMap[color] || ''}`;
  document.getElementById('toast-body').innerHTML = lines.map(l => `<p class="text-small text-secondary">${l}</p>`).join('');
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 8000);
}

function closeToast() {
  const toast = document.getElementById('refresh-toast');
  if (toast) toast.classList.add('hidden');
}

// --- Smart Refresh ---
let _refreshPollTimer = null;

async function smartRefresh() {
  const btn = document.getElementById('btn-smart-refresh');
  const label = document.getElementById('refresh-label');
  const icon = document.getElementById('refresh-icon');
  if (!btn) return;

  btn.disabled = true;
  btn.style.opacity = '0.6';
  if (label) label.textContent = 'Refreshing...';
  if (icon) icon.classList.add('animate-spin');

  try {
    const data = await apiFetch('/api/refresh/smart', { method: 'POST' });
    if (!data.ok) {
      showToast('Refresh Blocked', [data.message || 'Already running'], 'amber');
      resetRefreshBtn();
      return;
    }
    _refreshPollTimer = setInterval(pollSmartRefresh, 2000);
  } catch (err) {
    showToast('Refresh Error', [err.message], 'red');
    resetRefreshBtn();
  }
}

async function pollSmartRefresh() {
  try {
    const data = await apiFetch('/api/refresh/smart/status');

    if (!data.running && data.status !== 'running') {
      clearInterval(_refreshPollTimer);
      resetRefreshBtn();

      if (data.status === 'completed') {
        const lines = [];
        if (data.odds_updated) lines.push('Odds updated');
        if (data.market_analysis_updated) lines.push('Market intelligence refreshed');
        if (data.predictions_updated) lines.push('Predictions refreshed');
        if (data.player_analysis_updated) lines.push('Player analysis refreshed');
        if (data.lineup_predictions_updated) lines.push('Lineup predictions refreshed');
        if (data.new_results && data.new_results.length) {
          lines.push(`${data.new_results.length} new result(s) fetched`);
        }
        if (data.settled_bets && data.settled_bets.length) {
          const sb = data.settled_bets[0];
          lines.push(`Settled ${sb.settled} bet(s): ${sb.won}W/${sb.lost}L`);
        }
        showToast('Refresh Complete', lines.length ? lines : ['All data up to date'], 'green');
        setTimeout(() => location.reload(), 2000);
      } else if (data.status === 'failed') {
        showToast('Refresh Failed', [data.error || 'Unknown error'], 'red');
      }
    }
  } catch {}
}

function resetRefreshBtn() {
  const btn = document.getElementById('btn-smart-refresh');
  const label = document.getElementById('refresh-label');
  const icon = document.getElementById('refresh-icon');
  if (btn) { btn.disabled = false; btn.style.opacity = ''; }
  if (label) label.textContent = 'Refresh';
  if (icon) icon.classList.remove('animate-spin');
}

// --- Sidebar ---
function initSidebar() {
  const sidebar = document.getElementById('sidebar');
  const toggleBtn = document.getElementById('sidebar-toggle');
  if (!sidebar || !toggleBtn) return;

  // Restore state from localStorage
  const saved = localStorage.getItem('sidebar-collapsed');
  if (saved === 'true') sidebar.classList.add('collapsed');

  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
    localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
  });
}

// --- Sidebar Stats ---
async function loadSidebarStats() {
  try {
    const data = await apiFetch('/api/betting');
    const b = data.bankroll || {};
    const bankrollEl = document.getElementById('sidebar-bankroll');
    const roiEl = document.getElementById('sidebar-roi');
    if (bankrollEl) {
      bankrollEl.textContent = eurFmt(b.current_balance || 1000);
    }
    if (roiEl) {
      // Use pre-computed ROI from backend (sourced from history.json)
      const roi = b.roi != null ? b.roi : 0;
      roiEl.textContent = signedPct(roi);
      roiEl.style.color = roi >= 0 ? 'var(--green)' : 'var(--red)';
    }
  } catch {}
}

// --- Next Match Countdown ---
async function loadNextMatchCountdown() {
  try {
    const data = await apiFetch('/api/dashboard');
    const el = document.getElementById('sidebar-next-match');
    if (!el || !data.matches || !data.matches.length) return;

    // Find earliest upcoming match
    const upcoming = data.matches.filter(m => m.status !== 'completed' && m.commence_time);
    if (!upcoming.length) { el.textContent = 'No upcoming'; return; }

    const earliest = upcoming[0];
    el.textContent = timeUntil(earliest.commence_time);

    // Update every 30 seconds for consistent countdowns
    setInterval(() => {
      el.textContent = timeUntil(earliest.commence_time);
    }, 30000);
  } catch {}
}

// --- Value Bet Alert System ---
const ALERT_CHECK_INTERVAL = 5 * 60 * 1000; // 5 minutes
let _alertsEnabled = localStorage.getItem('alerts_enabled') !== 'false'; // default ON
let _alertTimer = null;
let _firstAlertCheck = true; // suppress alerts on initial page load

function initAlerts() {
  // Request notification permission on first interaction
  if (_alertsEnabled && 'Notification' in window && Notification.permission === 'default') {
    // We'll request on first user interaction to avoid browser blocking
    document.addEventListener('click', function _reqPerm() {
      Notification.requestPermission();
      document.removeEventListener('click', _reqPerm);
    }, { once: true });
  }

  // Start polling
  _alertTimer = setInterval(checkAlerts, ALERT_CHECK_INTERVAL);
  // First check after 10 seconds (let page load first)
  setTimeout(checkAlerts, 10000);
}

async function checkAlerts() {
  if (!_alertsEnabled) return;
  try {
    const data = await apiFetch('/api/alerts/check');
    if (_firstAlertCheck) {
      _firstAlertCheck = false;
      return; // Don't alert on first load — everything is "new"
    }
    if (!data.has_alerts) return;

    const newBets = data.new_bets || [];
    const improved = data.improved_bets || [];

    // In-app toast
    const lines = [];
    for (const b of newBets.slice(0, 3)) {
      lines.push(`🎯 ${b.match}: ${b.selection} @${b.odds} (${b.edge_pct.toFixed(1)}% edge)`);
    }
    for (const b of improved.slice(0, 2)) {
      lines.push(`📈 ${b.match}: odds ${b.old_odds} → ${b.new_odds}`);
    }
    if (newBets.length > 3) lines.push(`...and ${newBets.length - 3} more`);
    if (lines.length > 0) {
      showToast('Value Bet Alert', lines, 'green');
    }

    // Browser notification
    if ('Notification' in window && Notification.permission === 'granted' && newBets.length > 0) {
      const body = newBets.slice(0, 3).map(b =>
        `${b.match}: ${b.selection} @${b.odds} (${b.edge_pct.toFixed(1)}%)`
      ).join('\n');
      new Notification('SerieAI — New Value Bets', {
        body: body + (newBets.length > 3 ? `\n+${newBets.length - 3} more` : ''),
        icon: '/static/img/favicon.png',
        tag: 'seriea-alert', // replaces previous notification
      });
    }
  } catch (e) {
    // Silent fail — alerts are non-critical
  }
}

function toggleAlerts(enabled) {
  _alertsEnabled = enabled;
  localStorage.setItem('alerts_enabled', enabled ? 'true' : 'false');
  if (enabled && !_alertTimer) {
    _alertTimer = setInterval(checkAlerts, ALERT_CHECK_INTERVAL);
  } else if (!enabled && _alertTimer) {
    clearInterval(_alertTimer);
    _alertTimer = null;
  }
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  initSidebar();
  initTeamSearch();
  loadTeamsList();
  loadSidebarStats();
  loadNextMatchCountdown();
  initAlerts();
});
