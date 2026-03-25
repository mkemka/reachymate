/* ═══════════════════════════════════════════════════
   Reachy Receptionist Dashboard — main.js
   Polls /receptionist/status, /camera/snapshot,
   /chat/poll and manages enroll + members UI.
   ═══════════════════════════════════════════════════ */

'use strict';

// ── Helpers ───────────────────────────────────────────────────────────────────

function qs(sel) { return document.querySelector(sel); }

async function apiFetch(url, opts = {}, timeoutMs = 5000) {
  const ctrl = new AbortController();
  const tid = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...opts, signal: ctrl.signal });
    return resp;
  } finally {
    clearTimeout(tid);
  }
}

async function apiJson(url, opts, timeoutMs) {
  try {
    const r = await apiFetch(url, opts, timeoutMs);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── State ─────────────────────────────────────────────────────────────────────

let _lastChatSeq = 0;
let _sessionTimerInterval = null;
let _sessionExpiresAt = 0;       // ms epoch when current session expires
let _lastStateKey = '';

// ── DOM refs ──────────────────────────────────────────────────────────────────

const $ = {
  overlay:        qs('#disabled-overlay'),
  headerLed:      qs('#header-led'),
  headerStateText:qs('#header-state-text'),
  stateBadge:     qs('#state-badge'),
  identityName:   qs('#identity-name'),
  identityPoints: qs('#identity-points'),
  pointsValue:    qs('#points-value'),
  identityMsg:    qs('#identity-message'),
  sessionTimer:   qs('#session-timer'),
  cameraImg:      qs('#camera-img'),
  noCameraMsg:    qs('#no-camera'),
  enrollPreview:  qs('#enroll-preview'),
  enrollNoCamera: qs('#enroll-no-camera'),
  logEntries:     qs('#log-entries'),
  enrollName:     qs('#enroll-name'),
  enrollBtn:      qs('#enroll-btn'),
  enrollResult:   qs('#enroll-result'),
  memberList:     qs('#member-list'),
  memberCount:    qs('#member-count'),
  refreshBtn:     qs('#refresh-members-btn'),
};

// ── LED helper ────────────────────────────────────────────────────────────────

function setLed(state) {
  const led = $.headerLed;
  led.className = 'led';
  const map = { green: 'green', blue: 'blue', red: 'red', yellow: 'yellow', off: 'off' };
  led.classList.add(map[state] || 'off');
}

// ── Status polling ────────────────────────────────────────────────────────────

async function pollStatus() {
  const data = await apiJson('/receptionist/status');

  if (!data) return;

  // If receptionist mode is disabled, show overlay
  if (data.state === 'DISABLED') {
    $.overlay.classList.add('show');
    return;
  }
  $.overlay.classList.remove('show');

  const stateKey = data.state + (data.member_id || '') + (data.roo_points ?? '');
  if (stateKey === _lastStateKey) return;   // no change
  _lastStateKey = stateKey;

  // LED
  setLed(data.led_state || 'off');
  $.headerStateText.textContent = data.state || '—';

  // State badge
  $.stateBadge.className = 'state-badge';
  const stateClass = {
    IDLE:        'idle',
    SCANNING:    'scanning',
    RECOGNISED:  'recognised',
    DENIED:      'denied',
  }[data.state] || 'idle';
  $.stateBadge.classList.add(stateClass);
  $.stateBadge.textContent = _stateBadgeText(data);

  // Identity card
  if (data.state === 'RECOGNISED' && data.display_name) {
    $.identityName.textContent = data.display_name;
    $.identityName.style.display = '';
    $.identityPoints.style.display = '';
    $.pointsValue.textContent = data.roo_points != null ? data.roo_points : '—';
    $.identityMsg.textContent = data.message || '';

    // Session timer
    if (data.session_expires_in > 0) {
      _sessionExpiresAt = Date.now() + data.session_expires_in * 1000;
      _startSessionTimer();
    }
  } else {
    $.identityName.style.display = 'none';
    $.identityPoints.style.display = 'none';
    $.identityMsg.textContent = data.message || 'Waiting for visitor...';
    _stopSessionTimer();
    $.sessionTimer.style.display = 'none';
  }
}

function _stateBadgeText(data) {
  switch (data.state) {
    case 'IDLE':       return 'IDLE — Waiting for visitor';
    case 'SCANNING':   return 'SCANNING — Identifying...';
    case 'RECOGNISED': return 'RECOGNISED — Access granted';
    case 'DENIED':     return 'DENIED — ' + (data.message || 'Check-in failed');
    default:           return data.state;
  }
}

function _startSessionTimer() {
  _stopSessionTimer();
  $.sessionTimer.style.display = '';
  _sessionTimerInterval = setInterval(() => {
    const secs = Math.max(0, Math.round((_sessionExpiresAt - Date.now()) / 1000));
    $.sessionTimer.textContent = `Session expires in ${secs}s`;
    if (secs === 0) _stopSessionTimer();
  }, 1000);
}

function _stopSessionTimer() {
  if (_sessionTimerInterval) {
    clearInterval(_sessionTimerInterval);
    _sessionTimerInterval = null;
  }
}

// ── Camera feed ───────────────────────────────────────────────────────────────

async function pollCamera() {
  try {
    const url = `/camera/snapshot?_=${Date.now()}`;
    const r = await apiFetch(url, {}, 3000);
    if (r.ok && r.status !== 503) {
      const blob = await r.blob();
      const objUrl = URL.createObjectURL(blob);
      // Main camera
      if ($.cameraImg.src && $.cameraImg.src.startsWith('blob:')) URL.revokeObjectURL($.cameraImg.src);
      $.cameraImg.src = objUrl;
      $.cameraImg.style.display = 'block';
      $.noCameraMsg.style.display = 'none';
      // Enrollment preview (same feed)
      if ($.enrollPreview.src && $.enrollPreview.src.startsWith('blob:')) URL.revokeObjectURL($.enrollPreview.src);
      $.enrollPreview.src = objUrl;
      $.enrollPreview.style.display = 'block';
      $.enrollNoCamera.style.display = 'none';
    } else {
      _hideCameraFeed();
    }
  } catch {
    _hideCameraFeed();
  }
}

function _hideCameraFeed() {
  $.cameraImg.style.display = 'none';
  $.noCameraMsg.style.display = '';
  $.enrollPreview.style.display = 'none';
  $.enrollNoCamera.style.display = '';
}

// ── Chat log polling ──────────────────────────────────────────────────────────

async function pollChat() {
  const data = await apiJson(`/chat/poll?after=${_lastChatSeq}`);
  if (!data || !Array.isArray(data.messages) || data.messages.length === 0) return;

  _lastChatSeq = data.seq || _lastChatSeq;

  // Remove empty placeholder
  const placeholder = $.logEntries.querySelector('.log-empty');
  if (placeholder) placeholder.remove();

  for (const msg of data.messages) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const role = document.createElement('span');
    role.className = 'log-role ' + (msg.tool ? 'tool' : (msg.role || 'user'));
    role.textContent = msg.tool ? 'TOOL' : (msg.role || 'user').toUpperCase();

    const content = document.createElement('span');
    content.className = 'log-content';
    content.textContent = msg.content || '';

    entry.appendChild(role);
    entry.appendChild(content);
    $.logEntries.appendChild(entry);
  }

  // Auto-scroll to bottom
  $.logEntries.scrollTop = $.logEntries.scrollHeight;
}

// ── Enrollment ────────────────────────────────────────────────────────────────

$.enrollBtn.addEventListener('click', async () => {
  const name = $.enrollName.value.trim();
  if (!name) {
    _showEnrollResult(false, 'Please enter a display name first.');
    return;
  }

  $.enrollBtn.disabled = true;
  $.enrollBtn.textContent = 'Enrolling...';
  _hideEnrollResult();

  const r = await apiFetch('/receptionist/enroll', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ display_name: name }),
  }, 15000);

  $.enrollBtn.disabled = false;
  $.enrollBtn.textContent = 'Capture + Enroll';

  if (!r) {
    _showEnrollResult(false, 'Request timed out. Is the app running?');
    return;
  }

  let data;
  try { data = await r.json(); } catch { data = null; }

  if (r.ok && data && data.ok) {
    _showEnrollResult(true,
      `Enrolled: ${data.display_name}\nPassphrase: "${data.phrase_saved}"\nID: ${data.person_id}`
    );
    $.enrollName.value = '';
    // Refresh members tab
    await loadMembers();
  } else {
    const errMsg = data ? (data.error || data.hint || 'Unknown error') : 'Server error';
    const hint   = data && data.hint && data.hint !== errMsg ? `\n${data.hint}` : '';
    _showEnrollResult(false, `Error: ${errMsg}${hint}`);
  }
});

function _showEnrollResult(ok, msg) {
  $.enrollResult.style.display = '';
  $.enrollResult.className = 'enroll-result ' + (ok ? 'ok' : 'err');
  $.enrollResult.textContent = msg;
}
function _hideEnrollResult() {
  $.enrollResult.style.display = 'none';
}

// ── Members tab ───────────────────────────────────────────────────────────────

async function loadMembers() {
  const data = await apiJson('/receptionist/people');
  if (!data || !data.ok) {
    $.memberList.innerHTML = '<div class="empty-state"><p>Could not load members.<br>Is receptionist mode active?</p></div>';
    $.memberCount.textContent = '—';
    return;
  }

  const people = data.people || [];
  $.memberCount.textContent = `${people.length} enrolled`;

  if (people.length === 0) {
    $.memberList.innerHTML = '<div class="empty-state"><p>No members enrolled yet.<br>Use the Enroll tab to add someone.</p></div>';
    return;
  }

  $.memberList.innerHTML = '';
  for (const p of people) {
    $.memberList.appendChild(_buildMemberCard(p));
  }
}

function _buildMemberCard(p) {
  const card = document.createElement('div');
  card.className = 'member-card';
  card.dataset.personId = p.person_id;

  const pts = p.roo_points != null ? p.roo_points : '—';

  card.innerHTML = `
    <div class="member-name">${_esc(p.display_name || p.person_id)}</div>
    <div class="member-meta">
      <span class="member-points-badge">${_esc(String(pts))} pts</span>
      <span class="member-id">${_esc(p.person_id)}</span>
    </div>
    <div class="member-actions">
      <form class="add-points-form" data-pid="${_esc(p.person_id)}">
        <input class="add-points-input" type="number" min="1" max="9999" placeholder="pts" required />
        <button type="submit" class="btn btn-yellow btn-sm">+ Add Points</button>
      </form>
      <button class="btn btn-danger btn-sm" data-action="delete" data-pid="${_esc(p.person_id)}">Delete</button>
    </div>
  `;

  // Add points
  card.querySelector('.add-points-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = e.target.querySelector('input');
    const pts = parseInt(input.value, 10);
    if (!pts || pts < 1) return;
    const r = await apiFetch(`/receptionist/people/${encodeURIComponent(p.person_id)}/add_points`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ points: pts }),
    }, 5000);
    if (r && r.ok) {
      const d = await r.json();
      const badge = card.querySelector('.member-points-badge');
      if (badge) badge.textContent = `${d.new_balance} pts`;
      input.value = '';
    }
  });

  // Delete
  card.querySelector('[data-action="delete"]').addEventListener('click', async () => {
    if (!confirm(`Remove ${p.display_name || p.person_id} from the registry? This cannot be undone.`)) return;
    const r = await apiFetch(`/receptionist/people/${encodeURIComponent(p.person_id)}`, { method: 'DELETE' }, 5000);
    if (r && r.ok) {
      card.remove();
      const remaining = $.memberList.querySelectorAll('.member-card').length;
      $.memberCount.textContent = `${remaining} enrolled`;
      if (remaining === 0) {
        $.memberList.innerHTML = '<div class="empty-state"><p>No members enrolled yet.<br>Use the Enroll tab to add someone.</p></div>';
      }
    }
  });

  return card;
}

function _esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Tab switching ─────────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const tab = qs(`#tab-${btn.dataset.tab}`);
    if (tab) tab.classList.add('active');
    if (btn.dataset.tab === 'members') loadMembers();
  });
});

$.refreshBtn.addEventListener('click', loadMembers);

// ── Poll loops ────────────────────────────────────────────────────────────────

async function statusLoop() {
  while (true) {
    await pollStatus();
    await sleep(1500);
  }
}

async function cameraLoop() {
  while (true) {
    await pollCamera();
    await sleep(1000);
  }
}

async function chatLoop() {
  while (true) {
    await pollChat();
    await sleep(2000);
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────

(async function boot() {
  // Initial data load
  await Promise.all([pollStatus(), pollCamera()]);
  loadMembers();

  // Start background loops
  statusLoop();
  cameraLoop();
  chatLoop();
})();
