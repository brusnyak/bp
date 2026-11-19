/* Model router page UI. Talks to /api/* on the local backend; with ?mock=1 it
   reads ui/models/mock/*.json so the page can be previewed without the server.
   Every network call fails soft: the page still renders, errors show inline. */
document.addEventListener('DOMContentLoaded', () => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const t = (s) => (window.i18n ? window.i18n.t(s) : s);
  const MOCK = /[?&]mock=1\b/.test(location.search);

  let state = {
    hardware: null,
    models: [], languages: [],
    src: 'en', tgt: 'sk',
    plan: null, route: null,
    tab: 'stt', search: '', landLang: '', profile: 'this', profiles: [],
    job: null,
  };

  /* ---------- guest token (auto-acquire for real backend) ---------- */
  async function ensureToken() {
    if (localStorage.getItem('bp_token')) return;
    try {
      const r = await fetch('/api/guest', { method: 'POST' });
      if (r.ok) { const d = await r.json(); localStorage.setItem('bp_token', d.token); }
    } catch (e) { /* backend may be down; mock mode handles it */ }
  }

  /* ---------- mock fetch ---------- */
  async function mock(path) {
    const r = await fetch(`/ui/models/mock/${path}.json`, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`mock ${path}: ${r.status}`);
    return r.json();
  }

  async function api(path, opts = {}) {
    if (MOCK) {
      const map = {
        '/hardware': 'hardware',
        '/catalog': 'catalog',
        '/route/plan': 'plan',
        '/route': 'route',
      };
      const key = map[path];
      if (key) return mock(key);
      if (path.startsWith('/jobs/')) return mock('jobs');
      throw new Error(`no mock for ${path}`);
    }
    const token = localStorage.getItem('bp_token');
    const headers = opts.headers || {};
    if (token) headers.Authorization = `Bearer ${token}`;
    const r = await fetch('/api' + path, { ...opts, headers });
    if (r.status === 401 && token) {   // stale/expired guest token: drop it and retry once
      localStorage.removeItem('bp_token');
      return api(path, opts);
    }
    if (r.status === 401) { throw new Error('unauthorized'); }
    return r.json();
  }

  /* ---------- toast ---------- */
  const toast = $('#toast');
  function showToast(msg, kind = 'info') {
    toast.textContent = msg;
    toast.className = 'toast show ' + kind;
    setTimeout(() => { toast.className = 'toast hidden'; }, 2500);
  }

  /* ---------- hardware ---------- */
  function renderHardware() {
    renderProfileTabs();
    const prof = state.profiles.find((x) => x.key === state.profile);
    const h = state.profile === 'this' || !prof ? (state.hardware || {}) : { cpu_cores: prof.cpu_cores, ram_gb: prof.ram_gb, gpu: prof.gpu, disk_free_gb: null };
    const list = $('#hwList'); list.textContent = '';
    const gpu = (h.gpu && h.gpu.backend && h.gpu.backend !== 'cpu') ? h.gpu : null;
    const rows = [
      { label: t('CPU'), value: h.cpu_cores != null ? h.cpu_cores + ' cores' : '—' },
      { label: t('RAM'), value: h.ram_gb != null ? h.ram_gb + ' GB' : '—' },
      { label: t('GPU'), value: gpu ? (gpu.name || gpu.backend) : '—' },
      { label: t('VRAM'), value: gpu && gpu.vram_gb != null ? gpu.vram_gb + ' GB' : '—' },
      { label: t('Free disk'), value: h.disk_free_gb != null ? h.disk_free_gb + ' GB' : '—' },
    ];
    rows.forEach((row) => {
      const item = document.createElement('div');
      item.className = 'tile';
      item.innerHTML = '<span class="tile-label">' + escapeHtml(row.label) + '</span><span class="tile-value">' + escapeHtml(row.value) + '</span>';
      list.appendChild(item);
    });
    const v = $('#hwVerdict');
    if (gpu && gpu.vram_gb != null && gpu.vram_gb >= 4) {
      v.textContent = state.profile === 'this' ? t('This machine has a GPU: the recommended route can use larger models.') : t('Planning for this target: fit labels below are for it, not for this computer.');
    } else if (gpu && gpu.backend === 'mps') {
      v.textContent = t('Apple Silicon: small and medium models run on the GPU or CPU; CUDA-only models are marked too heavy.');
    } else {
      v.textContent = state.profile === 'this' ? t('This machine has no GPU: the recommended route uses small CPU models.') : t('Planning for a CPU-only target: GPU-only models are marked too heavy.');
    }
  }

  function renderProfileTabs() {
    const box = $('#profileTabs'); if (!box) return;
    box.textContent = '';
    const mk = (key, label) => {
      const b = document.createElement('button'); b.type = 'button'; b.dataset.profile = key;
      b.setAttribute('aria-pressed', String(state.profile === key)); b.textContent = label; box.appendChild(b);
    };
    mk('this', t('This computer'));
    state.profiles.forEach((p) => mk(p.key, p.label));
  }

  /* ---------- language picker ---------- */
  function renderLanguages() {
    const srcSel = $('#srcLang'), tgtSel = $('#tgtLang'), landLang = $('#landLang');
    // kept for compatibility; src/tgt now come from plan? No — the plan route uses
    // src/tgt from state. We render selects only if they exist.
    if (!srcSel) return;
    [srcSel, tgtSel].forEach((s) => { s.textContent = ''; });
    landLang.textContent = '';
    { const all = document.createElement('option'); all.value = ''; all.textContent = t('All languages'); landLang.appendChild(all); }
    const seen = new Set();
    state.languages.forEach((l) => {
      if (seen.has(l.code)) return; seen.add(l.code);
      const opt = document.createElement('option');
      opt.value = l.code; opt.textContent = (l.name || l.code.toUpperCase()) + ' (' + l.code + ')';
      const o2 = opt.cloneNode(true);
      srcSel.appendChild(opt); tgtSel.appendChild(o2);
      const lo = document.createElement('option'); lo.value = l.code; lo.textContent = opt.textContent; landLang.appendChild(lo);
    });
    srcSel.value = state.src; tgtSel.value = state.tgt;
  }

  /* ---------- plan ---------- */
  function renderPlan() {
    const empty = $('#planEmpty'), full = $('#planFull');
    if (!state.plan) { if (empty) { empty.hidden = false; full.hidden = true; } return; }
    if (empty) { empty.hidden = true; full.hidden = false; }
    const p = state.plan;
    const chain = $('#chain'); chain.textContent = '';
    const stage = (key, model) => {
      const chip = document.createElement('span');
      chip.className = 'stage-chip' + (model && model.installed ? ' installed' : '');
      chip.textContent = model ? (model.name || model.id) : '—';
      return chip;
    };
    chain.appendChild(stage('stt', p.stt));
    chain.appendChild(makeArrow());
    chain.appendChild(stage('mt', p.mt));
    chain.appendChild(makeArrow());
    chain.appendChild(stage('tts', p.tts));
    $('#planSize').textContent = p.total_mb != null ? '≈ ' + p.total_mb.toFixed(0) + ' MB to download' : '';
    $('#planNote').textContent = p.note || '';
    const btn = $('#planInstall');
    const missing = p.missing || [];
    const complete = !!(p.stt && p.mt && p.tts);
    btn.disabled = !complete;
    btn.textContent = missing.length ? t('Install') + ' ' + missing.length + ' ' + t('missing') + ' · ' + t('use route') : t('Use this route');
    btn.title = complete ? '' : t('No complete route for this pair on this target.');
    btn.onclick = async () => {
      try {
        if (missing.length) {
          const j = await api('/models/install', { method: 'POST', body: JSON.stringify({ ids: missing }), headers: { 'Content-Type': 'application/json' } });
          state.job = j.job_id;
          showToast('Install started', 'info');
          pollJob();
        } else {
          await api('/route', { method: 'POST', body: JSON.stringify({ src: p.src, tgt: p.tgt, stt: p.stt.id, mt: p.mt.id, tts: p.tts.id }), headers: { 'Content-Type': 'application/json' } });
          showToast('Route saved', 'info');
        }
      } catch (e) { showToast('Failed: ' + e.message, 'error'); }
    };
  }
  function makeArrow() {
    const a = document.createElement('span');
    a.className = 'arrow'; a.textContent = '→';
    return a;
  }

  /* ---------- model cards (rows) ---------- */
  function renderLandscape() {
    const list = $('#landList'); list.textContent = '';
    let models = state.models.filter((m) => m.stage === state.tab);
    if (state.landLang) models = models.filter((m) => (m.languages || []).includes(state.landLang));
    if (state.search) {
      const s = state.search.toLowerCase();
      models = models.filter((m) => (m.name + ' ' + m.id + ' ' + m.engine + ' ' + (m.languages || []).join(' ')).toLowerCase().includes(s));
    }
    models.sort((a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0) || (a.size_mb || 0) - (b.size_mb || 0));
    if (models.length === 0) { list.innerHTML = '<div class="empty">' + t('No models match.') + '</div>'; return; }
    models.forEach((m) => {
      const row = document.createElement('article');
      row.className = 'model-row';
      const installed = m.installed ? '<span class="chip ok">' + t('Installed') + '</span>' : '';
      const fitText = { good: t('Runs well'), ok: t('Runs'), 'too-heavy': t('Too heavy') }[m.fit] || m.fit;
      const fitCls = { good: 'ok', ok: 'warn', 'too-heavy': 'bad' }[m.fit] || '';
      const basis = m.fit_basis === 'measured' ? t('measured') : (m.fit_basis === 'untested' ? t('not tested on this target') : t('from specs'));
      const fitBadge = '<span class="chip ' + fitCls + '">' + fitText + ' · ' + basis + '</span>';
      const recommended = m.recommended ? '<span class="chip accent">' + t('Recommended') + '</span>' : '';
      const facts = (m.measured || []).slice(0, 4).map((x) => '<span class="chip" title="' + escapeHtml(x.source || '') + '">' + escapeHtml((x.hardware || x.device || '') + ': ' + x.metric + ' ' + x.value) + '</span>').join('');
      row.innerHTML =
        '<div class="model-head"><h4>' + escapeHtml(m.name || m.id) + '</h4><span class="engine">' + escapeHtml(m.engine) + '</span>' + installed + fitBadge + recommended + '</div>' +
        '<div class="model-meta">' +
        (m.size_mb != null ? '<span>' + m.size_mb.toFixed(0) + ' MB</span>' : '') +
        '<span>' + (m.devices || []).join(' · ') + '</span>' +
        (m.streaming ? '<span>streaming</span>' : '') +
        (m.clone ? '<span>cloning</span>' : '') +
        '</div>' +
        '<div class="model-actions">' +
        (m.installed
          ? '<button class="btn btn-primary" data-act="use" data-id="' + escapeHtml(m.id) + '">' + t('Use') + '</button>'
          : '<button class="btn" data-act="install" data-id="' + escapeHtml(m.id) + '">' + t('Install') + '</button>') +
        '<button class="btn" data-act="remove" data-id="' + escapeHtml(m.id) + '" ' + (m.installed ? '' : 'disabled') + '>' + t('Remove') + '</button>' +
        '</div>' +
        (m.licence ? '<p class="licence">' + t('Licence') + ': <a href="' + escapeHtml(m.licence.url || '#') + '" target="_blank" rel="noopener">' + escapeHtml(m.licence.label) + '</a></p>' : '') +
        (m.when ? '<p class="model-when">' + escapeHtml(m.when) + '</p>' : (m.why ? '<p class="model-when">' + escapeHtml(m.why) + '</p>' : '')) +
        (m.avoid_when ? '<p class="model-avoid">' + t('Avoid when') + ': ' + escapeHtml(m.avoid_when) + '</p>' : '') +
        (facts ? '<div class="model-facts">' + facts + '</div>' : '') +
        '<p class="model-note">' + escapeHtml(m.notes || '') + '</p>';
      list.appendChild(row);
    });
  }
  $('#landList').addEventListener('click', (e) => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const id = b.dataset.id; const act = b.dataset.act;
    if (act === 'install') { installModel([id]); }
    else if (act === 'remove') { removeModel(id); }
    else if (act === 'use') { showToast('Selected ' + id, 'info'); }
  });

  /* ---------- install / remove ---------- */
  async function installModel(ids) {
    try {
      const j = await api('/models/install', { method: 'POST', body: JSON.stringify({ ids }), headers: { 'Content-Type': 'application/json' } });
      state.job = j.job_id; showToast('Install queued', 'info'); pollJob();
    } catch (e) { showToast('Install failed: ' + e.message, 'error'); }
  }
  async function removeModel(id) {
    if (!confirm(t('Remove this model?'))) return;
    try {
      await api('/models/' + encodeURIComponent(id), { method: 'DELETE' });
      showToast('Removed', 'info');
      loadCatalog();
    } catch (e) { showToast('Remove failed: ' + e.message, 'error'); }
  }

  /* ---------- job polling ---------- */
  async function pollJob() {
    if (!state.job) return;
    try {
      const j = await api('/jobs/' + encodeURIComponent(state.job));
      const pFill = $('#pFill'), pText = $('#pText'), prog = $('#planProgress');
      const pct = Math.round((j.progress || 0) * 100);
      if (pFill) { pFill.style.width = pct + '%'; pText.textContent = pct + '%'; }
      if (prog) { prog.hidden = false; }
      const btn = $('#planInstall');
      if (j.state === 'done' || j.state === 'error') {
        showToast(j.state === 'done' ? 'Install complete' : ('Install failed: ' + (j.error || '')), j.state === 'done' ? 'info' : 'error');
        if (prog) prog.hidden = true;
        state.job = null;
        loadCatalog();
        return;
      }
      setTimeout(pollJob, 1200);
    } catch (e) { state.job = null; }
  }

  function profileQ() { return state.profile && state.profile !== 'this' ? '&profile=' + encodeURIComponent(state.profile) : ''; }

  /* ---------- load data ---------- */
  async function loadCatalog() {
    await ensureToken();
    try {
      const [cat, hw] = await Promise.all([
        api('/catalog?stage=&lang=&q=' + profileQ()),
        api('/hardware'),
      ]);
      state.hardware = hw;
      state.models = cat.models || [];
      state.languages = cat.languages || [];
      state.profiles = cat.profiles || state.profiles;
      renderHardware(); renderLanguages(); renderLandscape();
      loadPlan();
    } catch (e) {
      showToast('Catalog load failed: ' + e.message, 'error');
      renderHardware(); renderLanguages(); renderLandscape();
    }
  }
  async function loadPlan() {
    try {
      const plan = await api('/route/plan?src=' + encodeURIComponent(state.src) + '&tgt=' + encodeURIComponent(state.tgt) + profileQ());
      state.plan = plan; renderPlan();
    } catch (e) { /* plan is optional */ }
  }
  async function loadRoute() {
    await ensureToken();
    try { state.route = await api('/route'); } catch (e) { /* ignore */ }
  }

  /* ---------- advanced add ---------- */
  const addForm = $('#addForm');
  if (addForm) {
    $('#addSave').addEventListener('click', async () => {
      const entry = {
        id: 'user-' + Date.now(),
        stage: $('#addStage').value,
        name: $('#addRef').value || 'user model',
        engine: $('#addEngine').value || 'local',
        languages: $('#addLangs').value.split(',').map((s) => s.trim()).filter(Boolean),
        pair: null, size_mb: 0, devices: ['cpu'], min_ram_gb: 1, min_vram_gb: null,
        streaming: false, clone: false, installed: false, fit: 'good', recommended: false,
        why: 'User-added entry.', licence: { label: 'verify per source', url: '#' },
        source: { kind: 'local', repo: 'local', files: {} }, notes: 'Added via UI.',
      };
      try {
        const cur = JSON.parse(localStorage.getItem('bp_user_models') || '[]');
        cur.push(entry); localStorage.setItem('bp_user_models', JSON.stringify(cur));
        state.models.push(entry);
        renderLandscape();
        addForm.reset(); $('#addMsg').textContent = t('Saved (local only until the backend stores it).');
        showToast('Model entry saved locally', 'info');
      } catch (e) { $('#addMsg').textContent = 'Save failed: ' + e.message; }
    });
  }

  /* ---------- wiring ---------- */
  const srcSel = $('#srcLang'), tgtSel = $('#tgtLang');
  if (srcSel) srcSel.addEventListener('change', (e) => { state.src = e.target.value; loadPlan(); });
  if (tgtSel) tgtSel.addEventListener('change', (e) => { state.tgt = e.target.value; loadPlan(); });
  const swap = $('.swap');
  if (swap) swap.addEventListener('click', () => { const a = $('#srcLang').value, b = $('#tgtLang').value; $('#srcLang').value = b; $('#tgtLang').value = a; state.src = b; state.tgt = a; loadPlan(); });
  $('#landRefresh').addEventListener('click', async () => {
    try { await api('/catalog/refresh', { method: 'POST' }); showToast('Catalog refreshed', 'info'); loadCatalog(); }
    catch (e) { showToast('Refresh failed: ' + e.message, 'error'); }
  });
  $$('.tab').forEach((tab) => tab.addEventListener('click', (e) => {
    $$('.tab').forEach((t2) => t2.setAttribute('aria-pressed', 'false'));
    e.currentTarget.setAttribute('aria-pressed', 'true');
    state.tab = e.currentTarget.dataset.stage; renderLandscape();
  }));
  $('#profileTabs').addEventListener('click', (e) => {
    const b = e.target.closest('[data-profile]'); if (!b) return;
    state.profile = b.dataset.profile; loadCatalog();
  });
  $('#landSearch').addEventListener('input', (e) => { state.search = e.target.value; renderLandscape(); });
  $('#landLang').addEventListener('change', (e) => { state.landLang = e.target.value; renderLandscape(); });

  loadCatalog();
  loadRoute();

  /* back-to-live link: keep ?v token consistent */

  /* i18n hook: re-render text on language change */
  document.addEventListener('languagechange', () => { renderLanguages(); renderPlan(); renderLandscape(); });
});

function escapeHtml(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]); }
