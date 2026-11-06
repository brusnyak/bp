/* Home page behaviour. Each block is independent and fails soft: a missing element or blocked storage never stops the rest. */
document.addEventListener('DOMContentLoaded', () => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const t = (s) => (window.i18n ? window.i18n.t(s) : s);
  const N_ = (s) => s;   // marks a string for extraction (scripts/i18n_tool.py); translated later with t()
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- bar, menu ---------- */
  const bar = $('#bar'), menu = $('#menu'), menuBtn = $('#menu-btn');
  const onScroll = () => bar.classList.toggle('scrolled', scrollY > 8);
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  const setMenuLabel = () => menuBtn.setAttribute('aria-label', menu.hidden ? t('Open menu') : t('Close menu'));
  const setMenu = (open) => {
    menu.hidden = !open;
    menuBtn.setAttribute('aria-expanded', String(open));
    setMenuLabel();
    menuBtn.firstElementChild.textContent = open ? 'close' : 'menu';
  };
  menuBtn.addEventListener('click', () => setMenu(menu.hidden));
  menu.addEventListener('click', (e) => { if (e.target.closest('a')) setMenu(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !menu.hidden) { setMenu(false); menuBtn.focus(); } });
  document.addEventListener('pointerdown', (e) => { if (!menu.hidden && !bar.contains(e.target)) setMenu(false); });

  /* ---------- section rail: active section, hidden over the hero ---------- */
  const rail = $('#rail'), secs = $$('main > section');
  const railLinks = $$('a', rail);
  if ('IntersectionObserver' in window) {
    const seen = new Map();
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => seen.set(en.target.id, en.intersectionRatio));
      const top = [...seen].sort((a, b) => b[1] - a[1])[0];
      const id = top && top[1] > 0 ? top[0] : '';
      rail.classList.toggle('show', !!id && id !== 'hero');
      railLinks.forEach((a) => a.classList.toggle('active', a.dataset.target === id));
    }, { threshold: [0, .25, .5, .75, 1] });
    secs.forEach((s) => io.observe(s));
  }

  /* ---------- reveal once ---------- */
  const rv = $$('.sec h2, .sec .sub, .panel, .stage-card, .measure, .status > div, .faq-item');
  rv.forEach((e) => e.classList.add('rv'));
  if (reduce || !('IntersectionObserver' in window)) rv.forEach((e) => e.classList.add('in'));
  else {
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { rootMargin: '0px 0px -8% 0px' });
    rv.forEach((e) => io.observe(e));
  }
  const bars = $('.bars');
  if (bars) {
    if (reduce || !('IntersectionObserver' in window)) bars.classList.add('in');
    else { const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { bars.classList.add('in'); io.disconnect(); } }, { threshold: .4 }); io.observe(bars); }
  }

  /* ---------- hero voice line: English words, a wave, Slovak words ---------- */
  const en = $('#vl-en'), sk = $('#vl-sk'), path = $('#vl-path'), replay = $('#vl-replay');
  let timers = [];
  const clear = () => { timers.forEach(clearTimeout); timers = []; };
  const later = (fn, ms) => timers.push(setTimeout(fn, ms));
  function playVoiceLine() {
    clear();
    const a = en.dataset.text, b = sk.dataset.text;
    if (reduce) { en.textContent = a; sk.textContent = b; path.style.strokeDashoffset = 0; return; }
    en.textContent = ''; sk.textContent = ''; path.setAttribute('pathLength', '1');
    path.style.transition = 'none'; path.style.strokeDashoffset = 1; void path.getBoundingClientRect();
    let at = 150;
    [...a].forEach((_, i) => later(() => { en.textContent = a.slice(0, i + 1); }, at + i * 38));
    at += a.length * 38 + 150;
    later(() => { path.style.transition = 'stroke-dashoffset 700ms cubic-bezier(.2,.8,.2,1)'; path.style.strokeDashoffset = 0; }, at);
    at += 600;
    b.split(' ').forEach((_, i, w) => later(() => { sk.textContent = w.slice(0, i + 1).join(' '); }, at + i * 140));
  }
  if (en && sk && path) {
    playVoiceLine();
    replay && replay.addEventListener('click', playVoiceLine);
  }

  /* Try your voice: real transcription + translation via WebSocket (not just local visualization). */
  const micBtn = $('#vl-mic'), micLabel = $('#vl-mic-label'), micStatus = $('#vl-status');
  let micWs = null, micActive = false, micStream = null, micCtx = null, micWorklet = null, micRaf = 0, micStop = 0;
  const restDef = path ? path.getAttribute('d') : '';
  function stopMic() {
    micActive = false;
    cancelAnimationFrame(micRaf); clearTimeout(micStop);
    if (micStream) micStream.getTracks().forEach((tr) => tr.stop());
    if (micWorklet) { try { micWorklet.disconnect(); } catch {} micWorklet = null; }
    if (micCtx) { try { micCtx.close(); } catch {} micCtx = null; }
    micStream = micCtx = null;
    if (micWs) { try { micWs.close(); } catch {} micWs = null; }
    if (!micBtn) return;
    micBtn.setAttribute('aria-pressed', 'false'); micLabel.textContent = t('Try your voice');
    path.setAttribute('d', restDef); path.style.strokeDasharray = ''; playVoiceLine();
    micStatus.textContent = '';
    $('#vl-en').textContent = $('#vl-sk').textContent = '';
  }
  async function startMic() {
    micActive = true;
    try {
      micStatus.textContent = t('Starting session…');
      const cfg = await (await fetch('/api/auth/config')).json();
      if (!cfg.guest_enabled) { micStatus.textContent = t('Guest sessions not enabled on this server.'); micActive = false; return; }
      const r = await fetch('/api/guest', { method: 'POST' });
      if (!r.ok) { micStatus.textContent = t('Could not start a guest session.'); micActive = false; return; }
      const d = await r.json();
      const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      micWs = new WebSocket(`${scheme}//${window.location.host}/ws?token=${encodeURIComponent(d.token)}`);
      await new Promise((resolve, reject) => {
        micWs.onopen = () => resolve();
        micWs.onerror = () => reject(new Error('WebSocket failed to open'));
        setTimeout(() => reject(new Error('WebSocket timeout')), 5000);
      });
      micWs.onmessage = (event) => {
        if (typeof event.data !== 'string') return;
        const msg = JSON.parse(event.data);
        if (msg.type === 'transcription_result') {
          $('#vl-en').textContent = msg.transcribed || '';
        } else if (msg.type === 'translation_result') {
          $('#vl-sk').textContent = msg.translated || '';
        } else if (msg.type === 'error') {
          micStatus.textContent = msg.message || 'Error';
        }
      };
      micWs.send(JSON.stringify({ type: 'join', role: 'speaker', source_lang: 'en', target_lang: 'sk', voice: '', tts_engine: 'piper' }));
      micWs.send(JSON.stringify({ type: 'start' }));
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const source = micCtx.createMediaStreamSource(micStream);
      await micCtx.audioWorklet.addModule('/ui/live-speech/audio-processor.js');
      micWorklet = new AudioWorkletNode(micCtx, 'audio-processor');
      source.connect(micWorklet);
      micWorklet.connect(micCtx.destination);
      micWorklet.port.onmessage = (event) => {
        if (micWs && micWs.readyState === WebSocket.OPEN) micWs.send(event.data.buffer);
      };
      const an = micCtx.createAnalyser(); an.fftSize = 512;
      source.connect(an);
      const buf = new Uint8Array(an.fftSize); let level = 0, phase = 0;
      path.removeAttribute('pathLength'); path.style.transition = 'none'; path.style.strokeDasharray = 'none'; path.style.strokeDashoffset = 0;
      micBtn.setAttribute('aria-pressed', 'true'); micLabel.textContent = t('Stop');
      micStatus.textContent = t('Say a sentence. It will be transcribed and translated.');
      $('#vl-en').textContent = t('Listening…'); $('#vl-sk').textContent = '';
      const frame = () => {
        if (!micActive) return;
        an.getByteTimeDomainData(buf);
        let sum = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
        level += (Math.min(1, Math.sqrt(sum / buf.length) * 5) - level) * .25;
        phase += .12 + level * .25;
        let d = '';
        for (let i = 0; i <= 64; i++) {
          const x = i * 600 / 64, taper = Math.sin(Math.PI * i / 64);
          const y = 30 + Math.sin(i * .5 - phase) * (3 + level * 24) * taper;
          d += (i ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1) + ' ';
        }
        path.setAttribute('d', d);
        micRaf = requestAnimationFrame(frame);
      };
      frame();
      micStop = setTimeout(stopMic, 30000);
    }
    catch (e) { micStatus.textContent = t('Microphone not available. The live page can still be tried.'); micActive = false; console.error(e); }
  }
  if (micBtn && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    micBtn.addEventListener('click', () => (micActive ? stopMic() : startMic()));
  } else if (micBtn) { $('.vl-try').hidden = true; }

  /* ---------- tabs (demo sentences, get-started ways): roving tabindex, arrow keys ---------- */
  function tabs(listId, onSelect) {
    const list = $(listId); if (!list) return;
    const btns = $$('[role="tab"]', list);
    const select = (i, focus) => {
      btns.forEach((b, k) => { b.setAttribute('aria-selected', String(k === i)); b.tabIndex = k === i ? 0 : -1; });
      if (focus) btns[i].focus();
      onSelect(i, btns[i]);
    };
    btns.forEach((b, i) => b.addEventListener('click', () => select(i)));
    list.addEventListener('keydown', (e) => {
      const cur = btns.findIndex((b) => b.getAttribute('aria-selected') === 'true');
      const next = { ArrowRight: cur + 1, ArrowLeft: cur - 1, Home: 0, End: btns.length - 1 }[e.key];
      if (next == null) return;
      e.preventDefault(); select((next + btns.length) % btns.length, true);
    });
    select(0);
  }

  const SENTENCES = [
    { heard: 'Good morning, and welcome to this live demonstration of real time speech translation.', out: 'so svojou cestou vysoko hore.', ok: true, verdict: N_('Recognised correctly, so the translation is right.') },
    { heard: 'I am speaking English, and you will hear my own voice speaking Slovak within seconds.', spoken: 'Slovak', out: 'Jeho priatelia hovoria, že hľadá časť zlata.', verdict: N_('One word misheard; the translation followed it.') },
    { heard: 'The system listens through the microphone and detects when speech begins and ends.', spoken: 'prism', out: 'Správať sa ako väzenie a vytvoriť dúhu.', verdict: N_('The recognizer misheard one word and the translation followed it.') },
  ];
  let curSentence = 0;
  function renderSentence() {
    const s = SENTENCES[curSentence];
    $('#demo-heard').textContent = '“' + s.heard + '”';
    $('#demo-spoken').textContent = s.spoken ? t('Spoken:') + ' “' + s.spoken + '”' : '';
    $('#demo-out').textContent = '“' + s.out + '”';
    $('#demo-verdict').textContent = t(s.verdict);
  }
  tabs('#demo-tabs', (i, btn) => {
    curSentence = i; renderSentence();
    $('#demo-panel').setAttribute('aria-labelledby', btn.id);
  });

  const WAYS = ['s-0', 's-1', 's-2'];
  tabs('#start-tabs', (i) => WAYS.forEach((id, k) => { $('#' + id).hidden = k !== i; }));

  /* ---------- pipeline stepper ---------- */
  const STAGES = [
    [N_('Microphone'), N_('Your browser sends 16 kHz audio over a WebSocket with your session token.'), ''],
    [N_('Voice activity'), N_('A segment closes after a 0.25 s pause or 3 s of speech, so the system answers while you keep talking.'), N_('Waits for your pause')],
    [N_('Speech to text'), N_('faster-whisper (int8) turns the segment into words, which also become the source captions.'), N_('about 0.4 s')],
    [N_('Translation'), N_('Opus-MT through CTranslate2 translates the sentence.'), N_('about 0.1 s')],
    [N_('Speech'), N_('Piper speaks one clip per clause. Cloning your voice (OmniVoice, GPU) is a research option.'), N_('about 0.1 s')],
    [N_('Playback'), N_('Clips play in order on the chosen device. Playback speeds up only while it is behind.'), N_('First sound 0.6–0.8 s after you pause')],
  ];
  const stageBtns = $$('#stages button');
  let curStage = 0;
  function showStage(i) {
    curStage = i;
    stageBtns.forEach((b, k) => b.setAttribute('aria-pressed', String(k === i)));
    $('#stage-title').textContent = (i + 1) + '. ' + t(STAGES[i][0]);
    $('#stage-text').textContent = t(STAGES[i][1]);
    $('#stage-time').textContent = STAGES[i][2] ? t(STAGES[i][2]) : '';
    $('#stage-fill').style.transform = 'scaleX(' + ((i + 1) / STAGES.length) + ')';
  }
  stageBtns.forEach((b, i) => b.addEventListener('click', () => showStage(i)));
  if (stageBtns.length) showStage(0);
  document.addEventListener('languagechange', () => { if (stageBtns.length) showStage(curStage); renderSentence(); setMenuLabel(); renderLangs(); });

  /* ---------- copy buttons ---------- */
  $$('.copy').forEach((btn) => btn.addEventListener('click', async () => {
    const src = document.getElementById(btn.dataset.copy), label = $('.copy-label', btn);
    try { await navigator.clipboard.writeText(src.textContent); label.textContent = t('Copied'); }
    catch (e) { label.textContent = t('Select and copy'); }
    setTimeout(() => { label.textContent = t('Copy'); }, 1600);
  }));

  /* ---------- demo video: thumbnail first, player only on click ---------- */
  $$('.demo-video[data-yt]').forEach((box) => {
    const btn = $('.demo-play', box);
    if (!btn) return;
    btn.addEventListener('click', () => {
      const f = document.createElement('iframe');
      f.src = 'https://www.youtube-nocookie.com/embed/' + box.dataset.yt + '?autoplay=1&rel=0';
      f.title = 'Real-Time Speech Translation demo';
      f.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
      f.referrerPolicy = 'strict-origin-when-cross-origin';
      f.allowFullscreen = true;
      btn.replaceWith(f);
    });
  });

  /* ---------- machine-translated locale: invite corrections ---------- */
  const improve = $('#improve');
  function renderImprove() {
    if (!improve || !window.i18n) return;
    const m = i18n.meta(), lang = i18n.lang();
    improve.hidden = !(m && m.drafts > 0 && lang !== 'en');
    improve.href = '/ui/extra/feedback.html?type=translation&lang=' + encodeURIComponent(lang) + '&page=' + encodeURIComponent(location.pathname);
  }
  document.addEventListener('languagechange', renderImprove);

  /* ---------- languages: whatever this server reports, nothing listed by hand ---------- */
  let capsData = null;
  const MAX_CHIPS = 7;
  function renderLangs() {
    const c = capsData; if (!c || !c.routes.length) return;
    const page = document.documentElement.lang || 'en';
    const codes = [...new Set(c.routes.flatMap((r) => [r.source, r.target]))];
    const tierOf = (code) => [c.tierOf(code, 'source'), c.tierOf(code, 'target')].filter(Boolean).sort((a, b) => caps.TIER_ORDER[a] - caps.TIER_ORDER[b])[0];
    const rows = codes.map((code) => ({ code, tier: tierOf(code), n: caps.names(code, page) }))
      .sort((a, b) => caps.TIER_ORDER[a.tier] - caps.TIER_ORDER[b.tier] || a.n.native.localeCompare(b.n.native, page));
    const speech = c.routes.filter((r) => r.tier === 'speech').length;
    const tp = (n, f) => (window.i18n ? window.i18n.tp(n, f) : f.other.replace('{n}', n));
    $('#langs-title').textContent = c.source === 'manifest'
      ? codes.length + ' ' + t('languages declared') + ' \u00B7 ' + t('server not checked')
      : tp(codes.length, { one: N_('{n} language on this server'), other: N_('{n} languages on this server'), few: N_('{n} languages on this server [few]'), many: N_('{n} languages on this server [many]') })
        + ' \u00B7 ' + tp(speech, { one: N_('{n} route with speech'), other: N_('{n} routes with speech'), few: N_('{n} routes with speech [few]'), many: N_('{n} routes with speech [many]') });
    const ul = $('#lang-chips'); ul.textContent = '';
    rows.slice(0, MAX_CHIPS).forEach((r) => {
      const li = document.createElement('li'), a = document.createElement('a');
      a.className = 'chip-lang'; a.setAttribute('translate', 'no');
      a.href = '/ui/live-speech/live.html?' + (c.tierOf(r.code, 'source') ? 'from=' : 'to=') + encodeURIComponent(r.code);
      const dot = document.createElement('i'); dot.className = 'dot ' + (r.tier === 'speech' ? 'ok' : r.tier === 'captions' ? 'warn' : 'no'); dot.title = t(caps.TIER_LABEL[r.tier]);
      a.append(dot, r.n.native); li.appendChild(a); ul.appendChild(li);
    });
    if (rows.length > MAX_CHIPS) {
      const li = document.createElement('li'), b = document.createElement('button');
      b.type = 'button'; b.className = 'chip-lang chip-all'; b.textContent = t('All') + ' ' + rows.length; li.appendChild(b); ul.appendChild(li);
      LangPicker.attach(b, {
        title: t('Languages'), value: () => '',
        items: () => rows.map((r) => ({ code: r.code, native: r.n.native, local: r.n.local, group: t(caps.TIER_LABEL[r.tier]), tier: r.tier,
          note: [c.tierOf(r.code, 'source') ? t('Speak') : '', c.tierOf(r.code, 'target') ? t('Hear') : ''].filter(Boolean).join(' \u00B7 ') })),
        onPick: (code) => { location.href = '/ui/live-speech/live.html?' + (c.tierOf(code, 'source') ? 'from=' : 'to=') + encodeURIComponent(code); }
      });
    }
    $('#langs').hidden = false;
  }
  if (window.caps) caps.load().then((c) => { if (c.routes.length) { capsData = c; renderLangs(); } });
});
