/* Live page language controls, driven by what the server reports (ui/shared/caps.js). No language is listed here.
   The two <select> elements stay as the state live-script.js reads; this file fills them, keeps them valid, and
   puts the shared picker on top. ?from=xx&to=yy preselects a pair (used by the home page language list). */
(function () {
  'use strict';
  var inSel = document.getElementById('inputLanguageSelect');
  var outSel = document.getElementById('outputLanguageSelect');
  var fromBtn = document.getElementById('fromBtn');
  var toBtn = document.getElementById('toBtn');
  var swapBtn = document.getElementById('swapLangBtn');
  if (!inSel || !outSel || !window.caps || !window.LangPicker) return;

  var t = function (s) { return window.i18n ? window.i18n.t(s) : s; };
  var enName = function (code) { return caps.names(code, 'en').local; };
  var page = function () { return document.documentElement.lang || 'en'; };
  var data = null;

  function usable(src, tgt) {
    var r = data.route(src, tgt);
    return !!r && (r.tier === 'speech' || data.source === 'manifest');   // manifest = server unreachable, nothing is known to be ready
  }

  function option(code, disabled) {
    var o = document.createElement('option');
    o.value = code; o.disabled = !!disabled;
    o.textContent = enName(code) + ' (' + caps.names(code, code).native + ')';   // live-script.js reads the name before " ("
    return o;
  }

  function fillInput(keep) {
    inSel.textContent = '';
    data.sources().forEach(function (code) {
      inSel.appendChild(option(code, !data.targets(code).some(function (tg) { return usable(code, tg); })));
    });
    inSel.value = [].some.call(inSel.options, function (o) { return o.value === keep && !o.disabled; }) ? keep
      : ([].find.call(inSel.options, function (o) { return !o.disabled; }) || inSel.options[0] || {}).value;
  }

  function fillOutput(keep) {
    var all = [];
    data.routes.forEach(function (r) { if (all.indexOf(r.target) === -1) all.push(r.target); });
    outSel.textContent = '';
    all.forEach(function (code) { outSel.appendChild(option(code, !usable(inSel.value, code))); });
    outSel.value = [].some.call(outSel.options, function (o) { return o.value === keep && !o.disabled; }) ? keep
      : ([].find.call(outSel.options, function (o) { return !o.disabled; }) || outSel.options[0] || {}).value;
  }

  function label(btn, caption, code) {
    btn.textContent = '';
    var s = document.createElement('small'); s.textContent = caption;
    var n = document.createElement('span'); n.setAttribute('translate', 'no'); n.textContent = code ? caps.names(code, page()).native : '—';
    btn.append(s, n);
    btn.setAttribute('aria-label', caption + ': ' + (code ? caps.names(code, page()).local : t('none')));
  }

  function refreshUi() {
    label(fromBtn, t('You speak'), inSel.value);
    label(toBtn, t('You hear'), outSel.value);
    if (swapBtn) {
      var back = usable(outSel.value, inSel.value);
      swapBtn.disabled = !back;
      swapBtn.title = back ? t('Swap languages') : t('No ready route in the other direction');
    }
  }

  function items(role) {
    var cur = inSel.value;
    var codes = role === 'source' ? data.sources() : (function () { var a = []; data.routes.forEach(function (r) { if (a.indexOf(r.target) === -1) a.push(r.target); }); return a; })();
    return codes.map(function (code) {
      var tier = role === 'source' ? data.tierOf(code, 'source') : (data.route(cur, code) || {}).tier || null;
      var ok = role === 'source' ? data.targets(code).some(function (tg) { return usable(code, tg); }) : usable(cur, code);
      var n = caps.names(code, page());
      var group = tier ? t(caps.TIER_LABEL[tier]) : t('Not available from') + ' ' + caps.names(cur, page()).native;
      return { code: code, native: n.native, local: n.local, group: group, disabled: !ok, tier: tier || 'zz',
               note: ok ? '' : (tier === 'captions' ? t('No voice installed') : tier === 'setup' ? t('Needs setup') : '') };
    }).sort(function (a, b) { return (caps.TIER_ORDER[a.tier] == null ? 9 : caps.TIER_ORDER[a.tier]) - (caps.TIER_ORDER[b.tier] == null ? 9 : caps.TIER_ORDER[b.tier]) || a.native.localeCompare(b.native, page()); });
  }


  /* Starting pair, in order of trust: the link (?from=&to=), the last pair this browser used, the browser's own
     language list (navigator.languages), then the first route that is ready. Nothing is fixed to English. */
  function saved() { try { return JSON.parse(localStorage.getItem('livePair') || 'null') || {}; } catch (e) { return {}; } }
  function remember() { try { localStorage.setItem('livePair', JSON.stringify({ from: inSel.value, to: outSel.value })); } catch (e) { /* storage blocked */ } }
  function browserCodes() {
    var out = [];
    (navigator.languages || [navigator.language || '']).forEach(function (l) {
      l = String(l).toLowerCase();
      [l, l.split('-')[0]].forEach(function (c) { if (c && out.indexOf(c) === -1) out.push(c); });
    });
    return out;
  }
  function choosePair() {
    var q = new URLSearchParams(location.search), sv = saved(), bc = browserCodes();
    var ready = function (src) { return data.targets(src).filter(function (tg) { return usable(src, tg); }); };
    var sources = data.sources().filter(function (c) { return ready(c).length; });
    // Prefer English as the source language (demo route EN→SK/CS), then fall back to link/saved/browser/first
    var from = [q.get('from'), sv.from].concat(bc).filter(Boolean).find(function (c) { return sources.indexOf(c) !== -1; });
    if (!from && sources.indexOf('en') !== -1) from = 'en';
    if (!from) from = sources[0];
    var targets = from ? ready(from) : [];
    // Prefer Slovak as target when source is English (the demo route EN→SK)
    var preferredTo = (from === 'en' && targets.indexOf('sk') !== -1) ? 'sk' : null;
    var to = [q.get('to'), sv.to, preferredTo].concat(bc.filter(function (c) { return c !== from; })).filter(Boolean).find(function (c) { return targets.indexOf(c) !== -1; }) || targets[0];
    return { from: from, to: to };
  }

  function change(sel) { sel.dispatchEvent(new Event('change', { bubbles: true })); }

  caps.load().then(function (c) {
    data = c;
    if (!c.routes.length) return;   // server and manifest both unreachable: leave the static fallback options alone
    var pair = choosePair();
    fillInput(pair.from);
    fillOutput(pair.to);
    document.body.classList.add('lp-enhanced');
    refreshUi(); change(inSel); change(outSel);

    inSel.addEventListener('change', function () { fillOutput(outSel.value); refreshUi(); remember(); });
    outSel.addEventListener('change', function () { refreshUi(); remember(); });
    document.addEventListener('languagechange', refreshUi);

    LangPicker.attach(fromBtn, {
      title: t('You speak'), value: function () { return inSel.value; }, items: function () { return items('source'); },
      onPick: function (code) { inSel.value = code; change(inSel); change(outSel); }
    });
    LangPicker.attach(toBtn, {
      title: t('You hear'), value: function () { return outSel.value; }, items: function () { return items('target'); },
      onPick: function (code) { outSel.value = code; change(outSel); }
    });
  });
})();
