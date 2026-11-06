/* Interface languages. English text is the key (gettext style): untranslated strings stay English.
   Dictionaries: /ui/i18n/<lang>.json  { "English text": "Translated text" }
   Plurals use the browser's Intl.PluralRules, the same API i18next relies on. */
(function () {
  'use strict';

  // Interface locales are discovered: /ui/i18n/index.json is generated from the dictionary files (scripts/ui_manifest.py).
  // English is built in; if the index cannot be read, the page stays English plus whatever dictionary loads by code.
  var LANGS = { en: 'English' };
  var META = { en: { code: 'en', dir: 'ltr', coverage: 1, machine_draft: false } };
  var SKIP = { SCRIPT: 1, STYLE: 1, CODE: 1, PRE: 1, TEXTAREA: 1, NOSCRIPT: 1 };
  var ATTRS = ['placeholder', 'title', 'aria-label', 'alt'];
  // Raw machine output (catalog _meta.raw_keys, e.g. Opus-MT) is not shown: add ?drafts=1 to preview it. LLM drafts (_meta.draft_keys) are shown and labelled.
  var DRAFTS = /[?&]drafts=1\b/.test(location.search) || (function () { try { return localStorage.getItem('i18nDrafts') === '1'; } catch (e) { return false; } })();
  var dict = {};
  var cur = 'en';
  var nodes = new WeakMap();   // text node -> { en, tr }
  var attrs = new WeakMap();   // element -> { attr: { en, tr } }
  var busy = false;

  function norm(s) { return String(s).replace(/\s+/g, ' ').trim(); }
  function lookup(en) { var d = dict[cur]; return (d && d[en]) || null; }

  function pick() {
    var q = new URLSearchParams(location.search).get('lang');
    if (q && LANGS[q]) return q;   // ?lang=sk makes a shareable link
    try { var saved = localStorage.getItem('lang'); if (saved && LANGS[saved]) return saved; } catch (e) { /* storage blocked */ }
    var list = navigator.languages || [navigator.language || 'en'];
    for (var i = 0; i < list.length; i++) {
      var code = String(list[i]).slice(0, 2).toLowerCase();
      if (LANGS[code]) return code;
    }
    return 'en';
  }

  function t(msg, vars) {
    var s = lookup(norm(msg)) || msg;
    if (vars) Object.keys(vars).forEach(function (k) { s = s.split('{' + k + '}').join(vars[k]); });
    return s;
  }

  // tp(3, { one: '{n} sentence', few: '{n} sentences', other: '{n} sentences' })
  function tp(n, forms) {
    var cat = 'other';
    try { cat = new Intl.PluralRules(cur).select(n); } catch (e) { /* old browser */ }
    var f = forms[cat] || forms.other;
    return t(f, { n: new Intl.NumberFormat(cur).format(n) });
  }

  function translateText(node) {
    if (node.parentElement && node.parentElement.closest('[translate="no"]')) return;   // language names and code samples stay as written
    var raw = node.nodeValue, rec = nodes.get(node), en, lead, trail;
    if (rec && raw === rec.tr) { en = rec.en; lead = rec.lead; trail = rec.trail; }   // our own value: keep the original whitespace
    else {                                                                           // new, or changed by the page: treat as English
      en = norm(raw);
      lead = (raw.match(/^\s*/) || [''])[0];
      trail = (raw.match(/\s*$/) || [''])[0];
    }
    if (!/[A-Za-z\u00C0-\u024F]/.test(en)) return;
    var tr = cur === 'en' ? null : lookup(en);
    if (!rec && !tr) return;                     // nothing to do for this node
    var out = lead + (tr ? tr.trim() : en) + trail;
    nodes.set(node, { en: en, tr: out, lead: lead, trail: trail });
    if (node.nodeValue !== out) node.nodeValue = out;
  }

  function translateAttrs(el) {
    var rec = attrs.get(el) || {};
    ATTRS.forEach(function (a) {
      if (!el.hasAttribute || !el.hasAttribute(a)) return;
      var v = el.getAttribute(a), r = rec[a], en = (r && v === r.tr) ? r.en : norm(v);
      var tr = cur === 'en' ? null : lookup(en);
      if (!r && !tr) return;
      var out = tr || en;
      rec[a] = { en: en, tr: out };
      if (v !== out) el.setAttribute(a, out);
    });
    attrs.set(el, rec);
  }

  var htmlEn = new WeakMap();
  function translateHtml(el) {
    var en = htmlEn.get(el);
    if (!en) { en = norm(el.innerHTML); htmlEn.set(el, en); }
    var tr = cur === 'en' ? null : lookup(en);
    el.innerHTML = tr || el.dataset.en || en;
    if (!tr && !el.dataset.en) el.dataset.en = en;
  }

  function apply(root) {
    if (!root) return;
    busy = true;
    try {
      if (root.querySelectorAll) root.querySelectorAll('[data-i18n]').forEach(translateHtml);
      var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT, {
        acceptNode: function (n) {
          if (n.nodeType === 1 && (SKIP[n.tagName] || n.hasAttribute('data-i18n') || n.getAttribute('translate') === 'no' || n.classList.contains('material-symbols-outlined'))) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      });
      var n = walker.currentNode;
      if (n.nodeType === 1) translateAttrs(n);
      while ((n = walker.nextNode())) {
        if (n.nodeType === 3) translateText(n); else translateAttrs(n);
      }
    } finally { busy = false; }
    var title = document.querySelector('title');
    if (title && title.firstChild) translateText(title.firstChild);
  }

  function load(lang) {
    if (lang === 'en' || dict[lang]) return Promise.resolve();
    return fetch('/ui/i18n/' + lang + '.json', { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (j) {
        var drop = (j && j._meta && j._meta.raw_keys) || [];
        if (!DRAFTS) drop.forEach(function (k) { delete j[k]; });
        delete j._meta;
        dict[lang] = j;
      })
      .catch(function () { dict[lang] = {}; });
  }

  function setLang(lang) {
    if (!LANGS[lang]) lang = 'en';
    return load(lang).then(function () {
      cur = lang;
      try { localStorage.setItem('lang', lang); } catch (e) { /* storage blocked */ }
      document.documentElement.lang = lang;
      apply(document.body);
      apply(document.head);
      document.documentElement.dir = (META[lang] && META[lang].dir) || 'ltr';
      document.dispatchEvent(new CustomEvent('languagechange', { detail: lang }));
    });
  }

  function nativeName(code) {
    try { var n = new Intl.DisplayNames([code], { type: 'language' }).of(code) || code; return n.charAt(0).toLocaleUpperCase(code) + n.slice(1); }
    catch (e) { return code; }
  }

  function loadIndex() {
    return fetch('/ui/i18n/index.json', { cache: 'no-cache' })
      .then(function (r) { return r.ok ? r.json() : { locales: [] }; })
      .catch(function () { return { locales: [] }; })
      .then(function (idx) {
        (idx.locales || []).forEach(function (l) { META[l.code] = l; if (!l.preview || DRAFTS) LANGS[l.code] = nativeName(l.code); });
      });
  }

  // The picker is shared with the speech language chooser; load it on demand so no page needs an extra tag.
  function loadPicker() {
    if (window.LangPicker) return Promise.resolve();
    return new Promise(function (resolve) {
      var css = document.createElement('link');
      css.rel = 'stylesheet'; css.href = '/ui/shared/lang-picker.css'; document.head.appendChild(css);
      var js = document.createElement('script');
      js.src = '/ui/shared/lang-picker.js'; js.onload = resolve; js.onerror = resolve; document.head.appendChild(js);
    });
  }

  function mount() {
    if (document.getElementById('lang-btn')) return;
    var btn = document.createElement('button');
    btn.type = 'button'; btn.id = 'lang-btn'; btn.className = 'lp-trigger lang-btn';
    btn.setAttribute('aria-label', 'Interface language');
    var g = document.createElement('span'); g.className = 'material-symbols-outlined'; g.setAttribute('aria-hidden', 'true'); g.textContent = 'language';
    var lab = document.createElement('span'); lab.className = 'lang-btn-label';
    btn.append(g, lab);
    function refresh() { lab.textContent = cur.toUpperCase(); btn.title = LANGS[cur] || cur; }
    refresh();
    document.addEventListener('languagechange', refresh);
    var theme = document.getElementById('theme-toggle');
    var host = document.querySelector('.nav-column.right-column .theme-toggle-wrapper, .bar-actions #theme-toggle, .live-bar #theme-toggle, .header .theme-toggle-wrapper, .lang-slot')
      || theme;
    if (host && host.classList && host.classList.contains('lang-slot')) host.appendChild(btn);
    else if (host && host.parentNode) host.parentNode.insertBefore(btn, host);
    else { btn.style.cssText = 'position:fixed;top:0.75rem;right:0.75rem;z-index:60'; document.body.appendChild(btn); }
    if (Object.keys(LANGS).length < 2) { btn.hidden = true; return; }   // only English known: nothing to choose
    loadPicker().then(function () {
      if (!window.LangPicker) return;
      LangPicker.attach(btn, {
        title: t('Interface language'),
        value: function () { return cur; },
        items: function () {
          return Object.keys(LANGS).sort(function (a, b) { return a === 'en' ? -1 : b === 'en' ? 1 : LANGS[a].localeCompare(LANGS[b]); })
            .map(function (code) {
              var m = META[code] || {}, note = '';
              if (m.reviewed != null && m.reviewed < 0.5) note = t('Machine draft');
              else if (m.coverage != null && m.coverage < 1) note = Math.round(m.coverage * 100) + '%';
              var en = ''; try { en = new Intl.DisplayNames(['en'], { type: 'language' }).of(code) || ''; } catch (e) { /* keep empty */ }
              return { code: code, native: LANGS[code], local: en, note: note };
            });
        },
        onPick: function (code) { setLang(code); }
      });
    });
  }

  function start() {
    var style = document.createElement('style');
    style.textContent = '.lang-btn{padding:0 .7rem;min-height:40px;font-size:.85rem}.lang-btn .material-symbols-outlined{font-size:1.1rem}';
    document.head.appendChild(style);
    loadIndex().then(boot);
  }

  function boot() {
    mount();
    // Strings the page's own scripts write later (status lines, errors) are translated as they appear.
    new MutationObserver(function (muts) {
      if (busy || cur === 'en') return;
      muts.forEach(function (m) {
        if (m.type === 'characterData') translateText(m.target);
        else m.addedNodes.forEach(function (n) { if (n.nodeType === 3) translateText(n); else if (n.nodeType === 1) apply(n); });
      });
    }).observe(document.body, { childList: true, subtree: true, characterData: true });
    setLang(pick());
  }

  window.i18n = { t: t, tp: tp, meta: function (c) { return META[c || cur] || null; }, setLang: setLang, lang: function () { return cur; }, languages: LANGS };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
