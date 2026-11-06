/* Language picker: one component for interface language and for speech From/To.
   LangPicker.open({ title, items, value, anchor, onPick })
     items: [{ code, native, local, group, badge, note, disabled }]   (group = heading text, items arrive pre-sorted)
   Popover under the trigger on wide screens, bottom sheet on phones. Search matches native name, English name, code,
   ignoring case and diacritics ("slov" finds Slovenčina, "cesky" finds Čeština). */
(function () {
  'use strict';

  var open = null;

  function fold(s) { return String(s).normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase(); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function icon(name, cls) { var s = el('span', 'material-symbols-outlined ' + (cls || ''), name); s.setAttribute('aria-hidden', 'true'); return s; }
  function t(s) { return window.i18n ? window.i18n.t(s) : s; }

  function close(restore) {
    if (!open) return;
    var o = open; open = null;
    document.removeEventListener('keydown', o.onKey, true);
    document.documentElement.style.overflow = o.prevOverflow;
    o.scrim.classList.remove('open');
    var done = function () { o.scrim.remove(); };
    var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    reduce ? done() : setTimeout(done, 200);
    if (restore !== false && o.anchor && o.anchor.focus) o.anchor.focus();
  }

  function show(opts) {
    close(false);
    var items = opts.items || [], idx = -1, query = '';
    var scrim = el('div', 'lp-scrim');
    var box = el('div', 'lp');
    box.setAttribute('role', 'dialog'); box.setAttribute('aria-modal', 'true'); box.setAttribute('aria-label', opts.title);
    box.tabIndex = -1;
    box.appendChild(el('div', 'lp-grab'));

    var head = el('div', 'lp-head');
    var h = el('h2', '', opts.title);
    var x = el('button', 'lp-close'); x.type = 'button'; x.setAttribute('aria-label', t('Close')); x.appendChild(icon('close'));
    head.append(h, x);
    box.appendChild(head);

    var search = el('div', 'lp-search'), input = el('input');
    input.type = 'search'; input.placeholder = t('Search languages'); input.setAttribute('aria-label', t('Search languages'));
    input.autocomplete = 'off'; input.autocapitalize = 'off'; input.spellcheck = false;
    input.setAttribute('role', 'combobox'); input.setAttribute('aria-expanded', 'true'); input.setAttribute('aria-controls', 'lp-list');
    if (items.length > 7) { search.appendChild(input); box.appendChild(search); }

    var list = el('ul', 'lp-list'); list.id = 'lp-list'; list.setAttribute('role', 'listbox');
    box.appendChild(list);
    scrim.appendChild(box);

    function visible() {
      var q = fold(query.trim());
      return items.filter(function (it) { return !q || fold([it.native, it.local, it.code].join(' ')).indexOf(q) !== -1; });
    }

    function render() {
      list.textContent = '';
      var rows = visible(), lastGroup = null, n = 0;
      if (!rows.length) { list.appendChild(el('li', 'lp-empty', t('No language matches.'))); return; }
      rows.forEach(function (it) {
        if (it.group && it.group !== lastGroup) {
          var g = el('li', 'lp-group', it.group); g.setAttribute('role', 'presentation'); list.appendChild(g); lastGroup = it.group;
        }
        var li = el('li', 'lp-item'); li.id = 'lp-o' + n; li.setAttribute('role', 'option'); li.dataset.i = n;
        li.setAttribute('aria-selected', String(it.code === opts.value));
        if (it.disabled) li.setAttribute('aria-disabled', 'true');
        var name = el('div', 'lp-name'); name.setAttribute('translate', 'no'); name.appendChild(el('b', '', it.native));
        var sub = [it.local !== it.native ? it.local : '', it.note || ''].filter(Boolean).join(' · ');
        if (sub) name.appendChild(el('span', '', sub));
        li.appendChild(name);
        if (it.badge) li.appendChild(el('span', 'lp-badge', it.badge));
        li.appendChild(icon('check', 'lp-check'));
        li._item = it; list.appendChild(li); n++;
      });
      setActive(Math.max(0, rows.findIndex(function (it) { return it.code === opts.value && !it.disabled; })), false);
    }

    function options() { return list.querySelectorAll('.lp-item'); }
    function setActive(i, scroll) {
      var os = options(); if (!os.length) return;
      i = Math.max(0, Math.min(os.length - 1, i)); idx = i;
      os.forEach(function (o, k) { o.classList.toggle('active', k === i); });
      input.setAttribute('aria-activedescendant', os[i].id);
      if (scroll !== false) { var o = os[i], top = o.offsetTop, bot = top + o.offsetHeight; if (top < list.scrollTop) list.scrollTop = top - 4; else if (bot > list.scrollTop + list.clientHeight) list.scrollTop = bot - list.clientHeight + 4; }
    }
    function pick(li) {
      if (!li || !li._item || li._item.disabled) return;
      close(); if (opts.onPick) opts.onPick(li._item.code, li._item);
    }

    list.addEventListener('click', function (e) { pick(e.target.closest('.lp-item')); });
    list.addEventListener('pointermove', function (e) { var li = e.target.closest('.lp-item'); if (li && +li.dataset.i !== idx) setActive(+li.dataset.i, false); });
    input.addEventListener('input', function () { query = input.value; render(); setActive(0); });
    x.addEventListener('click', function () { close(); });
    scrim.addEventListener('pointerdown', function (e) { if (e.target === scrim) close(); });

    var onKey = function (e) {
      if (e.key === 'Escape') { e.preventDefault(); close(); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); setActive(idx + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(idx - 1); }
      else if (e.key === 'Enter' && document.activeElement !== x) { e.preventDefault(); pick(options()[idx]); }
      else if (e.key === 'Tab') {
        var f = [].slice.call(box.querySelectorAll('input, button')).filter(function (n) { return n.offsetParent; });
        if (!f.length) return;
        var first = f[0], last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };
    document.addEventListener('keydown', onKey, true);

    open = { scrim: scrim, anchor: opts.anchor, onKey: onKey, prevOverflow: document.documentElement.style.overflow };
    document.documentElement.style.overflow = 'hidden';
    document.body.appendChild(scrim);

    if (opts.anchor && window.matchMedia && matchMedia('(min-width: 720px)').matches) {
      var r = opts.anchor.getBoundingClientRect(), w = 352;
      box.style.left = Math.max(8, Math.min(r.left, innerWidth - w - 8)) + 'px';
      box.style.top = Math.min(r.bottom + 6, innerHeight - 200) + 'px';
      box.style.maxHeight = Math.max(240, innerHeight - r.bottom - 24) + 'px';
    }
    render();
    requestAnimationFrame(function () {
      scrim.classList.add('open');
      var touch = window.matchMedia && matchMedia('(pointer: coarse)').matches;   // no keyboard pop-up on phones
      (items.length > 7 && !touch ? input : box).focus({ preventScroll: true });
      var sel = list.querySelector('[aria-selected="true"]'); if (sel) list.scrollTop = Math.max(0, sel.offsetTop - list.clientHeight / 2 + sel.offsetHeight / 2);
    });
  }

  /* Wire a trigger button: items are built when it opens, so late-loaded capabilities are always current. */
  function attach(btn, cfg) {
    btn.setAttribute('aria-haspopup', 'dialog');
    btn.addEventListener('click', function () {
      Promise.resolve(cfg.items()).then(function (items) {
        show({ title: cfg.title, items: items, value: cfg.value(), anchor: btn, onPick: cfg.onPick });
      });
    });
  }

  window.LangPicker = { open: show, close: close, attach: attach };
})();
