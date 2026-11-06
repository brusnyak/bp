/* One guided button for the Run step. It forwards to the real controls (Initialize, sound check, headphones
   confirmation, Start/Stop), which stay in the page, so every safety gate in live-script.js still applies.
   State comes from window.__live (set by live-script.js) and from the buttons themselves. */
(function () {
  'use strict';
  var go = document.getElementById('goBtn');
  var init = document.getElementById('initBtn');
  var start = document.getElementById('startStopBtn');
  var check = document.getElementById('soundCheckBtn');
  var iso = document.getElementById('isolationConfirmed');
  if (!go || !init || !start || !check || !iso) return;
  var t = function (s) { return window.i18n ? window.i18n.t(s) : s; };
  var N_ = function (x) { return x; };   // marks a string for scripts/i18n_tool.py
  var action = null;

  function state() {
    var s = window.__live || {};
    if (s.isStreaming) return { label: N_('Stop'), run: function () { start.click(); }, tone: 'stop' };
    if (s.starting) return { label: N_('Starting...'), disabled: true };
    if (!s.modelsReady) return init.disabled ? { label: N_('Preparing...'), disabled: true } : { label: N_('Prepare route'), run: function () { init.click(); } };
    if (!s.soundChecked) return check.disabled ? { label: N_('Checking sound...'), disabled: true } : { label: N_('Check sound'), run: function () { check.click(); } };
    if (!s.isolated) return { label: N_("I'm using headphones"), run: function () { iso.checked = true; iso.dispatchEvent(new Event('change', { bubbles: true })); } };
    return { label: N_('Start'), run: function () { start.click(); } };
  }

  function render() {
    var st = state();
    action = st.run || null;
    go.textContent = t(st.label);
    go.disabled = !!st.disabled;
    go.classList.toggle('btn-stop', st.tone === 'stop');
  }

  go.addEventListener('click', function () { if (action) action(); });
  document.addEventListener('livestate', render);
  document.addEventListener('languagechange', render);
  var mo = new MutationObserver(render);
  [init, start].forEach(function (b) { mo.observe(b, { attributes: true, attributeFilter: ['disabled', 'class'], childList: true, characterData: true, subtree: true }); });

  document.body.classList.add('guided');
  go.hidden = false;
  render();
})();
