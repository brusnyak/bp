(function () {
  'use strict';
  // Guest-first notice: who is signed in, and where the audio is processed.
  var hint = document.getElementById('accountHint');
  var link = document.getElementById('accountLink');
  var where = document.getElementById('whereRuns');
  if (!hint || !where) return;
  var T = function (s, v) { return window.i18n ? window.i18n.t(s, v) : s; };
  function render() {
    var token = null, guest = false, name = '';
    try { token = localStorage.getItem('userToken'); guest = localStorage.getItem('userGuest') === '1'; name = localStorage.getItem('userName') || ''; } catch (e) { /* storage blocked */ }
    var signedIn = token && !guest;
    hint.textContent = signedIn ? T('Signed in as {name}. Your voices are saved in your account.', { name: name }) : T('No account needed. Create one only to save your own voice.');
    if (link) link.hidden = !!signedIn;
    var local = ['localhost', '127.0.0.1', '[::1]'].indexOf(location.hostname) !== -1;
    where.textContent = local ? T('Runs on this computer: your audio does not leave it.') : T('Audio goes to the server at {host}. It is processed, not stored.', { host: location.host });
  }
  render();
  document.addEventListener('languagechange', render);
})();
