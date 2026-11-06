// Nav sign-in state: when a token is stored, "Sign in" links become "Sign out".
// Keys match auth.js: userToken, userName, userEmail. Storage blocked = no change.
(function () {
    'use strict';
    var name = null;
    try {
        if (!localStorage.getItem('userToken') || localStorage.getItem('userGuest') === '1') return;  // guests keep the Sign in link
        name = localStorage.getItem('userName') || localStorage.getItem('userEmail') || '';
    } catch (e) { return; }
    document.querySelectorAll('a[href="/ui/auth/auth.html"]').forEach(function (link) {
        link.textContent = 'Sign out';
        if (name) link.title = 'Signed in as ' + name;
        link.setAttribute('href', '#sign-out');
        link.addEventListener('click', function (event) {
            event.preventDefault();
            try {
                localStorage.removeItem('userToken');
                localStorage.removeItem('userName');
                localStorage.removeItem('userEmail');
                localStorage.removeItem('userGuest');
            } catch (e) { /* nothing to clear */ }
            window.location.reload();
        });
    });
})();
