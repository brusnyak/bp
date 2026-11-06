// Sign-in / registration for the live page. Endpoints and storage keys are unchanged:
// POST /api/login, POST /api/register, GET /api/auth/config, POST /api/auth/google;
// localStorage keys userToken, userName, userEmail.
(function () {
    'use strict';
    var ALLOWED_NEXT = ['/ui/live-speech/live.html', '/ui/enroll/enroll.html'];
    var EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    var MIN_PASSWORD = 8;

    var $ = function (id) { return document.getElementById(id); };
    var message = $('auth-message');
    var forms = { login: $('panel-login'), register: $('panel-register') };
    var tabs = { login: $('tab-login'), register: $('tab-register') };

    function nextUrl() {
        var next = new URLSearchParams(window.location.search).get('next');
        return ALLOWED_NEXT.indexOf(next) !== -1 ? next : ALLOWED_NEXT[0];
    }

    function storeSession(data) {
        try {
            localStorage.setItem('userToken', data.token);
            localStorage.setItem('userName', data.username || '');
            localStorage.setItem('userEmail', data.email || '');
            if (data.guest) localStorage.setItem('userGuest', '1'); else localStorage.removeItem('userGuest');
            return true;
        } catch (e) { return false; }
    }

    function readSession() {
        try { if (localStorage.getItem('userGuest') === '1') return null; return localStorage.getItem('userToken') ? (localStorage.getItem('userName') || localStorage.getItem('userEmail') || 'your account') : null; }
        catch (e) { return null; }
    }

    function clearSession() {
        try {
            localStorage.removeItem('userToken');
            localStorage.removeItem('userName');
            localStorage.removeItem('userEmail');
            localStorage.removeItem('userGuest');
        } catch (e) { /* storage blocked: nothing to clear */ }
    }

    function setMessage(text, kind) {
        message.textContent = text || '';
        message.className = 'auth-message' + (kind ? ' ' + kind : '');
    }

    // The backend returns {detail} for HTTP errors and {message} on success.
    function errorText(status, body) {
        if (status === 429) return 'Too many attempts. Wait a minute and try again.';
        if (status === 501) return 'Google sign-in is not set up on this server yet.';
        if (body && typeof body.detail === 'string') return body.detail;
        if (status >= 500) return 'The server had a problem. Try again in a moment.';
        return 'Something went wrong. Check the details and try again.';
    }

    async function postJson(url, payload) {
        var response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        var body = null;
        try { body = await response.json(); } catch (e) { body = null; }
        return { ok: response.ok, status: response.status, body: body };
    }

    function setField(input, errorEl, text) {
        if (text) {
            errorEl.textContent = text;
            errorEl.hidden = false;
            input.setAttribute('aria-invalid', 'true');
        } else {
            errorEl.textContent = '';
            errorEl.hidden = true;
            input.removeAttribute('aria-invalid');
        }
        return !text;
    }

    function fieldsFor(form) {
        var out = {};
        Array.prototype.forEach.call(form.querySelectorAll('input[name]'), function (input) {
            out[input.name] = input;
        });
        return out;
    }

    function validate(kind) {
        var form = forms[kind];
        var f = fieldsFor(form);
        var ok = true;
        if (kind === 'register') {
            ok = setField(f.username, $('register-username-error'),
                f.username.value.trim() ? '' : 'Enter a username.') && ok;
        }
        var email = f.email.value.trim();
        ok = setField(f.email, $(kind + '-email-error'),
            !email ? 'Enter your email address.' : (EMAIL_RE.test(email) ? '' : 'This email address does not look right.')) && ok;
        var pw = f.password.value;
        var pwError;
        if (!pw) pwError = 'Enter your password.';
        else if (kind === 'register' && pw.length < MIN_PASSWORD) pwError = 'Use at least ' + MIN_PASSWORD + ' characters.';
        ok = setField(f.password, $(kind + '-password-error'), pwError) && ok;
        return ok;
    }

    function setBusy(button, busy, idleText, busyText) {
        button.disabled = busy;
        button.setAttribute('aria-busy', busy ? 'true' : 'false');
        button.innerHTML = busy ? '<span class="spinner" aria-hidden="true"></span>' + busyText : idleText;
    }

    function showTab(kind, focus) {
        ['login', 'register'].forEach(function (k) {
            var on = k === kind;
            tabs[k].setAttribute('aria-selected', String(on));
            tabs[k].tabIndex = on ? 0 : -1;
            forms[k].hidden = !on;
        });
        if (focus) tabs[kind].focus();
        setMessage('');
    }

    function showSignedIn() {
        var name = readSession();
        var signedIn = $('signed-in');
        var formsBox = $('auth-forms');
        if (name) {
            $('whoami').textContent = name;
            signedIn.hidden = false;
            formsBox.hidden = true;
        } else {
            signedIn.hidden = true;
            formsBox.hidden = false;
        }
    }

    // Tabs: click, arrow keys and Home/End, per the WAI-ARIA tabs pattern.
    tabs.login.addEventListener('click', function () { showTab('login'); });
    tabs.register.addEventListener('click', function () { showTab('register'); });
    [tabs.login, tabs.register].forEach(function (tab) {
        tab.addEventListener('keydown', function (event) {
            var order = ['login', 'register'];
            var i = order.indexOf(event.currentTarget === tabs.login ? 'login' : 'register');
            if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
                event.preventDefault();
                showTab(order[(i + 1) % 2], true);
            } else if (event.key === 'Home') { event.preventDefault(); showTab('login', true); }
            else if (event.key === 'End') { event.preventDefault(); showTab('register', true); }
        });
    });

    // Show / hide password.
    Array.prototype.forEach.call(document.querySelectorAll('.pw-toggle'), function (btn) {
        btn.addEventListener('click', function () {
            var input = $(btn.getAttribute('aria-controls'));
            var show = input.type === 'password';
            input.type = show ? 'text' : 'password';
            btn.setAttribute('aria-pressed', String(show));
            btn.textContent = show ? 'Hide' : 'Show';
            btn.setAttribute('aria-label', (show ? 'Hide' : 'Show') + ' password');
        });
    });

    // Clear a field error as soon as the user edits the field.
    Array.prototype.forEach.call(document.querySelectorAll('.auth-form input'), function (input) {
        input.addEventListener('input', function () {
            var err = document.getElementById(input.getAttribute('aria-describedby').split(' ').filter(function (id) { return /error$/.test(id); })[0] || '');
            if (err && !err.hidden) { err.hidden = true; input.removeAttribute('aria-invalid'); }
        });
    });

    forms.login.addEventListener('submit', async function (event) {
        event.preventDefault();
        if (!validate('login')) return;
        var f = fieldsFor(forms.login);
        var submit = forms.login.querySelector('.auth-submit');
        setBusy(submit, true, 'Sign in', 'Signing in…');
        setMessage('');
        var redirecting = false;
        try {
            var r = await postJson('/api/login', { email: f.email.value.trim(), password: f.password.value });
            if (r.ok && r.body && r.body.token) {
                setMessage('Signed in. Opening live translation…', 'success');
                storeSession(r.body);
                redirecting = true;
                window.location.href = nextUrl();
                return;
            }
            setMessage(r.status === 401 ? 'Email or password is wrong.' : errorText(r.status, r.body), 'error');
        } catch (e) {
            setMessage('Cannot reach the server. Check that it is running, then try again.', 'error');
        } finally {
            if (!redirecting) setBusy(submit, false, 'Sign in', '');
        }
    });

    forms.register.addEventListener('submit', async function (event) {
        event.preventDefault();
        if (!validate('register')) return;
        var f = fieldsFor(forms.register);
        var submit = forms.register.querySelector('.auth-submit');
        setBusy(submit, true, 'Create account', 'Creating account…');
        setMessage('');
        try {
            var r = await postJson('/api/register', {
                username: f.username.value.trim(),
                email: f.email.value.trim(),
                password: f.password.value,
            });
            if (r.ok) {
                var email = f.email.value.trim();
                forms.register.reset();
                showTab('login');
                $('login-email').value = email;
                $('login-password').focus();
                setMessage('Account created. Sign in with your new password.', 'success');
            } else {
                setMessage(errorText(r.status, r.body), 'error');
            }
        } catch (e) {
            setMessage('Cannot reach the server. Check that it is running, then try again.', 'error');
        } finally {
            setBusy(submit, false, 'Create account', '');
        }
    });

    $('signout-btn').addEventListener('click', function () {
        clearSession();
        setMessage('Signed out.', 'success');
        showSignedIn();
    });

    // Google: real sign-in only when the backend has GOOGLE_CLIENT_ID; otherwise explain.
    var googleBtn = $('google-btn');
    var googleNote = $('google-note');
    function googleUnavailable(text) {
        googleBtn.setAttribute('aria-disabled', 'true');
        googleNote.textContent = text;
        googleNote.hidden = false;
    }

    // Guest demo: shown only when the server allows it (loopback by default).
    async function setupGuest() {
        var btn = $('guest-btn');
        try {
            var r = await fetch('/api/auth/config');
            if (!r.ok || !(await r.json()).guest_enabled) return;
        } catch (e) { return; }
        btn.hidden = false;
        $('guest-note').hidden = false;
        btn.addEventListener('click', async function () {
            btn.disabled = true;
            setMessage('Starting guest session…');
            try {
                var res = await postJson('/api/guest', {});
                if (res.ok && res.body && res.body.token) {
                    storeSession(res.body);
                    window.location.href = nextUrl();
                    return;
                }
                setMessage(errorText(res.status, res.body), 'error');
            } catch (e) {
                setMessage('Cannot reach the server. Check that it is running, then try again.', 'error');
            }
            btn.disabled = false;
        });
    }
    setupGuest();

    async function setupGoogle() {
        var clientId = null;
        try {
            var r = await fetch('/api/auth/config');
            if (r.ok) clientId = (await r.json()).google_client_id || null;
        } catch (e) { /* backend down: password sign-in shows its own error */ }
        if (!clientId) {
            googleUnavailable('Google sign-in is not set up on this server yet. Use email and password.');
            return;
        }
        var tries = 0;
        while (!(window.google && window.google.accounts && window.google.accounts.id)) {
            if (++tries > 50) { googleUnavailable('Google sign-in could not load. Use email and password.'); return; }
            await new Promise(function (res) { setTimeout(res, 100); });
        }
        window.google.accounts.id.initialize({
            client_id: clientId,
            callback: async function (resp) {
                setMessage('Signing in with Google…');
                try {
                    var r = await postJson('/api/auth/google', { id_token: resp.credential });
                    if (r.ok && r.body && r.body.token) {
                        storeSession(r.body);
                        window.location.href = nextUrl();
                        return;
                    }
                    setMessage(errorText(r.status, r.body), 'error');
                } catch (e) {
                    setMessage('Cannot reach the server. Check that it is running, then try again.', 'error');
                }
            },
        });
        googleBtn.hidden = true;
        var slot = document.querySelector('.gis-slot');
        slot.hidden = false;
        window.google.accounts.id.renderButton(slot, { theme: 'outline', size: 'large', shape: 'pill', width: 320, text: 'signin_with' });
    }

    showTab('login');
    showSignedIn();
    googleBtn.addEventListener('click', function () {
        setMessage('Google sign-in is not set up on this server yet. Use email and password.', 'error');
    });
    setupGoogle();
})();
