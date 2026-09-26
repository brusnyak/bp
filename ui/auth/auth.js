document.addEventListener('DOMContentLoaded', function() {
    const loginTab = document.getElementById('login-tab');
    const registerTab = document.getElementById('register-tab');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const authMessage = document.getElementById('auth-message');

    function showForm(formToShow) {
        if (formToShow === 'login') {
            loginForm.classList.remove('hidden');
            registerForm.classList.add('hidden');
            loginTab.classList.add('active');
            registerTab.classList.remove('active');
        } else {
            registerForm.classList.remove('hidden');
            loginForm.classList.add('hidden');
            registerTab.classList.add('active');
            loginTab.classList.remove('active');
        }
        authMessage.textContent = ''; // Clear messages on tab switch
        authMessage.className = 'auth-message';
    }

    loginTab.addEventListener('click', () => showForm('login'));
    registerTab.addEventListener('click', () => showForm('register'));

    loginForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });
            const data = await response.json();
            if (response.ok) {
                authMessage.textContent = data.message;
                authMessage.classList.add('success');
                // Store user info in localStorage
                localStorage.setItem('userToken', data.token);
                localStorage.setItem('userName', data.username);
                localStorage.setItem('userEmail', data.email);
                console.log('Login successful, token:', data.token, 'username:', data.username, 'email:', data.email);
                window.location.href = '/ui/live-speech/live.html'; // Redirect to live speech page
            } else {
                authMessage.textContent = data.message || 'Login failed.';
                authMessage.classList.add('error');
            }
        } catch (error) {
            console.error('Error during login:', error);
            authMessage.textContent = 'Network error or server unreachable.';
            authMessage.classList.add('error');
        }
    });

    registerForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const username = document.getElementById('register-username').value; // Get username
        const email = document.getElementById('register-email').value;
        const password = document.getElementById('register-password').value;

        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, email, password }), // Include username in the body
            });
            const data = await response.json();
            if (response.ok) {
                authMessage.textContent = data.message;
                authMessage.classList.add('success');
                showForm('login'); // Switch to login tab after successful registration
            } else {
                authMessage.textContent = data.message || 'Registration failed.';
                authMessage.classList.add('error');
            }
        } catch (error) {
            console.error('Error during registration:', error);
            authMessage.textContent = 'Network error or server unreachable.';
            authMessage.classList.add('error');
        }
    });


    // Google Sign-In (real button only when the backend is configured with
    // GOOGLE_CLIENT_ID; otherwise the decorative buttons explain the setup).
    async function setupGoogle() {
        let clientId = null;
        try {
            const r = await fetch('/api/auth/config');
            if (r.ok) clientId = (await r.json()).google_client_id;
        } catch (e) { /* backend down — password forms still work */ }
        const slots = document.querySelectorAll('.gis-slot');
        const decorative = document.querySelectorAll('.google-button');
        if (!clientId) {
            decorative.forEach((b) => b.addEventListener('click', () => {
                authMessage.textContent = 'Google login is not configured yet (backend needs GOOGLE_CLIENT_ID).';
                authMessage.classList.add('error');
            }));
            return;
        }
        // Wait for the GIS script (async) before rendering.
        let tries = 0;
        while (!window.google || !window.google.accounts || !window.google.accounts.id) {
            if (++tries > 50) return;
            await new Promise((res) => setTimeout(res, 100));
        }
        window.google.accounts.id.initialize({
            client_id: clientId,
            callback: async (resp) => {
                try {
                    const r = await fetch('/api/auth/google', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ id_token: resp.credential }),
                    });
                    const data = await r.json();
                    if (!r.ok) {
                        authMessage.textContent = data.detail || 'Google login failed.';
                        authMessage.classList.add('error');
                        return;
                    }
                    localStorage.setItem('userToken', data.token);
                    localStorage.setItem('userName', data.username);
                    localStorage.setItem('userEmail', data.email);
                    window.location.href = '/ui/live-speech/live.html';
                } catch (e) {
                    authMessage.textContent = 'Network error or server unreachable.';
                    authMessage.classList.add('error');
                }
            },
        });
        decorative.forEach((b) => { b.style.display = 'none'; });
        slots.forEach((s) => window.google.accounts.id.renderButton(s, {
            theme: "filled_black", size: "large", shape: "pill", width: 300,
            text: s.dataset.gisText === "signup_with" ? "signup_with" : "signin_with",
        }));
    }
    setupGoogle();

    // Initial form display
    showForm('login');
});
