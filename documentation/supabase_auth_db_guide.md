# Supabase (DB + Google Auth) — setup guide, AFTER Monday

Status 2026-09-25: local SQLite + password login + Google ID-token login behind
`GOOGLE_CLIENT_ID` all work without Supabase. Migrate only when multi-machine or
handler-facing accounts matter. Nothing below blocks the demo.

## A. Google login right now (no Supabase needed)

1. Google Cloud Console → your existing project → APIs & Services → Credentials.
2. Create Credentials → OAuth client ID → **Web application**.
3. Authorized JavaScript origins: `https://localhost:8000` (add the deployed origin later).
   No redirect URI needed (GIS uses a POST-back-free token flow to our own backend).
4. Copy the Client ID → backend env: `export GOOGLE_CLIENT_ID="<id>.apps.googleusercontent.com"`.
5. Install deps (`google-auth` is already in `requirements.txt`), `make run`.
6. Open `/ui/auth/auth.html` → the decorative buttons become real Google buttons;
   without the env var they explain the setup instead of failing silently.
7. Backend verifies the ID token server-side (`POST /api/auth/google`), links/creates
   the local `users` row by Google `sub` (`google_sub` column, auto-migrated on SQLite),
   returns the existing token shape — frontend storage flow unchanged.

## B. Supabase migration (DB + auth together)

1. supabase.com → New project (region closest to SK) → save the DB password.
2. Authentication → Providers → Google → enable with the **same** Console client ID
   + client secret (Credentials page). Site URL: your deployed origin.
3. Table Editor → create `profiles` (`id uuid PK = auth.users.id`, `username text unique`,
   `email text`); RLS: users read/update own row only.
4. Data migration (one-off): SQLite `users` → CSV → import into `profiles`;
   `speaker_voices/speaker_voices.json` → Supabase Storage bucket `voices` (private),
   metadata column or a `voices` table. Keep local files as fallback until verified.
5. Backend: add `supabase` + `gotrue` (or plain `pyjwt` + Supabase JWKS URL) and a
   `get_current_user_supabase` dependency verifying the Supabase access token;
   keep the SQLite path behind `DATABASE_URL` unset. `db_manager.get_db_session_and_engine`
   already accepts any URL — Postgres works with `connect_args` adjusted (remove
   `check_same_thread` for non-sqlite).
6. Frontend: `supabase-js` `signInWithOAuth({provider:'google'})` replaces the GIS
   button; session token goes to the backend as before.

## C. What NOT to do

- Don't run Supabase + local SQLite writes simultaneously (split-brain voices).
- Don't commit service-role keys or the DB password — env only, never in repo.
- Don't migrate before Monday: auth changes the night before a demo are how demos die.
