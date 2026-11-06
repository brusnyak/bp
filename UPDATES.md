# BP — Updates (pre-push)

Prepared 2026-10-30. Local clone: `~/Documents/STU/BP` at GitHub `main` (`faf68ff`, 2026-10-21). These are uncommitted changes awaiting your push approval (AGENTS.md hard rule: no push without owner go-ahead).

## 1. SK→EN STT quality tiers (`backend/main.py`)

**Problem:** Slovak→English was the default SK path but `whisper-small-sk` produces Czech-like hallucinations (WER 0.27–0.32 > 0.2 threshold) — admitted as "experimental" yet shipped as default. Not production-ready for a conference/lecture use case.

**Fix:** `_pick_stt_model()` now exposes a quality tier via env var:
- `BP_SK_STT_MODEL=kinit/whisper-large-v3-turbo-sk` — best quality (~6–13% WER on Slovak per SloPal 2026 benchmark), slower (~800 MB model). Faster-whisper downloads HF names on first use.
- `BP_SK_STT_MODEL=<any CT2 dir or HF size>` — explicit override.
- Fallback chain (unchanged behavior when env unset): env var → local fine-tuned `whisper-small-sk` → `small-sk`.

Removed the old silent fallback to `large-v3-turbo` (worst of both worlds: slow + poor SK WER 0.44). Updated `test/config_tests.py` to match (`test_slovak_prefers_local_tuned_model_then_small`).

## 2. JWT authentication on WebSocket (`backend/main.py` + `ui/live-speech/live-script.js`)

**Problem:** `/ws` was unauthenticated and registration open — README itself warns not to expose to untrusted networks. A blocker for any public/university LAN deployment.

**Fix:**
- `handle_audio_stream(websocket, token=None)` now requires `?token=<jwt>` query param; closes `4003` (missing) or `4003` (invalid/expired/user-not-found) otherwise.
- Frontend `setupWebSocket()` reads `localStorage.userToken` (already set after login) and appends `?token=` to the WS URL.

## 3. Optional slowapi (rate limiting) (`app.py` + `backend/main.py`)

**Problem:** `slowapi` imported unconditionally but not in `requirements.txt` — fresh clone crashes on import (`ModuleNotFoundError`) even though rate limiting is opt-in via `BP_ENABLE_RATE_LIMIT=1`.

**Fix:** Wrapped slowapi imports in try/except; rate limiter only enabled when `BP_ENABLE_RATE_LIMIT=1` AND slowapi is installed. Removed duplicate `StaticFiles`/`Jinja2Templates` imports in `app.py`.

## 4. Conditional SSL origin for Cloudflare Tunnel (`app.py`)

**Problem:** Tunnel origin must be HTTP (cloudflared terminates public TLS), but backend defaulted to self-signed HTTPS → 502 Bad Gateway.

**Fix:** `uvicorn.run` ssl params are `None` when `BP_NO_SSL=1`, else self-signed certs (preserves existing local HTTPS flow).

## 5. Multi-language evidence tests (`test/language_evidence_test.py`, new)

**Problem:** 13 language packs declared in `languages/packs/` (en-sk, sk-en, en-cs, cs-en, en-de, en-pl, en-zh, fr-en, hi-en, ja-en, pl-en, en-hi, zh-en) but no test coverage; SK→EN was the only validated direction.

**Fix:** New stdlib-only test (no ML deps, runs in 0.4s) that:
- Loads all 13 packs and asserts every pack is `experimental` on a clean clone (no `ct2_models/` or eval records yet — honest: declared, not verified).
- Tests pack add/remove, STT/MT/TTS resolution, staleness rejection (legacy records, missing hashes, missing clip, unresolved pack), `cmd_test` timing.

Runs via `python3 test/language_evidence_test.py` (bypasses `conftest.py` which needs fastapi dev deps).

## 6. Measurements (clean clone, no `.venv`)

| Metric | Value |
|--------|-------|
| `git clone` wall time | ~23 s |
| Source tree excl. `.git` | ~50 MB |
| Full size incl. `.git` | 730 MB |
| `setup.py --dev` | 114 s |
| `make test` | 33 passed, 4 pre-existing VAD failures (mock webrtcvad, unrelated to these changes) |
| `test/language_evidence_test.py` | 13/13 pass |

## 7. Public access (tunnel, not a domain)

Backend runs on `http://localhost:8000` (`BP_NO_SSL=1`). Cloudflare Quick Tunnel exposes it at:
`https://seeds-vaccine-arkansas-entity.trycloudflare.com`

**This is ephemeral** — URL changes on every `cloudflared` restart, no uptime SLA. For a stable custom subdomain (e.g. `bp.yourdomain.com` or `real-time-translation-in-your-own-voice.yourdomain.com`), you must first register a domain and add it to Cloudflare (free tier); then I can create a named tunnel + DNS route.

## Thesis mode / UI staleness

**Thesis mode is present** in this clone (GitHub `main`): `ui/home/home.css` has `.thesis-mode` sections, `DESIGN.md` documents it. The UI served through the tunnel is the GitHub version.

If you removed thesis mode in a *different* local checkout and want that removal on GitHub, **pushing my changes will not do it** — my clone still has thesis mode. You would need to either:
(a) push your checkout with thesis mode removed, or
(b) remove thesis mode here (`ui/home/` thesis CSS/sections) and then push.

Ask if you want me to remove thesis mode from this clone before pushing.

## Files changed

Modified:
- `app.py` — optional slowapi, conditional SSL
- `backend/main.py` — SK→EN tiers, JWT on WS, optional slowapi
- `test/config_tests.py` — updated SK fallback assertion
- `ui/live-speech/live-script.js` — WS passes `?token=`

New:
- `test/language_evidence_test.py` — multi-language evidence tests
- `UPDATES.md` — this document

## Before pushing (your approval needed)

AGENTS.md hard rule: no push without owner go-ahead. Confirm:
1. Approve `git push origin main` with the above changes?
2. Remove thesis mode from this clone before pushing? (yes/no)
3. Any other changes to fold in first (e.g. the unverified `~/bp-review-2026-10-26/` workspace modifications to `ci.yml`/`Makefile`/`sdk.md`/`language_packs.py`)?
