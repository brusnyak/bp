# Security audit, 2026-09 (static review + dependency audit)

Scope: `app.py`, `backend/`, dependency set, repository contents, setup tooling. Method: manual code review, `pip-audit`, checksum verification of the downloaded `uv` binary. Regression tests: `test/security_tests.py`.

## Fixed

| # | Finding | Severity | Fix |
| - | --- | --- | --- |
| 1 | Hard-coded fallback JWT secret (`dev-only-insecure-secret-change-me`) published in the repo: anyone could forge a session token | High | `backend/utils/auth.py`: random per-process secret when `JWT_SECRET` is unset; `scripts/setup.py` writes a random one to `.env` |
| 2 | Default account `test@example.com` / `password` created on an empty DB | High | Opt-in only: `BP_DEMO_USER=1` |
| 3 | Server bound to `0.0.0.0` with open registration and an unauthenticated `/ws` | High | Binds `127.0.0.1` by default; `BP_HOST` opts in and prints a warning |
| 4 | `/speaker_voices` static mount served every user's uploaded recordings and `speaker_voices.json` (all filenames/user ids) | High | Audio extensions only, JSON never served; directory is no longer tracked in git |
| 5 | Path traversal in `PUT /voices/rename`: `new_name` went into `os.rename` unsanitised | High | Same character policy as upload plus `realpath` containment check (`_safe_voice_name`, `_voice_path`); delete uses the containment check too |
| 6 | `speaker_wav_path` from unauthenticated `/initialize` and the WebSocket reached the TTS layer (arbitrary local file path) | Medium | Restricted to files inside `speaker_voices/` in the shared `initialize_all_models` choke point and the WS config path |
| 7 | First 10 characters of every password (DEBUG) and of the demo user's hash written to logs | Medium | Removed |
| 8 | Google login linked accounts by email without checking `email_verified` | Medium | Rejected unless `email_verified is True` |
| 9 | `python-jose` 3.3.0 (algorithm-confusion / DoS CVEs) and its unmaintained `ecdsa` dependency | Medium | Replaced by PyJWT (HS256 only is used) |
| 10 | 134 known CVEs in 6 packages (nltk, transformers, starlette, python-jose, ecdsa, requests) | Medium | starlette/fastapi/requests/transformers bumped and pinned; pip/setuptools upgraded during setup; 134 -> 9 remaining (8 transformers, 1 nltk), see below |
| 11 | `upload_voice` crashed with a 500 when `Content-Type` was missing | Low | `(content_type or "")` |

## Accepted / not fixed

- `transformers` 4.57.6 has advisories fixed only in 5.x. The lite install uses it only for `AutoTokenizer` (no model or pickle loading), so the advisories' code paths are not reachable. Revisit when moving to transformers 5.
- `nltk` (dev-only, used by the evaluation framework) has one open advisory.
- With the full PyTorch install (not the default), `backend/stt/whisper_streaming_vendor/silero_vad_iterator.py` calls `torch.hub.load("snakers4/silero-vad")`, which downloads and executes code from a GitHub repository at runtime (unpinned). It is unreachable in the default install (no torch). Pin a commit or vendor the model before enabling it.
- The WebSocket `/ws` has no authentication. Mitigated by the loopback default; add token auth before exposing it on a network.
- No rate limiting on `/login` and `/register`; registration is open. Same mitigation.
- The whole uploaded body is read into memory before the 50 MB check.

## Repository hygiene (action required from the owner)

The public history contains material that cannot be removed by a normal commit:

- `certs/key.pem` (self-signed localhost key, low impact, but a private key in a public repo);
- `speaker_voices/*.m4a` (voice recordings) and fine-tuned personal Piper voice models (`*personal*.onnx`): biometric/personal data;
- about 360 MB of ONNX models that inflate every clone.

This change stops tracking them (`.gitignore`, `git rm --cached`) but they remain in history. Removing them from history needs a rewrite (`git filter-repo`) and a force-push, or a fresh repository. Treat the committed key as compromised (it is regenerated locally by `scripts/gen_cert.py`).

## Supply chain of the setup itself

- `uv` release zip verified against its published SHA-256.
- Model downloads come from huggingface.co (Whisper, Opus-MT, Piper voices); nothing is executed from downloaded content. `trust_remote_code` is not used.
- The `.claude/skills/speckit-*` files shipped in the repo contain no hooks or pre-approved tools.
