# AMD laptop fetch checklist (from GitHub, after push)

Machine: second laptop for testing (Linux). Nothing transfers except git —
venvs, models, and secrets all rebuild on site.

## 1. Pull

```
git clone <repo-url> && cd BP
git log --oneline -5   # expect the voice-lab + PWA + defaults commits
```

## 2. Environment (venvs are NOT in git — rebuild)

```
python3.11 -m venv venv && venv/bin/pip install -r requirements.txt
brew/apt: ffmpeg, espeak-ng, cmake, ninja   # apt-get on Linux
make install   # downloads: whisper (base/small/turbo ~800MB-1.5GB),
               #   Opus-MT ct2 (~70MB each), Piper voices, XTTS (~2GB, slow!)
```

Sizes to expect: `venv/` ~2G, HF cache ~2G. `processed/`, `*.wav`, `*.onnx`
personal voices: the SHIPPED onnx (sk male, en v1/v2) ARE in git — no retraining.

## 3. Secrets (never in git — copy from Mac by hand)

`.env` must contain (see Mac `.env`):
- `GOOGLE_CLIENT_ID=143226447148-a1ff...apps.googleusercontent.com`
- `JWT_SECRET=<hex from Mac>` (else all tokens invalidate on restart)

Check: `source ~/.zshrc` must NOT export a stale GOOGLE_CLIENT_ID
(the 403 saga — shell env beats `.env`).

## 4. Linux watch items (differ from Mac)

- `hardware.detect_backend`: no coreml on Linux — verify it falls back to
  cpu/cuda instead of crashing (boot log will say).
- `DYLD_LIBRARY_PATH` in Makefile is Darwin-only (guarded already).
- Audio input: browser mic via HTTPS works the same (`make run` serves
  `https://localhost:8000`, certs/ committed? if not, `make certs`).
- OAuth origin: same `https://localhost:8000` allowlisted — no Console change.

## 5. Smoke tests (in order, stop at first red)

```
venv/bin/python -m pytest test/backend_auth_tests.py -k "not initialize" -q  # ~1s, no models
venv/bin/python -m pytest test/hardware_test.py -q                            # 7/7 expected
python3 -m http.server 8080  →  http://localhost:8080/ui/voice-lab/lab.html   # hard-refresh!
make run  →  https://localhost:8000/ui/live-speech/live.html                  # login, 1 sentence EN→SK
```

## 6. Optional (heavy)

- `.venv-stt` (Parakeet EN): rebuild only if testing EN STT on AMD
  (`python3.11 -m venv .venv-stt`, transformers 5.x + torch + script).
- `.venv-train` (Piper fine-tune): rebuild only if training on AMD.
- Parakeet/NeMo spike, 5k-step resume: parked decisions, see PLAN.md.
