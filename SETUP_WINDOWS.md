# Windows setup — the short, tested path (CPU)

> Verified 2026-09-28 on stock Windows 11, no NVIDIA GPU, no admin rights.
> Result: full pipeline (VAD → faster-whisper → Opus-MT → Piper) runs on CPU,
> `test/hardware_test.py + test/vad_tests.py` → **11/11 green**.
> Plan ~30–60 min wall time, almost all of it downloads (torch, HF models).

## 0. You need (check, don't install globally)

- Git, FFmpeg, Node.js — `git --version`, `ffmpeg -version`, `npm --version`.
  If Python 3.11 is already available via `py -3.11`, skip step 1.

## 1. Portable Python 3.11 (no admin, lives inside the project)

```powershell
# from an empty folder that will hold everything (it deletes cleanly later)
$ROOT = "C:\Work\bp-test"          # any isolated path works
$env:UV_PYTHON_INSTALL_DIR = "$ROOT\.python"
$env:UV_CACHE_DIR = "$ROOT\.cache"
# uv 0.12.x portable: https://github.com/astral-sh/uv/releases
#   -> uv-x86_64-pc-windows-msvc.zip, extract to $ROOT\tools\uv\uv.exe
& "$ROOT\tools\uv\uv.exe" python install 3.11
& "$ROOT\tools\uv\uv.exe" venv --python 3.11 bp\venv
```

Why not system Python: the pins (`numpy==1.26.4`, `pandas==1.5.3`, `numba==0.60.0`)
target 3.11; 3.12 breaks the install. Why portable: the whole toolchain dies
with the folder — zero traces in the system.

## 2. Clone light, install, models

```powershell
cd $ROOT
$env:GIT_TERMINAL_PROMPT = "0"
# repo is ~510MB (models committed); skip the binaries, scripts fetch them anyway:
git clone --filter=blob:none --no-checkout https://github.com/brusnyak/bp.git bp-src
git -C bp-src sparse-checkout set --no-cone '/*' '!ct2_models/*' '!onnx_models/*' '!speaker_voices/*' '!bootstrap_cz/*' '!bootstrap_de/*' '!certs/*'
git -C bp-src checkout main
cd bp-src

$env:UV_PYTHON_INSTALL_DIR = "$ROOT\.python"
$env:UV_CACHE_DIR = "$ROOT\.cache"
& "$ROOT\tools\uv\uv.exe" pip install -r requirements-windows.txt --python ".\venv\Scripts\python.exe"
npm install   # package.json is at the repo ROOT (there is no frontend/ dir)
& "C:\Program Files\Git\usr\bin\openssl.exe" req -x509 -newkey rsa:4096 -nodes `
  -out certs/cert.pem -keyout certs/key.pem -days 365 -subj "/CN=localhost"
.\venv\Scripts\python.exe backend/tts/download_piper_models.py en_US-ryan-medium
.\venv\Scripts\python.exe backend/tts/download_piper_models.py sk_SK-lili-medium
.\venv\Scripts\python.exe backend/tts/download_piper_models.py cs_CZ-jirka-medium
# MT conversion (needs internet; HF cache stays inside the project):
$env:HF_HOME = "$ROOT\.cache\hf"
.\venv\Scripts\python.exe -c "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; c.convert_model('Helsinki-NLP/opus-mt-en-sk', 'ct2_models/Helsinki-NLP--opus-mt-en-sk', quantization='int8')"
# ... repeat for opus-mt-sk-en, opus-mt-en-cs (or run setup_windows.ps1, fixed version in repo)
```

Or the supported shortcut: `powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1`
once Python 3.11 exists — it does the checks, venv, deps, certs and models with
no admin rights. `setup_windows.ps1 -SkipModels` skips the slow model step.

## 3. Run (two env vars are load-bearing on stock Windows)

```powershell
$env:PYTHONIOENCODING = "utf-8"   # without this, any Slovak č/š/ž printed to a
                                  # cp1252 console kills the pipeline task
$env:HF_HOME = "$ROOT\.cache\hf"  # keeps model caches inside the project
.\venv\Scripts\python.exe app.py
# open https://localhost:8000 (accept the self-signed cert)
```

## 4. Verify

```powershell
$env:PYTHONIOENCODING = "utf-8"; $env:HF_HOME = "$ROOT\.cache\hf"
.\venv\Scripts\python.exe -m pytest test/hardware_test.py test/vad_tests.py -q
# expected: 11 passed
```

## Measured on the reference machine (Ryzen iGPU, CPU-only)

| Stage | Time |
|---|---|
| STT faster-whisper base, ~1.5s speech | ~0.65 s |
| MT Opus-MT en-sk (CT2 int8) | ~0.05 s |
| TTS Piper voice (streaming, 22 chunks) | ~2.9 s total |
| End-to-end | ~3.6 s |

## Known limits (not bugs in your setup)

- **XTTS voice cloning is unavailable** without MSVC Build Tools (Coqui ships
  Linux-only wheels). The app starts without it; the `xtts` engine reports
  unavailable. Piper + personal Piper voices work fully.
- **OmniVoice / OpenVoice** are excluded from the Windows set (dependency
  conflicts documented in `requirements-windows.txt`); both are optional at
  runtime via guarded imports.
- **Live streaming partials are disabled upstream** (whisper-hallucination
  guard) — the UI gets final results, not interim ones.
- Keep `HF_HOME`/`UV_CACHE_DIR` inside the project or caches leak into
  `%USERPROFILE%\.cache` (GBs) and survive folder deletion.

## Tear-down

```powershell
Remove-Item -LiteralPath "C:\Work\bp-test" -Recurse -Force
```

If you ever installed anything with `--scope user` outside the folder, uninstall
it separately — everything above lives strictly under `$ROOT`.
