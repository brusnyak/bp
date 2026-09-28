# Linux setup + test sheet (third OS, not yet run)

Constitution I says every required-path dependency must run on Windows, macOS **and**
Linux without per-OS hand-tuning. Windows 11 (CPU-only) and macOS (M1 Pro) are measured;
Linux is the remaining blank. This file is the plan and the lab sheet for that run.

Status honesty: **nothing below is measured on Linux yet.** Everything in *Expected* is
derived from `backend/hardware.py`'s probe chains, the two working machines and the
requirements files — treat it as a hypothesis to confirm or falsify, and fill in *Measured*.

## 1. What Linux should differ in (expected, verify each)

| Area | Expected on Linux | Why |
|---|---|---|
| Device chain, `stt` / `mt` (CTranslate2) | `cuda → rocm → cpu`; plain x86 laptop ⇒ **cpu** | `backend/hardware.py` `_CHAINS` |
| `tts_baseline` (Piper/ONNX) | `cuda → rocm → cpu`; Windows-only `directml` and macOS-only `coreml` never apply ⇒ **cpu** | same, plus `_directml()`/`_coreml()` are OS-gated |
| `tts_clone` (XTTS/OpenVoice) | `cuda → rocm → cpu` (no MPS) | same |
| XTTS / Hybrid availability | **available** — Coqui ships `manylinux` wheels (the Windows-only gap does not exist here) | `requirements.txt` notes |
| OpenVoice (Hybrid) | manual `--no-build-isolation --no-deps` install, same as macOS | `requirements.txt` |
| Virtual mic | PipeWire/PulseAudio `null-sink` + loopback, not BlackHole/VB-Cable | constitution V |
| `certs/` | committed (`certs/cert.pem`, `key.pem`) — no OpenSSL dance needed | repo |
| System deps | `ffmpeg`, `libsndfile1`, `espeak-ng`, `cmake`, `ninja-build`, `python3.11-venv` | Piper/audio/toolchain |
| `DYLD_LIBRARY_PATH` | macOS-only, already guarded in the Makefile | `Makefile` |

## 2. Setup (x86_64, no GPU)

```bash
sudo apt-get install -y python3.11 python3.11-venv ffmpeg libsndfile1 espeak-ng cmake ninja-build
git clone https://github.com/brusnyak/bp.git && cd bp
python3.11 -m venv venv && venv/bin/pip install -r requirements.txt   # ROCm/CUDA: see note below
# .env is never in git -- copy GOOGLE_CLIENT_ID + JWT_SECRET from the Mac by hand
make certs                                    # only if certs/ is absent from the checkout
venv/bin/python backend/tts/download_piper_models.py sk_SK-lili-medium
venv/bin/python backend/tts/download_piper_models.py en_US-ryan-medium
venv/bin/python backend/tts/download_piper_models.py cs_CZ-jirka-medium
venv/bin/python -c "import sys; sys.setrecursionlimit(2000); import backend.mt.convert_opus_mt_to_ct2 as c; c.convert_model('Helsinki-NLP/opus-mt-en-sk','ct2_models/Helsinki-NLP--opus-mt-en-sk',quantization='int8')"
# repeat for opus-mt-sk-en (and opus-mt-en-cs if CZ is still in scope)
```

GPU note: `requirements.txt` installs stock PyPI torch, which covers CUDA and CPU. **ROCm is
not covered** — on an AMD GPU install torch from the ROCm index first, then the rest; this is
the one path with no recorded evidence on any machine, so it goes in the *Unknown* table below.

## 3. Test ladder (same three gates as macOS; stop at the first red)

| # | Command | Pass means |
|---|---|---|
| 1 | `python3 scripts/demo_preflight.py` | required assets present (no server needed) |
| 2 | `venv/bin/python -m pytest test/hardware_test.py test/vad_tests.py test/backend_auth_tests.py test/backend_api_tests.py -q` | 20 passed |
| 3 | `venv/bin/python test/piper_pipeline_test.py` | exit 0 |
| 4 | `make run` + `make demo-check` | server up, engines listed, per-stage backends sane |
| 5 | `venv/bin/python test/interrupt_smoke_test.py` | translations + TTS chunks, clean teardown |
| 6 | Browser: `https://localhost:8000/ui/live-speech/live.html`, login, one EN→SK sentence | audio out in the personal SK voice |

## 4. Record sheet (fill on the Linux machine)

| Item | Expected | Measured | Notes |
|---|---|---|---|
| OS / kernel / CPU | any x86_64 | | |
| Python | 3.11.x | | |
| `pip install -r requirements.txt` wall time | ~10–30 min (downloads) | | |
| venv size | ~2.2 GB full profile | | |
| `hardware_backends` from `/api/voice-lab/status` | `stt=mt=cpu, tts_baseline=cpu, tts_clone=cpu` | | |
| pytest set | 20 passed | | |
| piper pipeline script | exit 0 | | |
| live WS rehearsal: STT / MT / TTS / total | ~0.5 / 0.1 / 0.2 / ~0.8 s (macOS M1 reference) | | CPU-only ⇒ expect slower TTS/STT |
| EN WER (`en_script_reading.m4a`) | ≤ 0.10 | | macOS base = 0.077 |
| SK WER (`sk_script_reading.m4a`) | ~0.41 (turbo) | | the known bottleneck |
| XTTS engine advertised | yes (Coqui installable) | | |
| Hybrid engine call | works after the manual OpenVoice step | | |
| Virtual mic visible in the UI dropdown | PipeWire null-sink | | |

## 5. Unknowns to resolve on that run

- ROCm install path (torch index) — no evidence on any machine yet.
- Whether `faster-whisper` `large-v3-turbo` (the SK default) fits a modest Linux laptop's
  RAM when the STT model upgrade fires.
- Parakeet EN (`.venv-stt`) on Linux CPU: expected fine, never tried.
- Chromium/PipeWire audio permissions under a headless or containerised session.
