# Live Speech Translation

Real-time speech translation that runs on your own machine: speak English, hear Slovak or Czech a moment later, with live captions on both sides. Built as a Bachelor's thesis project (STU FEI) and developed towards a commercial product.

[![Watch the demo](https://img.youtube.com/vi/_-jwEyGxDYs/maxresdefault.jpg)](https://www.youtube.com/watch?v=_-jwEyGxDYs)

*Click the image to watch the demo on YouTube.*

## What it does

Microphone → voice-activity detection → speech recognition (STT) → machine translation (MT) → speech synthesis (TTS) → browser playback and captions. Everything runs locally; no audio leaves the machine unless you choose a cloud route.

| Direction | Status |
|-----------|--------|
| English → Slovak | Works; the demo route |
| English → Czech | Works |
| Slovak / Czech → English | Experimental: recognition errors are still common on Slovak and Czech speech |
| Other packs (`languages/packs/`) | Declared; need installed models and a test before use |

**Speed (measured 2026-11-06, M1 Pro 16GB, CPU):** EN→SK first translated audio 0.39s; ~0.67s per 2s speech window (STT ~0.59s, MT ~0.03s, TTS ~0.05s); RTF 0.672 (faster than real-time on the reference Mac). Continuous streaming — audio is processed as it arrives, not batched into 3 s segments. Whisper `base` int8 for EN STT; Whisper `small-sk` int8 for SK/CS STT. SK→EN is experimental (whisper-small-sk WER 0.25–0.32 > 0.2 gate). CS→EN near real-time on Kaggle T4 GPU (p50 1.86s, p95 4.22s onset→first audio); not trusted on local CPU due to STT quality. Raw data: `documentation/VERIFICATION_RESULTS_2026-11-06.md` and `documentation/implementation-2026-11-04/`.

## Quick start

```bash
git clone https://github.com/brusnyak/bp.git
cd bp
python3.11 scripts/setup.py --dev   # venv, dependencies, .env, voices, translation models (needs Python 3.11 or 3.12 and ffmpeg)
make run                            # http://localhost:8000
```

Open **http://localhost:8000** in Chrome or Edge (plain HTTP on `localhost` counts as a secure page, so the microphone works and there is no certificate warning). Press **Try it as a guest** (or create an account), pick the direction, press **Initialize**, run the sound check, then **Start**.

- Use headphones, or route playback to a virtual cable (BlackHole on macOS, VB-Cable on Windows), so the translation does not feed back into the microphone. The app detects a virtual cable and offers it as an output.
- Another device on the network (phone, second laptop) needs HTTPS: run `BP_HTTPS=1 BP_HOST=0.0.0.0 make run`. This uses the self-signed certificate in `certs/`, which browsers warn about once.
- Safari and Firefox cannot choose the audio output device.

### Useful commands

| Command | Purpose |
|---------|---------|
| `make test` | Backend regression suite (offline) |
| `make demo-check` | Pre-flight for a live demo: assets and the running server |
| `.venv/bin/python scripts/bp.py selftest` | Asset and configuration checks |
| `python3 scripts/live/paced_replay.py <audio> --src en --tgt sk` | Replay a recording through the real WebSocket route at real time; reports per-segment lag |

### Configuration (environment variables)

| Variable | Default | Effect |
|----------|---------|--------|
| `BP_GUEST` | on for loopback | `1` allows no-account guest sessions on a network-exposed server, `0` disables them. Guests cannot manage voices |
| `BP_PORT` / `BP_HOST` | `8000` / `localhost` | Listen address. Anything other than loopback exposes open registration: use only on a trusted network |
| `BP_HTTPS` | off | `1` serves HTTPS with `certs/` |
| `BP_VAD_MAX_SEGMENT_S`, `BP_VAD_SILENCE_S` | `3.0`, `0.25` | Longest segment and pause that closes it. Shorter = lower delay, choppier translation |
| `BP_SILENCE_RMS` | `0` (off) | Pre-VAD volume gate. The old default `0.02` dropped quiet speech |
| `BP_STT_STEP_S` | `0.5` | STT step interval (seconds) — how often to run STT on the audio buffer |
| `BP_STT_WINDOW_S` | `2.0` | STT audio window size (seconds) |
| `BP_SILENCE_GAP_S` | `0.3` | Gap between words that starts a new phrase (seconds) |
| `BP_UTTERANCE_SILENCE_S` | `0.8` | Silence that ends an utterance (seconds) |
| `BP_MIN_UTTERANCE_S` | `1.0` | Minimum utterance length for STT (seconds) |
| `BP_ENABLE_RATE_LIMIT` | off on loopback, on otherwise | Limits registration, login and uploads. `0` disables, `1` forces on |
| `BP_LOG_SESSIONS` | off | `1` writes transcripts and translations to `processed/sessions/` (personal data; tell participants) |
| `BP_INCREMENTAL` | off | Deprecated — use continuous streaming mode instead |
| `BP_SK_STT_MODEL` | built-in | Slovak recognition model name or CTranslate2 directory |
| `BP_DEVICE`, `BP_MT_DEVICE` | auto | `cpu` or `cuda` for STT and MT |

## The stack

| Stage | Default engine | Licence | Optional |
|-------|---------------|---------|----------|
| VAD | WebRTC VAD | BSD-3 | Silero VAD |
| STT (EN) | Parakeet TDT v3 | CC-BY-4.0 | `whisper-small` (CPU fallback) |
| STT (SK/CZ) | Whisper `small-sk`/`small` int8 | MIT | `large-v3-turbo`, fine-tuned Slovak models (untested here) |
| MT | Opus-MT, CTranslate2 int8 | CC-BY-4.0 | MADLAD-3B (CUDA) |
| TTS | Piper | GPL-3.0-or-later runtime (not bundled; user installs via `setup.py --dev`); each voice has its own licence | Kokoro 82M (English), OmniVoice cloning (GPU, research) |

## Project structure

| Path | Purpose |
|------|---------|
| `app.py` | FastAPI app, WebSocket server, serves the UI |
| `backend/` | STT/MT/TTS/VAD engines, model pooling, registry, pipeline |
| `languages/packs/*.json` | Language packs: models, voice, segmentation per direction |
| `ui/` | Browser interface (plain HTML/CSS/JS, installable PWA) |
| `scripts/` | Setup, CLI (`bp.py`), benchmarks, Kaggle jobs (`scripts/gpu_bench/`) |
| `test/` | pytest suites; Node playback tests: `node --test test/session_playback_test.cjs` |
| `documentation/` | Thesis draft, findings, research, runbooks |

More: `PRODUCT.md` (positioning), `PLAN.md` (development plan), `AGENTS.md` (project memory), `documentation/implementation-2026-11-04/` (latest measurements).

## Integration (conference rooms, LHC)

| Mode | Status | How |
|------|--------|-----|
| One person, one direction, into Teams/Zoom | Works | Browser session plus a virtual audio cable (BlackHole, VB-Cable) as playback device and meeting microphone |
| Captions only | Works | Live page subtitle strip and pop-out window |
| Per-seat broadcast (one speaker, many listeners, each with own language and earbuds) | Designed, not built | One recognition pass, one translation and voice per language, one WebSocket per seat; see `documentation/HANDOFF_2026-11-04_session2_spec_plan.md` section 2.2 and 6.3 |
| Multi-speaker meeting with overlap | Not built | One stream per participant is the cheapest route; speaker separation is future work |

Interface contract: `POST /api/guest` or `/api/login` returns a token; open `ws://HOST/ws?token=TOKEN`, send the config message, stream 16 kHz audio; read `transcription_result`, `translation_result` and binary WAV `tts_audio` messages. Capacity measured on one T4 (English to Slovak, continuous speech): about 4 simultaneous sessions with median lag under 1.5 s, 8 sessions 3.7 s.

## Privacy

Recognition, translation and synthesis run locally. Session logs are off by default. Sign-in uses a local database; Google sign-in is optional and talks to Google. Voice recordings you upload stay on the machine, are tied to your account and are not used outside it. A shared deployment needs its own retention and access policy.

## Status and licence

**Not yet a commercial release.** Remaining gates before selling it:

| Gate | Remaining proof |
|------|-----------------|
| Slovak / Czech recognition | A stronger model and a scored test (WER) per direction |
| Cloned voice | Listener review of meaning, intelligibility and identity on the exact models and hardware |
| Real meeting | Run with a physical microphone, speakers and a virtual cable; record first-audio p50/p95 and queue growth |
| Clean install | Test on a fresh machine per supported OS |
| Licences | Default engines: faster-whisper/CT2 MIT/Apache-2.0, Opus-MT CC-BY-4.0, Piper GPL-3.0-or-later (not bundled; separate licence decision for distribution). Owner's personal fine-tunes, OmniVoice (CC-BY-NC weights) and Slovak STT fine-tune still need clearance. XTTS/NLLB/SeamlessM4T non-commercial — blocked by registry gate. |
| Shared operation | Capacity, retention, monitoring and a support owner |

Repository licence: see `LICENSE`. Engine, checkpoint and voice licences are separate; the registry gate (`backend/registry.py`) blocks unlicensed engines but is not a completed commercial audit.
