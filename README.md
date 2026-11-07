# Live Speech Translation

Real-time speech translation that runs on your own machine: speak English, hear German, Slovak or Czech a moment later, with live captions on both sides. Built as a Bachelor's thesis project (STU FEI) and developed towards a commercial product.

[![Watch the demo](https://img.youtube.com/vi/_-jwEyGxDYs/maxresdefault.jpg)](https://www.youtube.com/watch?v=_-jwEyGxDYs)

*Click the image to watch the demo on YouTube.*

## What it does

Microphone → voice-activity detection → speech recognition (STT) → machine translation (MT) → speech synthesis (TTS) → browser playback and captions. Everything runs locally; no audio leaves the machine unless you choose a cloud route.

| Direction | Status |
|-----------|--------|
| English → Slovak | Works; the demo route |
| English → German | Components and the chain are measured (`documentation/LANGUAGES.md`); not yet run live in the browser |
| English → Czech | Works |
| Slovak / Czech → English | Experimental: recognition errors are still common on Slovak and Czech speech |
| Other packs (`languages/packs/`) | Declared; need installed models and a test before use |

**Speed and accuracy (measured 2026-11-07; every number below is checked against its saved receipt by `make check-docs`):**

- **Live route, EN→SK, M1 Pro, delay from end of speech to the first translated audio** (`scripts/live/paced_replay.py` at 1×, evidence in `documentation/evidence/2026-11-07/`): Whisper `base` median <!-- claim:paced:fw_base_lag_median -->0.82<!-- /claim --> s; `mlx-whisper` large-v3-turbo median <!-- claim:paced:mlx_lag_median -->1.01<!-- /claim --> s; turbo with the pause mirror (below) median <!-- claim:paced:mlx_prosody_lag_median -->1.19<!-- /claim --> s, p90 <!-- claim:paced:mlx_prosody_lag_p90 -->2.65<!-- /claim --> s. An earlier "0.39 s first audio" figure could not be reproduced and was removed.
- **Speech recognition, Kaggle T4, mean word error on FLEURS read speech (60 clips per language)**, German: base 18.1%, small <!-- claim:stt:whisper-small:de:wer -->10.9<!-- /claim -->%, large-v3-turbo <!-- claim:stt:whisper-large-v3-turbo:de:wer -->4.8<!-- /claim -->%, Parakeet v3 <!-- claim:stt:parakeet-tdt-v3:de:wer -->6.3<!-- /claim -->%. Czech: turbo <!-- claim:stt:whisper-large-v3-turbo:cs:wer -->11.6<!-- /claim -->% (base 67%). Slovak: turbo <!-- claim:stt:whisper-large-v3-turbo:sk:wer -->10.7<!-- /claim -->%, Parakeet <!-- claim:stt:parakeet-tdt-v3:sk:wer -->9.9<!-- /claim -->%. Turbo wins German, Czech, Polish and English; Parakeet wins French and Slovak by under 2 points. Full table: `documentation/evidence/2026-11-07/lang_matrix.json`. Parakeet ran on CPU in that run (no CUDA provider), so its speed there is not a GPU number.
- **Translation, same T4 run, chrF on 200 opus-100 test sentences (relative scores):** German→English Opus-MT <!-- claim:mt:opus-mt:de:to_en:chrf -->57.9<!-- /claim --> versus NLLB-1.3B 54.6; Opus-MT won every language/direction cell measured, 4–8× faster.
- **CS→EN on Kaggle T4, full route (`route_receipt.json`):** default profile p50 <!-- claim:kaggle:default:p50 -->3.48<!-- /claim --> s, p95 <!-- claim:kaggle:default:p95 -->4.13<!-- /claim --> s from speech onset to first audio. The shorter 1.0 s-segment profile reaches p50 1.87 s / p95 4.26 s (choppier output). Earlier figures of 1.86/4.22 s belonged to that profile, not the default.
**Streaming foundation (Kaggle T4 simulation, offline replay of your English reading and a 2-minute German FLEURS stream; `documentation/evidence/2026-11-07/stream_sim.json`).** Latency is the time from a word being spoken until it is committed. Independent 2 s chunks (the current live route) with Whisper turbo: median <!-- claim:sim:en-owner|turbo|chunk-2.0:lat -->1.17<!-- /claim --> s but <!-- claim:sim:en-owner|turbo|chunk-2.0:wer -->19.1<!-- /claim -->% word error (German <!-- claim:sim:de-fleurs|turbo|chunk-2.0:wer -->33.1<!-- /claim -->%), because words are cut at chunk edges. Decoding the growing buffer every 1.0 s and committing only words two consecutive decodes agree on: median <!-- claim:sim:en-owner|turbo|agree-1.0:lat -->2.31<!-- /claim --> s and <!-- claim:sim:en-owner|turbo|agree-1.0:wer -->5.0<!-- /claim -->% error (German <!-- claim:sim:de-fleurs|turbo|agree-1.0:wer -->11.2<!-- /claim -->%). A 0.5 s step gives <!-- claim:sim:en-owner|turbo|agree-0.5:lat -->1.45<!-- /claim --> s and <!-- claim:sim:en-owner|turbo|agree-0.5:wer -->12.3<!-- /claim -->% but the engine falls up to 2 s behind; 0.3 s collapses (queue of 30 s). Parakeet is cheap per step (real-time factor 0.1–0.3) and reaches 1.2 s at 0.3 s steps, with <!-- claim:sim:en-owner|parakeet|agree-0.3:wer -->26.4<!-- /claim -->% error. No setting reached under 1 s with under 10% error.

- **Your own voice matters:** the same Slovak script scores 11.5% word error when read by a Slovak Piper voice and 35.4% when read by a non-native speaker (large-v3-turbo, Mac). Slovak→English is experimental for that reason; the model is not the only limit.
- **Test status:** 100 backend tests pass on macOS (`make test`). CI: Ubuntu and Windows pass; the macOS job failed once on `vad_tests.py::test_vad_long_speech_streaming` (timing-sensitive; passes locally).

## Quick start

```bash
git clone https://github.com/brusnyak/bp.git
cd bp
python3.11 scripts/setup.py --profile mac --langs de,sk   # venv, dependencies, .env, voices, models for your hardware
make run PRESET=mac                                         # http://localhost:8000
```

Profiles: `mac` (Apple Silicon, MLX), `nvidia` (CUDA), `cpu`; languages `de,sk,cs,fr,pl` (see `documentation/LANGUAGES.md` for the measured best configuration per language, `documentation/DEMO.md` for a presentation walkthrough). Add `--dev` for the test dependencies. Needs Python 3.11 or 3.12 and ffmpeg.

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
| `BP_STT_ENGINE` | `faster_whisper` | Speech recognition engine. `mlx_whisper` = large-v3-turbo on the Apple GPU (macOS arm64; `pip install mlx-whisper`) |
| `BP_MLX_MODEL` | `mlx-community/whisper-large-v3-turbo` | Model repo for `mlx_whisper` |
| `BP_PROSODY` | `1` | Mirror the speaker's pauses and speaking pace in the translated audio (Piper voices). `0` = play phrases back-to-back as before |
| `BP_PROSODY_EXTRA_LAG_S` | `0.6` | Slack that makes pause mirroring possible: more mimicry = more delay (median +0.18 s measured) |
| `BP_PROSODY_MAX_LEAD_S`, `BP_PAUSE_COMMA_S`, `BP_PAUSE_SENTENCE_S` | `2.0`, `0.25`, `0.5` | Longest pause added before a segment; silence after a comma / sentence end inside one translation |
| `BP_STT_HALLUCINATION_GUARD` | `1` | Drop loops ("1-1-1-…"), caption credits and stock phrases on short noise slices (mlx engine). `0` disables |
| `BP_STT_CONTEXT` | `0` | `1` feeds the previous sentence to the recogniser as context. Off: its first live test produced a repetition loop |
| `BP_DEVICE`, `BP_MT_DEVICE` | auto | `cpu` or `cuda` for STT and MT |

## The stack

| Stage | Default engine | Licence | Optional |
|-------|---------------|---------|----------|
| VAD | WebRTC VAD | BSD-3 | Silero VAD |
| STT (EN) | Whisper `base` int8 via faster-whisper (what `setup.py` installs) | MIT | `mlx-whisper` large-v3-turbo on Apple Silicon (`BP_STT_ENGINE=mlx_whisper`, recommended), faster-whisper large-v3-turbo on CUDA, Parakeet TDT v3 (CC-BY-4.0, not installed by setup) |
| STT (SK/CZ) | Whisper `small-sk`/`small` int8 | MIT | `large-v3-turbo` (10.7% Slovak / 11.6% Czech on T4, see above); `kinit/whisper-large-v3-turbo-sk` via `BP_SK_STT_MODEL` needs CTranslate2 conversion, untested here |
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

More: `PRODUCT.md` (positioning), `documentation/STATUS.md` (where the project stands), `AGENTS.md` (project memory), `documentation/implementation-2026-11-04/` (latest measurements).

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

**Not yet a commercial release.** The honest state, with what is measured, what is not built (accurate streaming route, room mode with one voice and language per listener) and what is waiting for a decision, is in [`documentation/STATUS.md`](documentation/STATUS.md). Measurements and how to reproduce them: [`documentation/BENCHMARKS.md`](documentation/BENCHMARKS.md).

Licences: default engines are faster-whisper/CTranslate2 (MIT/Apache-2.0), Opus-MT (CC-BY-4.0), Piper (GPL-3.0-or-later, not bundled), Parakeet (CC-BY-4.0, optional). Personal fine-tunes and OmniVoice (CC-BY-NC weights) are restricted; see `documentation/LICENSE_INVENTORY_2026-11-06.md`. Repository licence: see `LICENSE`. The registry gate (`backend/registry.py`) blocks unlicensed engines but is not a completed commercial audit.
