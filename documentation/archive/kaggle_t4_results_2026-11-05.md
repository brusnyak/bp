# Kaggle T4 Benchmark Results — BP Full-Route
Date: 2026-11-05
Kernel: yegorby/bp-production-ws-cs-en-default-vs-low-latency (v7)
Host: Linux x86_64 (Tesla T4)
Route: app.py:/ws (production WebSocket, CS→EN)
Metric: speech-end → first WAV audio send (excludes browser playback & physical capture)

## CS→EN Results

| Profile | p50 s | p95 s | Utterances | Stage-sum p50 s | Stage-sum p95 s |
|---|---|---|---|---|---|
| default | 3.47 | 4.03 | 17 | 0.56 | 1.98 |
| P_short | 1.98 | 2.88 | 32 | 0.47 | 1.94 |
| P_short2 | **1.58** | 3.52 | 46 | **0.37** | 1.76 |
| P_incr | 10.45 | 12.87 | 14 | 0.47 | 0.66 |

## Interpretation

- **Near real-time achieved on T4**: P_short2 p50 = 1.58s (< 2s target), P_short p50 = 1.98s.
- Pure pipeline processing (STT+MT+TTS) is 0.37–0.56s p50 — well under budget.
- Wall-clock is dominated by VAD segmentation, 20ms audio framing, and WebSocket round-trips, not model inference.
- P_incr (simultaneous/incremental) is slow (10.45s) — its longer max_segment_s delays first output; needs tuning for streaming use.

## Configuration used

- STT: faster-whisper `base`, cuda, float16, beam_size=5
- MT: opus-mt-cs-en (CT2 int8), cpu
- TTS: Piper `en_US-john-medium` (generic EN), cpu
- VAD: webrtcvad, aggressiveness 3, silence_timeout 0.3s, max_segment 3.0s
- HF_HUB_OFFLINE=1 (all models from staged ct2_models/)
- Audio: 15s Czech clip `me_cs_staromestske.m4a`, 3 repeats + warmup

## Next: DE/FR

Need to add:
- MT CT2 models: opus-mt-de-en, opus-mt-en-de, opus-mt-en-fr, opus-mt-fr-en
- Piper TTS voices: de_DE-*, fr_FR-* (generic)
- Language packs: de-en, en-de, en-fr, fr-en
- Then rerun Kaggle T4 for each direction
