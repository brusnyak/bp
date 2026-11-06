# Kaggle T4 Setup — BP Full-Route Benchmark

Kernel: `scripts/gpu_bench/kaggle_full_route/` (`job.py`, pushed as `yegorby/bp-full-route-t4`)

## Why Kaggle

The Mac M1 Pro (reference box) runs CPU-only STT (~6s for whisper-base on 10s audio). The Tesla T4 has GPU CUDA for STT/MT, which is the path to near-real-time. The Kaggle kernel runs **CPU-only** (no `libcublas.so.12` dependency) — this avoids the 8 prior push failures caused by CUDA STT, audio decoding, and VAD filtering. GPU acceleration is a follow-up once the CPU baseline is stable.

## What the kernel does

- Installs `faster-whisper`, `ctranslate2`, `transformers`, `piper-tts`, `ffmpeg`
- Downloads the STT model (whisper-base) and MT model (opus-mt-<src>-<tgt>) from HF at runtime
- Decodes audio via `ffmpeg` → WAV → `soundfile` (reliable on Kaggle; no `librosa` dependency)
- Runs the streaming STT→MT pipeline on 20ms VAD frames
- Measures p50/p95 speech-end→first-translation latency
- Saves `route_report.json` with stage timings and per-fragment results

## Default direction

**CS→EN** (constitution v1.1.0 — Czech↔English is the default pair; Slovak is optional/experimental).

Override with `--source`/`--target`:
```
--source cs --target en    # default: Czech → English
--source en --target cs    # English → Czech
--source fr --target en    # French → English (test-only, MT converted, no FR TTS voice)
--source de --target en    # German → English (test-only, needs MT conversion)
```

## Push to Kaggle

```bash
# From the BP repo root
kaggle kernels push -p scripts/gpu_bench/kaggle_full_route \
  --accelerator NvidiaTeslaT4 -t 1500
```

The kernel is `is_private: true`, `enable_gpu: true`, `enable_internet: true`. No dataset sources — audio is passed via `--audio` or the kernel discovers files in `/tmp/bp-full-route/audio/`.

## Audio file

Provide a 16kHz-compatible WAV or M4A clip. The kernel converts M4A→WAV via ffmpeg. For CS→EN testing, use `speaker_voices/me_cs_staromestske.m4a` (Czech, exists locally).

## Outputs

- `/tmp/kaggle-full-route/output/route_report.json` — full report (p50/p95, stage ms, per-fragment results)
- `route_report.json` — also written to the kernel's working directory

Example report fields:
```json
{
  "direction": "cs->en",
  "first_translation_p50_s": 2.1,
  "first_translation_p95_s": 3.2,
  "stt_mean_ms": 850.0,
  "mt_mean_ms": 120.0,
  "wall_s": 45.0,
  "results_emitted": 32
}
```

## Known limitations (documented, not bugs)

- TTS is measured locally (Piper ~0.05s RTF on CPU) but skipped on T4 — the kernel measures STT+MT latency only.
- No CUDA: avoids `libcublas.so.12` failure. GPU STT/MT is a follow-up.
- VAD RMS threshold defaults to 0.01; override with `BP_VAD_RMS` env var if speech is filtered.
- All non-default language pairs are `experimental` until STT+MT+TTS resolve and a `processed/language_eval/<pair>.json` record passes review.

## Adding a new test language

1. `bp language add <pair>` — scaffolds the pack (STT base, MT opus-mt, TTS auto-detect)
2. Convert MT: `python3 scripts/convert_models.py` (converts all missing pairs)
3. Install TTS voice for the target language (Piper `.onnx` + `.onnx.json` in `backend/tts/piper_models/`)
4. Run `bp language test <pair> --clip <audio>` to record an eval
5. Add `--source <src> --target <tgt>` to the Kaggle kernel invocation

## Failure causes from the 8 prior Kaggle pushes (all addressed)

| Failure | Fix in this kernel |
|---------|-------------------|
| No `libcublas.so.12` for CUDA STT | CPU-only: `device="cpu"`, `compute_type="int8"` |
| Audio decoding (librosa/audioread failures) | `ffmpeg` → WAV → `soundfile` |
| VAD/fragmenter filtering everything | RMS threshold 0.01 (tunable via `BP_VAD_RMS`), fragmenter min_words=2 |
| Wrong direction (SK→EN hardcoded) | `--source`/`--target` args, default CS→EN |
| MT model not converted | Downloaded from HF at runtime (transformers) |
| No TTS on T4 | Skipped, measured locally instead |
| Missing audio dataset | No dataset dependency; pass `--audio` |
| Python 3.14 torch incompatibility | Kaggle uses Python 3.10; no torch needed (faster-whisper + transformers only) |
