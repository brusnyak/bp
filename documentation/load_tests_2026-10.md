# Full-cascade load tests on a free GPU — 2026-10-02

Question: can one GPU run the whole cascade (STT → MT → OmniVoice zero-shot clone) fast enough for a presentation, and how many
simultaneous speakers can it carry? Harness: `scripts/gpu_bench/load_bench.py` (kernel `scripts/gpu_bench/kaggle_load/`).
Raw data: `processed/gpu_bench/load_T4/load_bench.json` (+ kernel log). Hardware: **Kaggle Tesla T4, 15.6 GB** (the session had 2×T4;
only one was used). Models: faster-whisper `base` (EN) / `whisper-small-sk` (SK), Opus-MT CT2 float16, **OmniVoice (zero-shot clone
of the owner's reference clip, num_step=16)**. All stages on the GPU, TTS serialised by a lock (one TTS worker, like the app).

## Inputs and what the numbers mean

- Real speech: 33 EN + 31 SK segments (5.5–12 s) cut at pauses from the owner's own recordings, plus 3 short clips per language.
  Repetition: the 40 s and 3-minute scenarios cycle through the same pool.
- No per-segment ground-truth script exists, so `wer_vs_idle` is drift against the idle (n=1) transcript, **not absolute accuracy**
  (absolute WER stays in `processed/stt_baseline.json`). It is ≈0 where nothing breaks and tells us nothing about base accuracy.
- `ttfa` = time to first translated audio after the utterance ends (STT + MT + TTS of the first sentence), sentence-chunked TTS as in the
  live app. `rtf` = total compute / input speech. `lag` (ramp) = finish minus scheduled arrival, i.e. includes the whole utterance.
- Assumed, not measured: VAD hangover 0.5 s, 0.4 s pause between segments. Single run per cell; n_utts per ramp cell is 1–55.

## 1. Speech length sweep (n=1, median of 3 repeats, T4)

| dir | speech (s) | STT | MT | TTS | ttfa | total | RTF |
|---|---|---|---|---|---|---|---|
| EN→SK | 2.9 | 0.09 | 0.02 | 0.73 | 0.83 | 0.83 | 0.29 |
| EN→SK | 7.2 | 0.14 | 0.04 | 1.47 | 1.65 | 1.65 | 0.23 |
| EN→SK | 13.5 | 0.23 | 0.04 | 3.03 | 1.75 | 3.30 | 0.24 |
| EN→SK | 20.2 | 0.34 | 0.04 | 4.41 | 1.92 | 4.79 | 0.24 |
| EN→SK | 41.0 | 0.54 | 0.04 | 9.91 | 2.22 | 10.49 | 0.26 |
| SK→EN | 3.0 | 0.14 | 0.01 | 0.87 | 1.02 | 1.02 | 0.34 |
| SK→EN | 7.4 | 0.27 | 0.05 | 1.81 | 2.13 | 2.13 | 0.29 |
| SK→EN | 13.5 | 0.41 | 0.04 | 2.84 | 2.04 | 3.29 | 0.24 |
| SK→EN | 20.8 | 0.52 | 0.04 | 3.61 | 1.89 | 4.17 | 0.20 |
| SK→EN | 44.9 | 1.12 | 0.06 | 6.87 | 2.82 | 8.05 | 0.18 |

- Total RTF 0.18–0.34 at every length: faster than real time with a cloned voice. TTS is 85–90 % of the time; STT is a few
  tenths of a second on the GPU (1.3 s on the Mac) and MT is negligible.
- First audio arrives after ~2 s whatever the length (1–2.8 s) because TTS is chunked per sentence; a 40 s speech does not wait 10 s.

## 2. Concurrency ramp (open loop: each stream speaks in real time, 90 s per level)

| duty | streams | ttfa p50 / p95 (s) | lag p95 (s) | TTS queue wait (s) | verdict |
|---|---|---|---|---|---|
| 1.0 (all talk) | 1 | 2.0 / 2.1 | 4.0 | 0 | keeps up |
| 1.0 | 2 | 2.4 / 5.7 | 7.0 | 1.1 | queue starts |
| 1.0 | 4 | 3.7 / 5.0 | 6.8 | 0.9 | queue |
| 1.0 | 8 | 39.8 / 101 | 103 | 16.6 | collapsed (work RTF 1.59) |
| 0.25 (meeting) | 1 | 2.0 / 2.0 | 3.7 | 0 | (n=1 utterance) |
| 0.25 | 2 | 2.1 / 2.2 | 3.9 | 0 | keeps up |
| 0.25 | 4 | 2.1 / 2.3 | 4.2 | 0 | keeps up |
| 0.25 | 8 | 2.1 / 2.1 | 4.0 | 0 | keeps up |
| 0.25 | 12 | 3.5 / 7.5 | 9.2 | 1.0 | knee |
| 0.25 | 16 | 6.4 / 20.8 | 22.6 | 6.0 | degraded |
| 0.25 | 24 | 22.5 / 73.6 | 75.3 | 24.8 | collapsed (work RTF 2.19) |

- Capacity of one T4 with one OmniVoice worker: **~2 continuously talking speakers, or ~8 participants in a realistic meeting
  (25 % speaking time)** with p95 first audio ≈ 2.3 s; 12 is the knee, 16+ falls over.
- The bottleneck is the single serialised TTS worker: TTS takes ~3.3 s per 8 s utterance (RTF ≈ 0.4), so the lock saturates at
  ~2.4 concurrent speakers. STT time grows under load only because it competes for the same GPU.
- Compare with the Mac (2025-11-28, Piper, 12 sessions 100 %, 15 sessions 73 %): the same order of concurrency, but there the voice is
  the generic Piper one; here it is the owner's zero-shot clone. Different metric and engine: not a like-for-like benchmark.
- STT drift under load stayed flat (`wer_vs_idle` ≈ 0.19–0.20 at every N; it is a constant offset from clip composition, not a trend).
  Output speech length ratio stayed ≈ 1.0, so no truncation or silent failures were observed.

## 3. Presentation timeline (virtual clock, 22 segments ≈ 3 min, processing run for real)

| scenario | speech | queue wait max | listener delay p50 / p95 / max | bounded |
|---|---|---|---|---|
| monologue EN→SK | 171 s | 0 s | 3.9 / 7.0 / 8.1 s | yes |
| alternating EN↔SK | 162 s | 0 s | 2.5 / 10.6 / 11.5 s | no (drifts at the end) |

- Processing never queued (0 s wait): each segment finished long before the next one ended. The listener delay is a **playback**
  effect: translated audio plays back to back, OmniVoice speaks slower than the source (out/in up to 1.3×, e.g. 15 s for 11.5 s), so
  a few long segments build a backlog that is only worked off in pauses. The alternating run ends with 8–11 s of backlog.
- This is a speaking-rate/design issue, not a compute one: more GPU does not remove it. Options (untested): OmniVoice speed
  parameter ≈1.15–1.3, drop or shorten stale segments when the backlog exceeds a threshold, shorter sentences.

## Not measured / limits

- **VRAM and GPU utilisation: not captured** (the `nvidia-smi` sampler returned nothing in the kernel). The "bottleneck is VRAM" idea
  stays unproven.
- One GPU model, one run per cell, serialised TTS only. A TTS pool (2×T4 were available), batched generation, fewer steps and a
  faster GPU are untested; they should raise the ~2-speaker ceiling roughly in proportion to TTS throughput.
- The live microphone → virtual-cable path was not exercised (Kaggle has no audio devices); the timeline is offline segments.
- Ear quality of the clone under load was not rated; only drift/length were checked.

## What this changes

- A cloned-voice cascade is **faster than real time on a free T4** (RTF 0.2–0.3) and holds a 3-minute presentation with no processing
  backlog. Better compute is needed for *concurrency* (several simultaneous cloned voices), not for a single presenter.
- The remaining presentation risk is playback pacing, which needs a speed/backlog policy rather than hardware.
- Next measurement: same harness with `--tts` pool size 2 on 2×T4, OmniVoice `speed` 1.2, and VRAM sampling fixed; then a live run on the
  presentation machine.
