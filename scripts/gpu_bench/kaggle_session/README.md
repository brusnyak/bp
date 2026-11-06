# Bounded OmniVoice session replay

This is **measured TTS plus synthetic queue replay**, not a microphone, STT→MT→TTS,
WebSocket, browser, actual conference or native-model streaming test.

Existing baseline `scripts/gpu_bench/gpu_bench.py` and matrix evaluator were reused
as API/method references. The previous matrix failed before synthesis because its
WAV validator required 16kHz mono PCM16 despite the mounted references not matching
that format. This runner decodes, mixes/resamples and writes a normalized private
reference, recording original and resulting metadata and hashes. It does not stub
imports or silently ignore dependency failures.

## Run

Local checks use the installed lightweight numerical dependencies, no model:

```sh
.venv/bin/python -m unittest discover -s scripts/gpu_bench/kaggle_session -p test_replay.py -v
```

Authorized remote execution only:

```sh
kaggle kernels push -p scripts/gpu_bench/kaggle_session --accelerator NvidiaTeslaT4 -t 1200
kaggle kernels status yegorby/bp-omnivoice-bounded-session-replay-20261104
```

Kernel stays private, mounts only the existing `yegorby/bp-gpu-bench-refs` dataset,
and uploads no new recordings. `job.py` refuses execution outside Kaggle. Runtime
installation is remote only. Both platform timeout and a process alarm cap the job
at 1200 seconds, including setup. OmniVoice package `0.2.1` and model revision
`c5fdb5ccb189668d56333f77ba2629f4cd7535f4` are fixed; actual Torch/Transformers
versions are recorded rather than fabricated. The runner refuses a non-T4 GPU.

## What the receipt means

For EN/SK/CS: prepare a clone prompt from the mounted reference, warm up once,
measure three distinct phrase generations and one joined paragraph. Save audio,
hashes, generation service time, whole-waveform availability, duration/RTF and
allocator peak. Then replay the measured phrases through three explicit synthetic
text-ready schedules: meeting turns, presentation cadence and conference contention.

The single-GPU and serial-playback timelines expose two different queue sources:
compute can fall behind arrivals, and playback can fall behind even if TTS is fast.
These are deterministic simulations using actual measured service times, not three
new real-time generation runs. No source-audio arrival timing is measured.

Because `generate()` returns a completed waveform, its elapsed time is the first
available audio here. It must not be called native streaming first-packet latency.
The Stop-at-5s report only simulates dropping/cutting playback; it does not test
GPU cancellation or browser stop behavior. Actual cancellation is explicitly
`not_tested`. No WER, human meaning/intelligibility or clone-identity pass is claimed.

Raw outputs contain private reference/clone audio. Download into a gitignored local
directory and never add them to git. Promote only sanitized receipts after checking
the exact version, expected 12 result rows, hashes, failures, audio and human grades.

Official usage checked 2026-11-04: [model card](https://huggingface.co/k2-fsa/OmniVoice)
and [repository](https://github.com/k2-fsa/OmniVoice). Their API examples establish
usage only; vendor speed/language claims are not BP route evidence.
