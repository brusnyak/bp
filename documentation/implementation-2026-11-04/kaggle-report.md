# Kaggle replay work — 2026-11-04

## Result

**Version 1 COMPLETE; component receipt independently checked**, Tesla T4, bounded to 1200 seconds:
[`yegorby/bp-omnivoice-bounded-session-replay-20261104`](https://www.kaggle.com/code/yegorby/bp-omnivoice-bounded-session-replay-20261104).
Parent observed COMPLETE using `kaggle kernels status yegorby/bp-omnivoice-bounded-session-replay-20261104`.
Worker retrieved outputs and verified actual GPU, 12 complete synthesis rows, 9
scenario timelines, zero errors, waveform hashes/formats/durations/RTFs and three
normalized reference hashes/formats. Human audio quality and identity remain
pending. No model was loaded or installed locally.

The first push metadata id lacked `bounded`, but Kaggle created the title-derived
slug above. Local metadata now matches the actual slug; there was only one push.
The submitted `job.py` SHA256 is
`738b7eaf44c2f14efeaf7330104f0a32244b7c7ba0eadb1e5e1045f750357c60`.

## Actual previous failure

`kaggle kernels output yegorby/bp-omnivoice-matrix-20261104 -p documentation/implementation-2026-11-04/kaggle-prior-output`
retrieved the latest error outputs. All three language result JSONs end at
`omnivoice_eval.py:142 → validate_wav:77` with:

> ValueError: requires 16kHz mono PCM16 WAV

Their reference files are mounted `en.wav`, `sk.wav`, `cs.wav` in the existing private
dataset. Model load had already succeeded (EN receipt: cold load 22.24 seconds,
Tesla T4). Therefore the latest failure is a reference-format validation failure,
**not an OmniVoice import failure or demonstrated Python-3.13 incompatibility**.
The logs include dependency warnings, but they are not the terminating exception.
Earlier kernel versions were not diagnosed here.

Also found a later latent evaluator defect: successful rows stored `audio_s` while
its aggregate requested `duration_s`. The new runner consistently uses `duration_s`.
The old worktree was left untouched. Raw prior outputs have a local `.gitignore`
because their corpus/reference content is private; do not commit them.

## New experiment scope

Files: `scripts/gpu_bench/kaggle_session/{job.py,kernel-metadata.json,test_replay.py,README.md}`.
Reused known `OmniVoice.from_pretrained/create_voice_clone_prompt/generate` API and
existing reference dataset; no new owner audio upload, cloud pipeline change,
secret exposure, paid compute or repository push.

The remote job normalizes reference format, prepares/warm-ups one cloned prompt per
language, measures three phrases plus their joined presentation paragraph in
EN/SK/CS, then replays those measurements through synthetic meeting/presentation/
contention queues. Measures cold load, whole-waveform availability, synthesis time,
duration/RTF, GPU allocator peak, audio hashes, generation queue, playback queue,
first-playable delay and queue trend. Twelve measured rows expected, plus 3 warmups.
Human intelligibility/meaning and identity are explicitly pending. TTS timing excludes
STT, MT, source speech detection, transport and actual speaker playback.

This does not prove live model streaming: it measures batch phrase completion.
The replay arrival times are declared synthetic text-ready times. Its Stop scenario
is mathematical playback cancellation, not actual GPU/browser cancellation. These
limits are carried in both code output and README to prevent route promotion.

## Verification

- `.venv/bin/python -m unittest discover -s scripts/gpu_bench/kaggle_session -p test_replay.py -v`
  passed **5/5** in 1.116 seconds. Checks slow-GPU backlog, fast-TTS/slow-playback
  backlog, cut/drop Stop semantics, invalid measurements, and FLOAT/stereo/24kHz
  reference normalization into mono PCM16/16kHz.
- `ast.parse` accepted both Python files; metadata assertion requires private kernel
  and exactly the existing private reference dataset.
- `git diff --check -- scripts/gpu_bench/kaggle_session` passed before submission.
- `kaggle kernels push -p scripts/gpu_bench/kaggle_session --accelerator NvidiaTeslaT4 -t 1200`
  succeeded, version 1. `kaggle kernels status ...` returned RUNNING once; no polling loop.
- Official [OmniVoice model card](https://huggingface.co/k2-fsa/OmniVoice) and
  [repository](https://github.com/k2-fsa/OmniVoice) were read for API usage. Package
  version from PyPI JSON was `0.2.1`; HF model API supplied the pinned revision.

## Parent continuation

Outputs have now been downloaded to `documentation/implementation-2026-11-04/kaggle-session-private/`,
which contains a `.gitignore` excluding all private output files. A sanitized,
transcript-free aggregate is [kaggle-sanitized-receipt.json](kaggle-sanitized-receipt.json).
Retrieval commands:

```sh
kaggle kernels status yegorby/bp-omnivoice-bounded-session-replay-20261104
kaggle kernels output yegorby/bp-omnivoice-bounded-session-replay-20261104 -p <gitignored-private-output-dir>
```

The verification command loaded `session-replay.json`, asserted the exact pinned
revision/package, T4, expected IDs (EN/SK/CS × four), 12 complete rows and 9 finite
timelines. It recomputed SHA256 for every audio file, parsed each PCM16 mono WAV,
checked positive frame count and exact reported sample rate/duration, recomputed
RTF and checked all timing values finite/positive. The three normalized reference
files matched their hashes and mono PCM16/16kHz format. The raw receipt SHA256 is
recorded in the sanitized JSON for provenance.

## Measured results and limits

Warm phrase generation returns the **whole waveform**, not a native stream:

| Target | Three phrase waveform-availability range | Joined paragraph waveform availability | RTF, phrases / paragraph |
|---|---:|---:|---:|
| EN | 0.649–0.860 s | 1.542 s | 0.147–0.214 / 0.102 |
| SK | 0.691–1.037 s | 1.938 s | 0.128–0.197 / 0.109 |
| CS | 0.558–0.713 s | 1.403 s | 0.119–0.163 / 0.103 |

Measured cold model load: 23.018 s; total job: 83.011 s. Versions: OmniVoice 0.2.1,
Torch/Torchaudio 2.11.0+cu130, Transformers 5.20.2, SoundFile 0.14.0, SciPy 1.16.3.
These small warmed measurements show a useful **chunked-cloning candidate on T4**;
they do not support calling the model unusable for real-time merely from RTF.
They also do not establish production latency or stable sustained workload.

| Synthetic schedule | EN maximum first-playable delay | SK maximum first-playable delay | CS maximum first-playable delay |
|---|---:|---:|---:|
| Meeting text ready at 0, 4, 9 s | 1.690 s | 4.127 s | 1.723 s |
| Presentation text ready at 0, 3, 6 s | 3.690 s | 7.081 s | 3.988 s |
| Contention text ready at 0, 0, 5 s | 4.519 s | 8.081 s | 4.988 s |

SK presentation delay grows by 6.390 s between the first and last phrase; contention
grows by 7.390 s. This comes largely from serial playback duration, not insufficient
GPU synthesis throughput. The synthetic arrival schedules are not measured source
speech or a realistic duration-matched bilingual lecture; use them to expose queue
behavior, not assert these delays will occur in every real meeting.

All WAVs are structurally valid and hash-verified. PCM16 saturated samples are present
(approximately 0.0028%–0.0736% across the saved clips). Pre-write floating-point peaks
were not recorded; do not infer clipping-free quality. Listen and grade meaning,
pronunciation and identity; a follow-up harness should record pre-write peaks and
an explicit normalization policy. No human-quality pass was performed.

Retain `route_unverified`: continuous STT→MT→cloned-TTS→meeting routing, native
streaming, sustained load, cancellation and human listening remain separate tests.

Weakest point: TTS-only synthetic arrival replay cannot certify a real meeting.
Cheapest failure: fast generation still creates playback backlog or unintelligible
cloned speech. Smallest fix: inspect queues and saved audio, then run the whole
configured route under continuous source audio with bilingual human review.
