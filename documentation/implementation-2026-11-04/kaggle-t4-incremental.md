# Kaggle T4: incremental streaming vs short segments (2026-11-04)

Kernel `yegorby/bp-incr-t4-20261104` v1 (private, COMPLETE per `kaggle kernels status`; report `status: complete`, 628 s of 1500 s, all 4 runs). EN->SK, 90 s of `me_en_rainbow` at 1x, server restarted per run. STT faster-whisper `small.en` CUDA fp16; MT Opus-MT en-sk int8 on CUDA (smoke test passed); TTS generic Piper `sk_SK-lili-medium` (personal fine-tune NOT uploaded; Piper ran on CPU, the repo does not wire its onnx GPU provider). Tesla T4, Python 3.13, ctranslate2 4.8.5, faster-whisper 1.2.0, torch 2.11.0+cu130.

## Table (scripts/live/summarize_replay.py semantics; chrF++ from fragment_quality.py)

| Run | Config | mean mid-span delay 1.0 / adaptive (s) | worst 1.0 / adaptive (s) | emitted segs voiced | audio s / 90 | chrF++ vs unfragmented MT | INCR STT step dt median / p95 / max (s) |
|---|---|---|---|---|---|---|---|
| A | pack defaults 0.25 / 3 s, incremental off | 2.1 / 2.1 | 3.2 / 2.9 | 32/32 | 57 | 59.9 | n/a |
| B | BP_INCREMENTAL=1 defaults | 3.7 / 3.7 | 6.2 / 6.2 | 33/36 | 61 | 60.2 | 0.19 / 1.76 / 3.58 (67 steps) |
| C | incr, STEP 0.4, MIN_WORDS 3, MAX_WAIT 1.0 | 2.9 / 2.9 | 5.0 / 4.9 | 40/43 | 60 | 63.6 | 0.166 / 1.45 / 3.85 (87 steps) |
| D | A again | 2.1 / 2.1 | 3.2 / 2.9 | 32/32 | 57 | 59.9 | n/a |

- Incremental does NOT beat A on T4: B is +1.6 s worse mean, C +0.8 s worse mean (worst +3.0 / +1.8 s). Only chrF++ edges up (C +3.7 vs A), an unfragmented-MT agreement score, not a quality grade; B/C also leave 3 of 36 / 3 of 43 emitted segments unvoiced.
- A vs D variance: mean identical to 0.1 s (both 2.1 / worst 3.2), under 0.5 s. Same STT transcripts (32 identical); TTS arrival times differ by ~10-30 ms. One input, one voice, so this is run-to-run noise only.
- Crashes: none. 0 Tracebacks and 0 "incremental step error" in all four server logs. Caveat: `dt` is my added log field and includes `model_call_lock` wait plus GPU STT; the 0.17-0.19 s median is the typical GPU step, the 1.5-3.9 s tail is contention with the MT/TTS calls sharing that lock (cause inferred, not isolated).

## Not proven / caveats

- No listening grade, no WER; hypotheses flicker (B log: "sunlight's dry", "sunlit strikes rainbows"), STT quality of incremental vs segment mode ungraded. Mid-span delay for A overstates delay slightly (summarize_replay's own note), so A's lead is if anything conservative.
- Piper SK is a generic voice, CPU; not the owner's voice, not OmniVoice. Result applies to a fast TTS where TTS is not the bottleneck.
- One 90 s EN->SK input; other languages and speakers untested. Route stays `route_unverified`.
- Deviations on the Kaggle copy only: `INCR step` log gets `dt=`; requirements relaxed for Python 3.13 (SQLAlchemy>=2.0.36); client `config_update` extended in-process with `stt_model_size=small.en` (`paced_replay.py` untouched). No licence edit needed.

## Files (gitignored private dir)

`/Users/yegor/Documents/STU/BP/documentation/implementation-2026-11-04/kaggle-route-private/`: `incr-report.json`, `paced-{A,B,C,D}.json`, `server-{A,B,C,D}.log`. Job: `/Users/yegor/Documents/STU/BP/scripts/gpu_bench/kaggle_incr/{job.py,kernel-metadata.json,build_package.py}`. New private dataset `yegorby/bp-incr-code-20261104` (code + 90 s owner voice + public lili Piper voice); existing datasets untouched.
