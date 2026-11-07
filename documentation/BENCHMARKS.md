# Benchmarks (measured 2026-11-07)

All raw data is in `documentation/evidence/2026-11-07/`. Numbers in the README and in `LANGUAGES.md` are marked and verified by `make check-docs`. Anything not marked here was read from the same files by hand.

## What was measured and how

| Question | Method | Where it ran | Evidence file | Reproduce |
|---|---|---|---|---|
| Speech recognition per language and model | FLEURS test, 60 clips per language, mean word error | Kaggle T4 | `lang_matrix.json` | kernel `scripts/gpu_bench/kaggle_matrix` |
| Translation quality | opus-100 test sentences (4-40 words), chrF | Kaggle T4 | `lang_matrix.json` | same kernel |
| Whole chain per language | speech to text to translation to voice on 25 clips of about 10 s | Kaggle T4 (voice on its CPU) | `e2e_chain.json` | `scripts/gpu_bench/kaggle_e2e` |
| Slovak models | FLEURS Slovak plus a 134 s reading | Kaggle T4 | `sk_compare.json` | `scripts/gpu_bench/kaggle_sk_turbo` |
| Streaming strategies | offline replay with a real-time clock, measured per-call cost | Kaggle T4 | `stream_sim.json` | `scripts/gpu_bench/kaggle_stream_sim` |
| Live route delay | paced replay through the WebSocket route at 1x | M1 Pro | `replay_*.json`, `paced_replay_summary.json` | `scripts/live/paced_replay.py` |
| Cost of one recognition call | median of 12 windows per length | M1 Pro | in this file | `scripts/` benchmarks in the session notes |

Heavy and repeated tests run on Kaggle, not on a laptop. Run a kernel with `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`.

## Key findings

**1. Whisper has a fixed cost per call.** It pads every input to 30 s, so one call costs the same for 0.5 s of audio as for 3 s. Parakeet's cost grows with the audio. M1 Pro, median of 12 windows:

| Window | MLX turbo | MLX small | Parakeet (CPU) |
|---|---|---|---|
| 0.5 s | 787 ms | 176 ms | 81 ms |
| 1.0 s | 720 ms | 161 ms | 96 ms |
| 2.0 s | 720 ms | 174 ms | 147 ms |
| 3.0 s | 722 ms | 177 ms | 224 ms |

Consequence: on a Mac, shortening chunks with turbo makes the delay worse (calls queue up). Live route, 60 s of your English reading, delay after a segment ends: 2.0 s chunks median 1.19 s; 1.5 s chunks 3.21 s; 1.0 s chunks 3.11 s with 35% word error.

**2. Streaming strategies, Kaggle T4 simulation** (latency = time from a word being spoken until committed; error vs the script):

| Strategy (Whisper turbo unless noted) | English latency | English error | German latency | German error |
|---|---|---|---|---|
| Independent 2 s chunks (current live route) | <!-- claim:sim:en-owner|turbo|chunk-2.0:lat -->1.17<!-- /claim --> s | <!-- claim:sim:en-owner|turbo|chunk-2.0:wer -->19.1<!-- /claim -->% | 1.23 s | <!-- claim:sim:de-fleurs|turbo|chunk-2.0:wer -->33.1<!-- /claim -->% |
| Decode growing buffer every 1.0 s, commit agreed words | <!-- claim:sim:en-owner|turbo|agree-1.0:lat -->2.31<!-- /claim --> s | <!-- claim:sim:en-owner|turbo|agree-1.0:wer -->5.0<!-- /claim -->% | 2.34 s | <!-- claim:sim:de-fleurs|turbo|agree-1.0:wer -->11.2<!-- /claim -->% |
| Same, every 0.5 s | <!-- claim:sim:en-owner|turbo|agree-0.5:lat -->1.45<!-- /claim --> s | <!-- claim:sim:en-owner|turbo|agree-0.5:wer -->12.3<!-- /claim -->% | 1.36 s | 23.9% |
| Same, every 0.3 s | collapses (engine falls 30 s behind) | | collapses | |
| Parakeet, every 0.3 s | 1.23 s | <!-- claim:sim:en-owner|parakeet|agree-0.3:wer -->26.4<!-- /claim -->% | 1.12 s | 28.5% |
| Parakeet, every 1.0 s | 2.91 s | 10.0% | 2.25 s | 13.8% |

No setting reached under 1 s with under 10% error. The accurate route trades about 1 s of extra delay for 4x fewer errors; the fast route is for captions.

**3. Whole-chain timing, T4**: 1.15-1.57 s for a 10 s utterance in every direction measured; the voice is the largest part (0.47-0.82 s). Parakeet on CUDA: 0.11-0.16 s per 10 s clip (about 3x faster than turbo); the earlier "slow Parakeet" number came from a run that only had the CPU provider.

**4. Negative results** (so nobody repeats them): Parakeet inside the old incremental route gave a 3.65 s median delay with duplicated and dropped fragments; feeding the previous sentence to the recogniser as context produced a repetition loop (off by default, `BP_STT_CONTEXT=1`); NLLB did not beat Opus-MT; loudness normalisation did not help a non-native Slovak recording.

## Limits of these numbers

- FLEURS is clean read speech; conversation, noise and accents are harder. The owner's own recording is the only real-voice test, one speaker.
- The simulation uses engine timestamps, not audio playback; add 0.3-0.6 s for translation and voice in the live route.
- Translation chrF on opus-100 is relative (subtitle-like text), not an absolute quality score; no human review was done.
- Slovak on the owner's recording is one 134 s file (about 190 words): differences of one point are two words.
