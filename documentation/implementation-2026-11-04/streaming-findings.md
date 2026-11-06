# Streaming / delay findings — 2026-11-04 (local, uncommitted)

Setup: M1 Pro, Piper CPU (`piper` → SK personal voice), EN→SK, first 45 s of `speaker_voices/me_en_rainbow.m4a` (continuous reading), real server over loopback, audio paced at 1x by `scripts/live/paced_replay.py`. Raw: `processed/paced_*.json` (gitignored). One run per row; STT wording changed between runs, so treat differences under ~1 s as noise.

## Defects found and fixed (code)

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | Speaker talking on cancelled the translation still being voiced ("barge-in") | baseline: 9 segments, only 7 voiced; 2 translations never produced audio | `session_config["barge_in"]` default False (`BP_BARGE_IN=1` restores); after fix 9/9 voiced |
| 2 | TTS streams sliced every clause into 150 ms WAVs and dumped them at once; browser made one `Audio` element per slice | 209 `tts_audio` messages for 45 s; 246 in `live-ws-check.txt` | Piper/Omni/Hybrid `synthesize_stream` yield one WAV per clause; 13 messages for the same 45 s |
| 3 | TTS of a later segment could start before an earlier one finished (STT/MT are FIFO, TTS is unlocked) | code reading of `_process_speech_segment_pipeline`; not reproduced | per-session sequence events: segment N waits for N-1 before TTS |
| 4 | Per-language VAD config from the previous session never applied over WebSocket (the session already existed; `.update()` copied the session's own values back) | `BP_VAD_MAX_SEGMENT_S=3` ignored: log still "exceeded 6s"; after fix segments ≈3 s | `vad_silence_s`/`vad_max_segment_s` computed once and applied in `.update()`; env `BP_VAD_SILENCE_S`, `BP_VAD_MAX_SEGMENT_S` override |
| 5 | `setTimeout` "gapless" backup could start a chunk over a playing one | code reading | removed; event-driven + `playbackRate` option + UI select (`Translation speed`) |

## Measured delay (listener model: one chunk at a time)

`lag` = listener hears first audio minus the moment the segment's speech ended. `first_word_lag` = minus the previous segment's end (upper bound on what the first word of a segment waits).

| Run | voiced | audio s (45 s input) | lag max ×1.0 / ×1.25 / ×1.4 | first_word max ×1.0 / ×1.25 / ×1.4 |
|---|---|---|---|---|
| baseline (before fixes) | 7/9 | 30.6 | 2.0 / 1.0 / 0.8 | 7.9 / 7.9 / 7.9 |
| fixes 1-3, pack 0.4 s / 6 s | 9/9 | 35.3 | 3.7 / 2.6 / 2.2 | 7.9 / 7.9 / 7.9 |
| silence 0.25 s, max 3 s | 16/16 | 40.0 | 3.5 / 2.4 / 2.0 | 5.9 / 4.9 / 4.6 |
| silence 0.3 s, max 4.5 s | 10/10 | 42.5 | 6.3 / 4.1 / 3.4 | 9.4 / 8.2 / 8.0 |

## Reading

- After a pause the route answers fast: first audio 0.57–0.84 s after segment end (Piper CPU, STT + MT + first clause).
- The delay a presenter feels is the segment length. 6 s segments mean up to ~8 s until the first word is heard. 3 s segments cut that to ~6 s. Mistakes appeared that look like recognizer errors on short context, not MT: the 3 s run heard "prism" as "prison" (→ "väzenie"), "me" for "many". The chrF++ check below isolates MT fragmentation only; STT word error was not measured (no WER run).
- Translated audio is 0.7–0.95× the source duration, so continuous speech without pauses sits near saturation and any burst becomes backlog. Faster playback (×1.25–1.4) is the cheapest lever. What scales is clause-aware cutting (streaming-STT partials) plus tighter MT, not more GPU.
- The 0.25 s / 3 s run has the lowest first-word delay; it is not promoted as default because meaning was not graded. Pack defaults (0.4–0.5 s / 6 s) are now actually in effect.

## Incremental ("per word") mode — built, measured, OFF by default

`BP_INCREMENTAL=1` (or `"incremental": true` in a pack's `vad` block). Code: `backend/stt/incremental.py` (local-agreement commit + clause fragmenter, 3 unit tests in `make test`), `_incremental_step/_incremental_close` in `backend/main.py`. While the speaker talks, the open utterance is re-transcribed every ~0.7 s with word timestamps; words that two consecutive hypotheses agree on are committed, grouped into clause-sized fragments (4–9 words, flush on punctuation or 1.6 s), translated and voiced at once. Fragment text arrives ~2 s after the words (two hypotheses + step time on the M1).

Comparable metric: word-weighted mean delay from the middle of the source words to the moment the listener hears them (`scripts/live/summarize_replay.py`). Same 45 s recording, Piper CPU, EN→SK:

| Config | audio / input | mean delay ×1.0 | mean delay adaptive | worst adaptive |
|---|---|---|---|---|
| pack default 0.4 s / 6 s segments | 35 s / 45 s | 4.1 s | 3.9 s | 4.5 s |
| **silence 0.25 s / 3 s segments** | 40 s / 45 s | **2.6 s** | **2.5 s** | **4.1 s** |
| incremental, defaults | 43 s / 45 s | 5.5 s | 4.8 s | 6.8 s |
| incremental, defaults (rerun) | 41 s / 45 s | 6.1 s | 4.9 s | 6.8 s |
| incremental, step 0.4 s, 3 words, 1.0 s wait | 41 s / 45 s | 6.4 s | 5.0 s | 9.4 s |

Reading: on CPU + Piper the incremental route is worse than short VAD segments. Reasons seen in the log (`INCR step` lines): agreement needs two hypotheses (~2.4 s), fragments are re-translated without sentence context (awkward Slovak: "Je, | Podľa legendy"), and translated audio is 0.9–0.95× the source duration so every extra fragment queues behind the previous one. The design pays off only when STT steps are tens of ms (GPU) and playback can run faster than the source. Re-test it on the T4 before enabling. Human simultaneous interpreters typically trail the speaker by about 2–4 s; a mean of 2.5 s is inside that band, unverified: sub-second per-word translation looks unreachable for EN→SK because Slovak output often needs the English clause's verb first.

Adaptive playback (`Translation speed: Auto`, default): normal speed, +15% from 1 s of queued audio, +30% from 2.5 s, +45% from 4 s, pitch preserved. Shaves 0.1–1.4 s off the mean and 0.4–1.6 s off the worst case in the table above; it cannot remove a backlog bigger than the speed-up allows.

## Tesla T4 follow-up (Kaggle, 90 s, EN→SK, GPU STT small.en fp16, MT CUDA int8, generic Piper SK on CPU)

Report: `kaggle-t4-incremental.md`. Mean word delay at ×1.0 / worst / chrF++ vs unfragmented MT:

| Run | mean | worst | chrF++ | voiced |
|---|---|---|---|---|
| A: pack defaults 0.25 s / 3 s, incremental off | **2.1 s** | **3.2 s** | 59.9 | 32/32 |
| B: incremental defaults | 3.7 s | 6.2 s | 60.2 | 33/36 |
| C: incremental, step 0.4 s, 3 words, 1.0 s wait | 2.9 s | 5.0 s | 63.6 | 40/43 |
| D: A again | 2.1 s | 3.2 s | 59.9 | 32/32 |

A vs D agree within 0.1 s (same transcript), so run-to-run variance for the short-segment profile is small on the T4. Incremental mode is worse here too (B +1.6 s, C +0.8 s): GPU step median 0.17–0.19 s, but fragments queue behind lock contention and the hypothesis flickers ("sunlight's dry" / "sunlit strikes rainbows"). Decision: incremental stays OFF; short VAD segments + Auto speed is the near-real-time path (2.1 s mean, 3.2 s worst on T4; 2.5 s / 4.2 s on M1 CPU). Not tried: lock-free STT step, longer agreement window. One 90 s input, one voice, no WER or listener grade.

## Roast follow-up (independent read-only review, same day)

Fixed after it: Piper registry label `commercial-ok` → `gpl-3.0` (+ `NOTICE.md`); `summarize_replay.py` printed a tautological "N/N voiced" (now counts emitted segments); Stop/Start now close the incremental state and `_run_pipeline_task` drops work from an older capture generation; the TTS-order wait has a 20 s timeout; MT calls share one lock. Still open from it: one run per row (±1 s noise plausible; the 2.6 s vs 4.1 s gap is not far outside it), the 2.5 s mean is mid-span while interpreter ear-voice span is measured from word onset (not like for like), the 3 s row is quality-blind, caption order across fragments and tails is not asserted, clone path never run end to end, README quick-start timings predate auth/offline changes, no virtual-mic (BlackHole/VB-Cable) step in README, self-signed cert has no trust step.

## Not proven

Physical mic/speaker/projector, GPU or cloned voice (this is Piper CPU), CS and reverse routes, a second run to size variance. After the changes: `make test` exit 0, Node playback 6/6, replay-model tests 6/6.
