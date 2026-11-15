# Review 2026-11-09: are we measuring the right things, and can Piper / OmniVoice behave like the big three?

Short answer: the components are mostly right, the instrument has four known flaws, and the one lever not yet tried is pace policy. Evidence below is from files in this repo (`documentation/evidence/2026-11-09/`, `scripts/live/paced_replay.py`, `backend/prosody.py`); nothing here is a new measurement unless it says so.

## What the big three do that our test does not model (read from their pages, `NEAR_REALTIME_GOAL.md`)
Google keeps "intonation, pacing and pitch" inside one model and balances waiting for context against translating immediately; Microsoft runs a cascade (recognition, English text, translation, voice) and can simulate the speaker's voice; OpenAI streams speech-to-speech. None publishes a lag number. "A few seconds behind" is a human-interpreter-style ear-voice span, so our 3 s goal is the right target, but nobody has shown how it is measured.

## Four flaws in how we measure (checked in code or data this session)
1. **Voxtral lag is understated.** `paced_replay.py` takes `source_end` from the span the server reports. For Whisper that is the word's real end time; for Voxtral it is the time of the audio push that completed the word (`backend/main.py`, `_push_stream_words`), i.e. at least the model's own delay (about 0.5 s) plus the push interval late. Voxtral and Whisper lag numbers are therefore not comparable until Voxtral spans use reference word times (forced alignment of the source audio, offline).
2. **The lag number is a sum nobody can take apart.** `final_metrics` has stt/mt/tts time but not the fragment hold or the playback wait. In the saved Transformers-Voxtral run the time from "word pushed" to "piece emitted" was near 0 s for short pieces but 2.5 s, then 3.3-6.6 s for the long pieces late in the minute (`replay_en-de_voxtral-agree_short.json`, one bogus 534 s span ignored), while MT (0.1-0.6 s) and TTS start (0.1-0.3 s) stayed small. That run is the engine at 1.0-1.1 x real time, so the growth is the recogniser falling behind, not the fragment policy; the vLLM run will show whether it disappears.
3. **(Resolved, not a scorer bug.)** In the Transformers-Voxtral replay the lag is identical at every playback rate because pieces *arrive* late (recogniser behind), so playback speed cannot help; in the Whisper replay of the same clip the lag falls with playback rate (table below), because there the queue is the bottleneck. Identical lag across rates is therefore a diagnostic: arrival-bound, not queue-bound.
4. **Test material is not conversation.** Generated Piper speech and a slow, paused reading. No overlapping speech, no natural pace, no real room. Quality is scored by word error only; nobody has judged intonation or seams between pieces.

## Pace is the lever (measured offline, 2026-11-09, `scripts/live/replay_whatif.py` on the saved Whisper T4 run, en to de, Piper, `agree_short`)
Piper's German audio is **1.32 x as long as the English source** (27 pieces with real word spans). `RateFollower` clamps the voice to 0.85-1.15 x (`backend/prosody.py:26`), so a deficit remains and the queue grows. Re-running the same saved events through the listener model at other playback rates:

| playback rate | lag median | p90 | max | growth (2nd half minus 1st) | playback ends after input end |
|---|---|---|---|---|---|
| 1.0 | 3.16 s | 5.10 s | 9.51 s | +1.11 s | +9.9 s |
| 1.15 (current cap) | 2.67 s | 3.89 s | 6.23 s | +0.57 s | +6.4 s |
| 1.3 | 2.40 s | 3.23 s | 5.35 s | +0.37 s | +5.3 s |
| 1.45 | 2.03 s | 3.23 s | 4.68 s | +0.30 s | +4.5 s |
| adaptive (UI policy: 1.0-1.45 by backlog) | 2.72 s | 4.08 s | 6.33 s | +0.49 s | +6.1 s |

Against the success test (median <= 3.0 s, p90 <= 4.5 s, growth small): this saved Whisper run already passes lag and p90 under the UI's adaptive playback. Caveats: one 60 s clip; the model assumes playback speed-up costs nothing (pitch-preserved speed-up of 1.3-1.45 x is audible: **an ear check is needed before this counts**); the earlier "fail" verdicts used the unadjusted queue. Piper is never the bottleneck (0.12 s per piece); its limits are the generic voice and the seams between independently synthesised pieces.
- OmniVoice is the same pipeline with a 0.8-1.0 s fixed cost per piece on a T4 (October data) and a cloned voice. Its `speed` now follows the rate follower (`test/omni_pace_tests.py`); `duration` from the source span is not built.

## What a configuration "like the big three" would be (design, untested end to end)
Streaming recogniser (Voxtral on vLLM: T4 lag 0.08 s after the audio) -> piece policy (2-6 words or a clause, 0.8 s cap) -> translation -> TTS per piece (Piper default, OmniVoice cloned voice on a GPU) -> rate follower with a wider cap, plus adaptive playback -> a lag budget of recogniser 0.6 s + hold 0.5-1 s + MT 0.2 s + voice 0.1-1 s. The sum is 1.4-2.8 s on paper. The route test with vLLM (queued on Kaggle) is the first time all stages run together.

## Better instruments (smallest first)
1. Built: `INCR emit: hold=...` lines in the server log (fragment hold per piece, `Fragmenter.hold_s`, `test/incremental_tests.py`) and `scripts/live/replay_whatif.py` (offline playback-rate sweep and TTS/source ratio on any saved replay). Still to build: MT and TTS-start times in the same record, so one log line takes a lag apart.
2. Reference word times for the source (offline alignment, run once per test clip) so every engine is scored against the same clock.
3. Done for en to de on the Whisper run (ratio 1.32); repeat for en to sk and on the vLLM and OmniVoice runs when they land.
4. A real two-person recording and an ear panel for seams and pace (maintainer), scored with the existing `score_success.py`.
