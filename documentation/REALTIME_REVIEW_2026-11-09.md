# Review 2026-11-09: are we measuring the right things, and can Piper / OmniVoice behave like the big three?

Short answer: the components are mostly right, the instrument has four known flaws, and the one lever not yet tried is pace policy. Evidence below is from files in this repo (`documentation/evidence/2026-11-09/`, `scripts/live/paced_replay.py`, `backend/prosody.py`); nothing here is a new measurement unless it says so.

## What the big three do that our test does not model (read from their pages, `NEAR_REALTIME_GOAL.md`)
Google keeps "intonation, pacing and pitch" inside one model and balances waiting for context against translating immediately; Microsoft runs a cascade (recognition, English text, translation, voice) and can simulate the speaker's voice; OpenAI streams speech-to-speech. None publishes a lag number. "A few seconds behind" is a human-interpreter-style ear-voice span, so our 3 s goal is the right target, but nobody has shown how it is measured.

## Four flaws in how we measure (checked in code or data this session)
1. **Voxtral lag is understated.** `paced_replay.py` takes `source_end` from the span the server reports. For Whisper that is the word's real end time; for Voxtral it is the time of the audio push that completed the word (`backend/main.py`, `_push_stream_words`), i.e. at least the model's own delay (about 0.5 s) plus the push interval late. Voxtral and Whisper lag numbers are therefore not comparable until Voxtral spans use reference word times (forced alignment of the source audio, offline).
2. **The lag number is a sum nobody can take apart.** `final_metrics` has stt/mt/tts time but not the fragment hold or the playback wait. In the saved Transformers-Voxtral run the time from "word pushed" to "piece emitted" was near 0 s for short pieces but 2.5 s, then 3.3-6.6 s for the long pieces late in the minute (`replay_en-de_voxtral-agree_short.json`, one bogus 534 s span ignored), while MT (0.1-0.6 s) and TTS start (0.1-0.3 s) stayed small. That run is the engine at 1.0-1.1 x real time, so the growth is the recogniser falling behind, not the fragment policy; the vLLM run will show whether it disappears.
3. **Adaptive playback is not credited.** For the same replay, lag median, p90 and max are identical at rate 1.0, 1.25 and "adaptive" (3.15 / 6.71 / 7.12 s); only the tail differs (10.4 / 9.4 / 9.9 s). If arrival lateness dominates, that is expected; if not, the scorer hides a gain. Unresolved: needs one replay where the queue, not the recogniser, is the bottleneck (Piper pieces longer than the source).
4. **Test material is not conversation.** Generated Piper speech and a slow, paused reading. No overlapping speech, no natural pace, no real room. Quality is scored by word error only; nobody has judged intonation or seams between pieces.

## Is the pace policy the missing lever? (code facts, not measurements)
- `RateFollower` clamps the voice to 0.85-1.15 x (`backend/prosody.py:26`) and the pause mirror adds silence. German and Slovak translations are usually longer than the English source, so a 15% cap can leave a structural deficit. Not measured here: the actual German/English audio-length ratio per piece. Cheap test: sum TTS seconds over source seconds for the existing replays; widen `RATE_MAX` offline in the listener model and see what the scorer says.
- Piper is never the bottleneck (0.12 s for a piece). Its limits are the generic voice and the seam between independently synthesised pieces.
- OmniVoice is the same pipeline with a 0.8-1.0 s fixed cost per piece on a T4 (October data) and a cloned voice. Its `speed` now follows the rate follower (`test/omni_pace_tests.py`); `duration` from the source span is not built.

## What a configuration "like the big three" would be (design, untested end to end)
Streaming recogniser (Voxtral on vLLM: T4 lag 0.08 s after the audio) -> piece policy (2-6 words or a clause, 0.8 s cap) -> translation -> TTS per piece (Piper default, OmniVoice cloned voice on a GPU) -> rate follower with a wider cap, plus adaptive playback -> a lag budget of recogniser 0.6 s + hold 0.5-1 s + MT 0.2 s + voice 0.1-1 s. The sum is 1.4-2.8 s on paper. The route test with vLLM (queued on Kaggle) is the first time all stages run together.

## Better instruments (smallest first)
1. Per-stage timestamps in `final_metrics` (hold, MT, TTS start, queue wait) so a lag number can be taken apart.
2. Reference word times for the source (offline alignment, run once per test clip) so every engine is scored against the same clock.
3. TTS/source duration ratio per piece and direction, and an offline `RATE_MAX` sweep on saved replays.
4. A real two-person recording and an ear panel for seams and pace (owner), scored with the existing `score_success.py`.
