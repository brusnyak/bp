# Real-time pipeline rearchitecture (T043-T047): non-blocking, captioned, interruptible (2026-08-22)

## Context

Following `documentation/simultaneous_translation_research_2026-08-22.md`'s verdict (no viable
path to sub-2s incremental MT without an unacceptable quality cost or a from-scratch ML project),
the user's scope decision was: don't chase Google's 2-3s ceiling with a new model — fix what's
actually broken in this project's own architecture. Direct code inspection that same session found
the real problem was never just "STT+MT+TTS takes ~4s" — it was that **capture blocked entirely
while translation ran**, and there was **zero interruption/barge-in handling**. This document covers
T043-T047: fixing that, wiring in the already-verified streaming STT as live captions, chunking TTS
output, and verifying barge-in actually works — plus every real bug found and fixed along the way,
each caught by live-testing before being called done, not assumed safe.

## T043: non-blocking pipeline + safe cancellation

`_process_speech_segment_pipeline` calls converted from a bare `await` inline in the same loop that
reads audio off the websocket, to `asyncio.create_task` (tracked per-session as
`active_pipeline_task`). Capture/VAD now runs continuously regardless of whether a translation is
in flight.

**Real correctness bug found and fixed, live**: the literal original design ("a new speech-start
event cancels any in-flight pipeline task") was tested against an 8-sentence continuous speech clip
and **silently dropped 3 of 8 sentences** — natural inter-sentence pauses are shorter than one
STT+MT round trip, so each new sentence cancelled the previous one's still-running
transcription/translation before it could send results. Fixed with a per-session `model_call_lock`
(`asyncio.Lock`) around STT+MT only. Barge-in cancellation now only fires when that lock is *not*
held — i.e. the previous task is in/past TTS, the only phase where overlapping spoken output is
actually a problem. Re-verified on the same clip: **0/8 dropped**.

**Second real bug found and fixed, live**: `KeyError: 'model_call_lock'` on first real run. Root
cause: two independent code paths initialize a session's dict (`initialize_all_models` at
`main.py:364`, called by `app.py`'s `/ws` route *before* `handle_audio_stream`'s own
`active_sessions.setdefault` at `main.py:965`) — pre-existing tech debt (two hand-maintained copies
of the same default-session shape). The new keys only existed in one. Both now carry them. Not
consolidated into one source of truth this pass — flagged, not fixed, to keep the diff minimal.

Regression: direct A/B against unmodified `main.py` confirmed the one server-log error seen
(`Cannot call "receive" once a disconnect message has been received`) is pre-existing, not a
regression from this work.

## T044: streaming captions (additive)

New `backend/stt/streaming_captions.py` (`SessionCaptioner`) wraps the already-verified LocalAgreement
+VAC processor from T034, using the exact construction the 2026-08-21 benchmark measured (`--vac`, no
`--vad`, min-chunk-size 1.0s, `buffer_trimming=("segment",15)`) — not an untested deviation.

Wired as a fire-and-forget side channel: every audio chunk feeds it (bounded by a `captioner_lock` —
a chunk is *dropped*, never queued, if the previous `feed()` call is still running, so this can never
build backlog); a committed partial goes out as a new `caption_partial` message. `_run_pipeline_task`
(T043's single funnel point for every trigger) resets the captioner at the start of each real segment.
**Never touches or gates** the SILENCE_TIMEOUT final-segment path — the real transcript/translation/
audio still come from the primary STT model exactly as before.

Live-verified: first caption at 3.41s vs. that sentence's real `transcription_result` at 5.19s —
matches the ~3.3s figure from the original benchmark. All 8 sentences' real output still correct.

**Known limitations, not fixed**: loads a second whisper model ('base', int8) per session — doubles
STT memory per session, same deferred concern as every other per-session model in this project
(T018). Frontend doesn't render `caption_partial` yet (backend plumbing only, by design of this task).
The "finalize-of-already-computed-partial" latency win described in the original task wording was
**not** implemented — the final segment still runs STT from zero. Wiring the streaming processor's
committed state into the *final* (not just caption) output was judged out of scope: it would make
the final transcript load-bearing on this additive path, changing its risk profile, and would need
its own correctness pass (word-level alignment between the streaming buffer and the final full-segment
audio).

## T045: chunked TTS output

New `backend/tts/text_chunking.py`: `split_into_phrases` (punctuation-based clause splitting) and
`chunk_audio_by_duration` (~150ms slices). `PiperTTS` and `HybridTTS` both gained
`SUPPORTS_STREAMING = True` and a `synthesize_stream()` that phrase-splits the translated text,
synthesizes each phrase with the engine's existing, unchanged `synthesize()`, and sub-chunks the
result — decompose-into-smaller-calls, not a new synthesis path, so it inherits all of `synthesize()`'s
existing correctness (caching, fallback behavior).

**Two more real bugs found and fixed, live**:
1. `HybridTTS.sample_rate` was hardcoded `16000` and never actually read anywhere before this task
   (dormant — the non-streaming path always read sample rate from `synthesize()`'s own return tuple).
   Confirmed the real value directly from the OpenVoice V2 converter's own
   `models/openvoice_v2/checkpoints_v2/converter/config.json` (`sampling_rate: 22050`) and fixed it.
   This would have played every cloned streaming chunk at the wrong pitch/speed the moment streaming
   started reading this attribute — not a hypothetical, it's exactly what main.py's streaming consumer
   does (`getattr(tts_engine, "sample_rate", 24000)`, read once, applied to every chunk).
2. `main.py`'s TTS dispatch had `elif tts_model_choice != "xtts":` as its "did we get audio"
   fallback-error check — written when XTTS was the only streaming engine. With piper/hybrid now also
   streaming, this fired on **every successful streaming synthesis**, sending the client a false
   `"TTS model not ready"` error and returning before `final_metrics` was ever sent. Caught by a live
   smoke test (not shipped), fixed by generalizing to `not getattr(tts_engine, "SUPPORTS_STREAMING", False)`
   — the same flag the dispatch above it already branches on, instead of a second hardcoded
   engine-name check that had silently drifted out of sync with it.

Verified: real chunk sizes ~150ms (3307 samples @ 22050Hz for Piper/Hybrid), correct total duration
on reassembly, clean live smoke test (WER 0.0), 8-sentence clip still 0 dropped post-fix.
HybridTTS's streaming path was verified standalone (correct chunking, real cloned audio) but not
through a live end-to-end WS test with a real `speaker_wav_path` — the existing test harness doesn't
pass one, and it shares the exact integration code path already live-verified via Piper.

## T046: barge-in — verified live, not just by design

While re-running the interrupt smoke test after T045, the server log captured a genuine mid-TTS-send
cancellation **on its own**, without a specially engineered test case: segment "Speak trick ignition,
machine translation, and text to speech." started streaming TTS at 07:15:03.634; the next segment's
speech began 88ms later; cancellation fired at 07:15:03.723 — within ~1ms of the new speech being
detected. Confirms the chunked TTS send loop (`main.py`'s `while True: chunk = await queue.get()`) is
cancelled near-instantly, because it awaits a plain `asyncio.Queue`, not a `run_in_executor` future.
The earlier-documented caveat ("cancellation is deferred until the blocking call returns") applies to
STT/MT and non-streaming-TTS `run_in_executor` calls — **not** to the chunk-send loop once T045 made
TTS chunked. The background `run_streaming_tts` daemon thread is not itself killed (both remaining
phrases were still synthesized and logged after cancellation — wasted CPU, bounded and harmless), but
the queue-draining/sending loop stopped, so no further `tts_audio` bytes for the cancelled segment
reached the client.

**Real, non-obvious finding — flagged, not silently resolved**: this cancellation fired between two
*natural* back-to-back sentences in one continuous test clip, not a deliberate interruption (the test
script never sent a second utterance at that point). Fast continuous multi-sentence speech can
truncate the *previous* sentence's spoken audio once the next sentence's VAD speech-start fires —
the same class of ambiguity already fixed for STT/MT (T043), now visible one layer up, at TTS. It was
**not** fixed here: TTS cancellation is deliberately left unconditional (any new speech-start cancels
in-flight TTS), because that is barge-in's actual job. Whether this is the right behavior is framing-
dependent: for a genuine interruption it's exactly right; for a fast monologue of short sentences, it
means only the last sentence before a real pause reliably gets its full audio delivered. This is the
same shape of tradeoff EMMA's 66%-BLEU-for-speed choice represents at the MT layer (research doc,
section 1) — keep pace vs. deliver everything. **A product decision, not a bug.**

## T047: regression + soak

- `test/hardware_test.py`: 7/7 passing throughout every stage of this work (baseline, unaffected —
  this file tests hardware backend selection, not the pipeline).
- Normal non-interrupted single-utterance smoke test: clean at every stage (WER 0.0, correct
  transcription/translation/audio, no errors).
- Soak (`test/soak_interrupt_cycles.py`, 10 back-to-back interrupt/resume cycles in one session,
  alternating `Hello.wav` / `Can you hear me_.wav` with a deliberate short-silence-then-interrupt
  between them): **completed in 84.1s, no hang.** 20/20 transcription+translation results received
  (10 cycles × 2 utterances, none dropped/hung). 0 errors. Server RSS: 1,954,416 KB → 1,697,088 KB
  (**-257 MB**, i.e. no leak — the decrease reflects GC settling from earlier test runs in this same
  session, not this soak test's own effect, but rules out any growth trend across 10 real cycles).

**Real end-to-end latency, post-T043-T046, N=2 (directly comparable to T038's 4.02s baseline)**:
3.64s and 5.01s, avg **4.33s**. This is flat/noise-band-equivalent to the pre-work baseline, **not
a reduction** — and that's expected, not a miss: T043-T046 never touched STT/MT/TTS compute speed for
a single non-interrupted utterance. What they fix is architectural dead time and interruption
handling, which this single-utterance-latency number doesn't capture at all. Reporting this plainly
rather than letting the number imply a speed win that wasn't the target.

## What actually changed, in plain terms

| Before (start of this session's T043 work) | After |
|---|---|
| Capture blocked for ~4s during every translation | Capture never stops; VAD/audio processing runs continuously |
| No interruption handling at all | New speech cancels a stale in-flight pipeline — verified live, cancels within ~1ms during TTS |
| Naive cancellation would have dropped ordinary multi-sentence speech | STT+MT protected by a lock; only TTS (the actual "two voices" case) is cancellable |
| No captions; first sign of transcription was the final result, ~5s in | Partial captions arrive ~3.3-3.4s ahead of the final result |
| TTS sent as one blob per sentence | TTS sent as a train of ~150ms chunks, enabling fast, clean cancellation |
| Single-utterance E2E latency: ~4.02s | Single-utterance E2E latency: ~4.33s (unchanged, not the target of this work) |

## Proven / assumed / unknown

**Proven** (live-verified, not assumed): non-blocking capture; STT/MT correctness under rapid
continuous speech (0/8 dropped, was 3/8 with the naive design); captions arriving ahead of final
output; chunked TTS producing correct audio; barge-in cancelling TTS near-instantly, confirmed by an
organically-occurring real event, not just a constructed test; 10-cycle soak with no hang and no
memory growth; two dormant/newly-triggered bugs (HybridTTS sample rate, the `!= "xtts"` dispatch
check) caught before being called done.

**Assumed, not verified**: HybridTTS's streaming path through the full live WS integration with a
real `speaker_wav_path` (verified standalone + via the shared code path, not via its own live WS
test). Behavior under genuinely long-form (multi-minute) continuous speech. Behavior under concurrent
multi-session load with the new per-session locks and the second captioning model (T018's existing
concurrency-scaling caveat is unchanged by this work, not newly introduced).

**Open, flagged for a decision, not a bug**: whether TTS barge-in should stay unconditional (current
behavior — correct for genuine interruptions, truncates trailing audio in fast monologues) or should
get its own STT/MT-style "was this really new speech or just the next sentence" heuristic. Not
resolved here.

## What's next

- T018 (shared model pool) remains explicitly out of scope (user's own prior scope call).
- Frontend rendering of `caption_partial` — backend plumbing is done, UI is not, by design of this task.
- The barge-in-vs-monologue-completeness tradeoff above, if it matters for the actual demo.
- Otherwise: this closes the engineering-only path to "feels near-real-time" that was scoped after
  the simultaneous-MT research came back negative. Per the user's own framing, focus moves to
  documentation/thesis writeup next.
