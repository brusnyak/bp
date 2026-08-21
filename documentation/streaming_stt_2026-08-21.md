# Incremental streaming STT (T034-T036): SimulStreaming rejected, LocalAgreement verified viable (2026-08-21)

## SimulStreaming: investigated, not used

`ufal/SimulStreaming` (2025 successor to whisper_streaming, AlignAtt policy) was the newer option
named in prior research. Its own README rules it out for this hardware before any code was run:
"Recommended HW is GPU with at least 10G VRAM, to run the best-performing Whisper model large-v3...
The code works also on CPU but would be too slow for real-time." It also wraps `openai-whisper`
(a different, heavier STT stack than this project's `faster-whisper`), needs a 1.5B-param model,
and its `requirements_whisper.txt` pulls `torch`/`torchaudio`/`triton` fresh. No CUDA exists on
this M1 Pro. Not attempted further — the authors' own stated hardware floor disqualifies it here,
not worth burning install time to confirm empirically.

## whisper_streaming (LocalAgreement): the right-scoped choice, verified working

Older, lighter, and — critically — supports `faster-whisper` as a backend directly, matching what
this project already uses (`backend/stt/faster_whisper_stt.py`, `base` model, int8). No PyPI
release from the actual `ufal/whisper_streaming` project (a same-named PyPI package exists but is
an unrelated, unofficially-repackaged project with a suspicious backdated file timestamp — not
used). Vendored the 3 real source files directly from `github.com/ufal/whisper_streaming`
(MIT-licensed) into `backend/stt/whisper_streaming_vendor/`.

**One real patch needed**: `FasterWhisperASR.load_model()` upstream hardcodes
`device="cuda", compute_type="float16"` (their dev box). Changed to `device="auto",
compute_type="int8"` to match this project's own already-working `faster_whisper_stt.py` config —
this project has no CUDA path on macOS (`backend/hardware.py`'s stt/mt chain is
`cuda→rocm→cpu`, no MPS support in CTranslate2).

## Real numbers, this M1 Pro, `base` model, int8

Ran the vendored `whisper_online.py`'s own real-time-simulation-from-file mode (wall-clock,
computationally aware — not the `--comp_unaware` shortcut) against
`test/My test speech_xtts_speaker_clean.wav` (29.4s, 65 words):

**Without `--vac` (no VAD gating, plain LocalAgreement on fixed 1s chunks)**: unstable. Falls
behind real time as the clip progresses — per-chunk processing lag grows from ~1s early on to an
8s gap, then a **15.6s lag on the final chunk alone**. Total wall time 33.8s for 29.4s of audio —
worse than real-time overall (effective RTF ~1.15, and the backlog was actively growing, not
stable).

**With `--vac` (Silero VAD-gated, the authors' recommended default)**: stable. First partial
transcript at **3.3s**. Final output at 29.4s wall time for 28.2s of confirmed speech — **RTF
~1.001, no growing backlog**. Re-ran on `test/Hello.wav` (4.46s) for a second data point: correct
transcription, no backlog, consistent behavior.

**Root cause of the difference**: without VAD gating, every 1s tick forces a fresh Whisper call
regardless of whether there's a natural pause to trim the buffer at, so the audio context fed to
the model keeps growing unboundedly within an utterance. VAD-based segmentation resets/bounds the
buffer at actual speech boundaries, which is exactly why `--vac` is the recommended default, not
an optional extra.

## The actual failure mode this project hit before: reproduced as fixed, not just assumed

The original disable reason was hallucination on short non-speech sounds. Built a synthetic
5.3s test clip (near-silent noise floor, then a short 0.3s noise burst, then more near-silence —
deliberately not real speech) and ran it through the `--vac` path: **zero transcription output,
zero hallucination**. The Silero VAD gate correctly never classified it as speech, so it never
reached Whisper at all. This is the mechanism, not a fluke — VAD-gated LocalAgreement structurally
can't hallucinate on non-speech the way the old fixed-chunk approach could, because non-speech
never gets submitted for transcription in the first place.

## Why main.py wasn't touched this pass

The old disabled code path (`backend/main.py` around line 1216) didn't do incremental
LocalAgreement at all — it re-ran the **full STT→MT→TTS pipeline**, including a fresh
independent TTS synthesis call, on every fixed-length streaming chunk (see
`_process_speech_segment_pipeline`, called with `is_final=False` at that old call site, which
still triggers TTS — see line ~902's `_save_audio_segment_for_debug(audio_wav, ...)` running
regardless of `is_final`). That's architecturally why it hallucinated: independent, context-free
Whisper calls on short isolated chunks is a known Whisper failure mode, and it would have also
caused repeated/stuttering audio output from re-synthesizing growing partial text via TTS on every
tick.

T035's actual scope is narrower and correct: "live partial transcription/translation **captions**"
— text only, not incremental TTS. Wiring that in requires a per-session, persistent
`VACOnlineASRProcessor` instance (mirroring how `session_data["stt_model"]` already tracks a
persistent model per session), fed from the existing audio-chunk receive loop, with partial
results sent to the client as caption-only updates — while the existing
`speech_frames`/`SILENCE_TIMEOUT`-based final-segment detection (already verified regression-free,
T011) continues completely unchanged for the STT→MT→TTS path that actually produces spoken output.

This is real, scoped, additive work — not done this pass. Reasons: (1) it touches the live
WebSocket audio loop that T011 already verified as regression-free post-refactor, and rushing a
change into it without dedicated session-lifecycle testing risks breaking a currently-working
path late in a long session; (2) the feasibility question (does LocalAgreement work at all on this
hardware without hallucinating) is what actually needed answering first, and that's now settled
with real numbers. T035 is unblocked and well-specified as a follow-up, not a research question.

## Proven / assumed / unknown

**Proven**: LocalAgreement + Silero VAD gating (`--vac`) runs at real-time-or-better (RTF ~1.0) on
this M1 Pro with `faster-whisper base`/int8, doesn't hallucinate on non-speech, first partial in
~3.3s.

**Assumed, not measured**: behavior on genuinely long-form continuous speech (multi-minute), on
non-English languages, and under concurrent multi-session load (each session would need its own
`VACOnlineASRProcessor` + Silero VAD instance — same per-session-memory-multiplication caveat as
every other engine in this project, T018).

**Unknown**: whether wiring this into `main.py` for captions-only actually improves perceived
UX enough to matter for the demo, versus the current SILENCE_TIMEOUT-based final-segment-only
approach which already works and is verified. That's T036's question, explicitly deferred until
T035 is actually built.

## Status

T034: **done** — feasibility proven with real numbers, not assumed. SimulStreaming correctly
rejected (hardware-disqualified per its own docs), LocalAgreement+VAC verified as the right,
working choice.
T035: **not done, well-specified** — wiring into `main.py` behind the existing `is_final=False`
path, captions-only, is real follow-up work with a concrete design, not started this pass.
T036: **blocked on T035** — nothing to compare before/after yet.
