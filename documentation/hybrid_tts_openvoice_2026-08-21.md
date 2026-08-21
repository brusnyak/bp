# HybridTTS / OpenVoice: unblocked, fixed, benchmarked (2026-08-21)

## Context

Prior status (T012, `personal_voice_bootstrap_2026-08-19.md`, `tts_registry_verification_2026-08-21.md`):
`openvoice` had no PyPI release and was believed to fail building from GitHub source on this
toolchain due to a Cython/pyav incompatibility. `HybridTTS` was guarded with a clean `ImportError`
and never actually exercised. This was the architecturally-correct answer to the project's core
goal — fast Piper synthesis in the target language, then OpenVoice tone-color conversion to the
user's cloned voice, avoiding XTTS v2's slow full cross-lingual generation (RTF 1.46-1.72) — but
it was unverified and blocked.

## What was actually blocking it

The Cython/pyav error was real but a red herring: it only showed up because pip's isolated build
environment tried to build `av` from source using a version incompatible with the installed Cython.
Installing with `--no-build-isolation` (use the venv's already-satisfied, modern `av==15.1.0`
instead of building a fresh one) got past that immediately.

The actual blocker: `openvoice`'s own `setup.py` hard-pins 2022-era dependency versions —
most critically `numpy==1.22.0`, which does not build on Python 3.11 (`ModuleNotFoundError:
distutils.msvccompiler`, a Windows-only shim numpy 1.22 assumes exists). This isn't a build
quirk, it's an abandoned package: no updates since roughly 2023, pinned to versions that
predate this project's entire modern stack (faster-whisper 1.2.0 vs its pinned 0.9.0, numpy
1.26.4 vs its pinned 1.22.0, etc).

**Fix**: install with `--no-build-isolation --no-deps`, letting it run against this project's
own already-pinned, modern versions instead of its stale ones. This surfaced a chain of missing
transitive imports one at a time (`whisper_timestamped` → `openai-whisper` → `tiktoken` →
`dtw-python` → `eng_to_ipa` → `cn2an` → `proces` → `langid` → `wavmark` → `resampy`), each
installed the same way. All are small, pure-Python or light packages; none pulled in a
conflicting numpy requirement — **except `dtw-python`, once**, which silently upgraded numpy to
2.2.6 and broke ABI compatibility for `numba`/`accelerate`/`pandas` project-wide (`pandas`
raised `ValueError: numpy.dtype size changed, may indicate binary incompatibility` on import).
Caught immediately via a real smoke test (not assumed fine), rolled back to the pinned
`numpy==1.26.4`, reverified with `test/hardware_test.py` (7/7 passing) and a live Piper
synthesis before continuing.

OpenVoice V2 checkpoints (`config.json` + `checkpoint.pth`, 131MB) are not on PyPI or bundled;
downloaded from HF hub `myshell-ai/OpenVoiceV2` to `models/openvoice_v2/checkpoints_v2/converter/`
(gitignored — not committed, same treatment as other large model artifacts in this repo).

## A real bug this surfaced in `hybrid_tts.py` (pre-existing, never caught because never run)

`HybridTTS.synthesize()` was re-extracting the **source** speaker embedding (Piper's own fixed
base voice) from the just-generated per-call output every single time. Two consequences:

1. OpenVoice's `se_extractor.get_se()` asserts `input audio is too short` below a minimum
   duration — reproduced directly: a single Piper sentence (~4.1s) fails; a 41s reference clip
   works. Nearly any real single-utterance TTS call would have hit this and silently fallen back
   to uncloned Piper output via the existing `except Exception` handler — the fallback masked the
   bug rather than crashing loudly.
2. Piper's base voice never changes between calls, so re-deriving its embedding every time was
   pure wasted latency even when it didn't crash.

**Fix** (root cause, not the failure site): the source embedding is now extracted once in
`HybridTTS.__init__` from a longer bootstrap synthesis (~17s of generic English text — the
language of the bootstrap text doesn't matter, only its duration and that it comes from Piper's
own voice) and cached as `self.source_se`. `synthesize()` now reuses it unconditionally. This
removes the crash risk entirely (bootstrap text is always long enough) and removes a full
embedding-extraction pass from every call's critical path.

## Real, independently-measured numbers (M1 Pro, backend `mps`)

Target-voice embedding extraction (`_get_target_se`, keyed by `speaker_wav_path`) was already
cached per-target in the pre-existing code — first call for a never-before-seen target voice
pays a one-time cost (~4.4s here, since it also cold-loads the XTTS-adjacent conversion model
graph on first use); every subsequent call for that same target voice is fast:

| Language | Latency | Audio duration | RTF |
|---|---|---|---|
| Slovak (sk) | 0.691s | 4.10s | **0.169** |
| German (de) | 0.579s | 3.94s | **0.147** |
| English (en) | 0.561s | 3.60s | **0.156** |

All three from the same single English reference recording (`speaker_voices/voice_rec_1m.wav`),
converted at request time to the correct language's Piper voice with the reference speaker's
timbre applied.

**Compare to XTTS v2** (same-day measurement, `documentation/` earlier this session): DE
cross-lingual clone RTF 1.46, load time ~29s. Hybrid is **~10x faster** and stays under RTF 1.0
(faster than real-time) — the first cross-lingual cloned path in this project to do so.

## Proven / assumed / unknown

**Proven**: HybridTTS installs, loads, and produces real cross-lingual cloned audio on this
machine, at the RTF numbers above, independently re-run twice with consistent results (0.169 vs
an initial cold-cache 1.055 for SK confirms the caching behavior is real, not a fluke).

**Assumed, not yet measured**: subjective voice-likeness quality of the tone-color-converted
output vs. XTTS's full-generation cloning — no resemblyzer/listening comparison done yet between
the two cross-lingual approaches (only latency was benchmarked here). Samples were sent to the
user directly for a listening judgment call, alongside the earlier XTTS DE sample, since no
automated proxy replaces that for a voice people will actually hear.

**Unknown**: watermarking was left enabled (`wavmark` installed rather than disabled, since the
library's own `enable_watermark=False` kwarg path is broken — see code comment) — unverified
whether the embedded watermark is audible or affects perceived quality; worth a listen. Behavior
under concurrent/multi-session load (would each session's `HybridTTS` instance hold its own
target-embedding cache, multiplying memory per the existing per-session-model-instance
architecture?) — not tested, same caveat as every other engine per the deferred T018.

## What this changes

`piper_personal`/`piper_personal_v2` remain the fastest path but are English-only, fixed voice,
requiring a full ~21min retrain per new language with real target-language recordings. `xtts`
remains the only true zero-shot any-language-in path but is slow (RTF 1.46+). **`hybrid` is now
the answer to "fast + cross-lingual + cloned from one recording"** — the core ask behind this
whole feature — verified working, not just architecturally intended.
