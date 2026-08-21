# TTS engine registry refactor — verification, 2026-08-21

Verifies the `backend/hardware.py` + `backend/tts/base.py` (`TTS_ENGINES` registry) refactor
landed earlier the same session, which replaced a 4-way if/elif TTS-engine dispatch that was
duplicated in five places across `backend/main.py` with a single registry. That refactor was
syntax-checked and unit-tested at the time but never run against the real pipeline — no working
project venv existed on this machine. This session built one and re-verified for real.

## Environment built

No `venv/` existed in the project, and none of this machine's existing conda envs (`TTS`,
`xtts_env`, `rvc_env`) had the full stack together. Built `./venv` with `~/.local/bin/python3.11`
(Python 3.11.15, matching the `Makefile`'s pin) and installed `requirements.txt`.

**Real blocker found and fixed**: `omnivoice==0.1.4` requires `transformers>=5.3.0`, which
conflicts with the `transformers==4.46.3` pin `TTS==0.22.0`/`faster-whisper` need — a genuine,
pre-existing dependency conflict in `requirements.txt`, not something this session introduced.
Installed everything else (`pip install -r requirements.txt` minus the `omnivoice` line);
`backend/tts/omni_tts.py` already guards its own `from omnivoice import OmniVoice` import with
try/except, so this doesn't break anything importing it.

**Second real blocker found and fixed**: `backend/tts/hybrid_tts.py` hard-imports `openvoice` at
module level. `openvoice` has no PyPI release and building it from GitHub source
(`pip install git+https://github.com/myshell-ai/OpenVoice.git`) fails during its `av` dependency's
Cython build (`av/logging.pyx` — a `noexcept`/exception-spec incompatibility against this
machine's newer clang). This isn't new: `openvoice` was never in `requirements.txt` in the first
place — it was installed ad hoc in whatever earlier session wrote `hybrid_tts.py`, and the
OpenVoice V2 checkpoint files it also needs (`models/openvoice_v2/checkpoints_v2/converter/`)
aren't on disk on this machine either. Since `backend/tts/base.py` (and therefore `backend.main`,
and therefore every test via `conftest.py`) imports `hybrid_tts` eagerly, this blocked *all*
testing, not just HybridTTS's own. Fixed by guarding the import exactly the way `omni_tts.py`
already guards `omnivoice` — `HybridTTS()` now raises a clear `ImportError` naming the install
command and checkpoint path if constructed without the package, instead of the whole app failing
to import.

Confirmed working: `fastapi`, `torch`, `piper`, `TTS.api` all import; `from backend.tts.base
import TTS_ENGINES` imports cleanly; `from app import create_app; create_app()` boots a real
FastAPI app instance.

## Tests run — real venv, real models, not toy

- `test/hardware_test.py`: **7/7 passed**.
- `test/piper_pipeline_test.py` (run as a script per `make test`'s own convention — it has no
  pytest-discoverable `test_*` functions despite pytest-style names in some sibling files):
  real STT→MT→TTS run through the refactored dispatch. STT WER 0.0000 (perfect transcription).
  MT "FAILED" on one exact-string comparison (paraphrase variance vs. the ground-truth string,
  e.g. "klonovanie" vs "účely klonovania") — pre-existing MT model behavior, MT code wasn't
  touched by this refactor. TTS synthesized real audio via `PiperTTS: using backend 'cpu'` (the
  new `hardware.py` path, live).
  - **Found and fixed a genuine pre-existing bug, unrelated to the TTS refactor**: this script
    imports `from test.voice_similarity import calculate_speaker_similarity`, but
    `test/voice_similarity.py` was deleted in commit `fa0d750` ("chore: remove deprecated
    Coqui/XTTS experimental test files") without updating this import — meaning `make test`'s
    first line has been broken since that commit, independent of anything this session touched.
    Restored the file verbatim from git history (`git show fa0d750^:test/voice_similarity.py`).
    It was already a stub before deletion (returns `-1.0`, "not supported as F5-TTS has been
    removed") — restored as-is rather than inventing a new implementation, since a real
    resemblyzer-based similarity metric already exists at `scripts/voice_similarity_qc.py` and
    that script's own docstring explicitly says not to install `resemblyzer` into the main
    project venv (its `webrtcvad` dependency needs `setuptools<81`, which the main venv doesn't
    pin). Fabricating a second, different similarity implementation here wasn't in scope.
- `test_full_pipeline.py` (also script-style — confirmed via its own `if __name__ == "__main__":`
  guard, not pytest fixtures, despite `pytest` initially reporting fixture errors when
  mis-invoked): 3 real Piper pipeline runs. TTS latency 0.0967s / 0.1793s / 1.1526s
  (text-length-dependent) — consistent with the ~0.35s single-sentence Piper baseline in
  `documentation/TTS_COMPARISON_REPORT.md`. No regression.
- `test/xtts_registry_smoke_test.py` (new, this session): loads `TTS_ENGINES["xtts"]()` for
  real (XTTS v2 weights were already cached locally), confirms `SUPPORTS_STREAMING=True`,
  `LANGUAGE_OVERRIDES={'sk': 'cs'}`, `REQUIRES_SPEAKER_WAV=True` are all live and correct on the
  actual loaded engine, and calls `synthesize_stream()` with a real speaker reference
  (`speaker_voices/voice_rec_1m.wav`) — produced 4.05s of real audio, written to
  `test_output/xtts_registry_smoke.wav`. This is the highest-risk path the refactor touched
  (streaming + language-proxy logic moved from inline `main.py` branches to declarative class
  attributes) and it works end-to-end with the real model.
- `test/piper_personal_smoke_test.py` (new, this session) — see next section.

**Not run**: `benchmark_full_pipeline.py`'s full report and the 12-concurrent-user load test
methodology from `PERFORMANCE_TEST_RESULTS.md` (that's T019's scope, separate and
longer-running than single-request latency checks). `HybridTTS`/`OmniVoiceTTS`'s actual
clone-with-fallback behavior — blocked by the two environment gaps above, not by anything in the
refactor itself.

## Personal voice (`en_US-personal-medium.onnx`) — now reachable

Added a `piper_personal` factory to `backend/tts/base.py`'s `TTS_ENGINES` registry (mirrors the
existing `piper` factory, just points at `model_id="en_US-personal-medium"`). Because the whole
point of the earlier registry refactor was "add an engine without touching `main.py`," this
required editing exactly one file. `tts_model_choice="piper_personal"` is now a valid selection
from the client.

**Real measured result**: loaded via `TTS_ENGINES["piper_personal"]()`, synthesized a 116-character
test sentence → 7.01s of audio in 0.3128s → **RTF 0.0446**. Consistent with the 0.0562 (5-run
average) already documented in `documentation/personal_voice_bootstrap_2026-08-19.md` for the
same model — same order of magnitude, not a fluke, not a different model being loaded by
mistake. `hardware.py` selected `coreml` as the backend name for this run; per the existing
`ponytail:` note in `piper_tts.py`, that name selection doesn't yet route into an actual
onnxruntime CoreML execution provider (`piper-tts`'s `PiperVoice.load()` doesn't expose a
`providers=` argument in the installed version), so this RTF is CPU-class execution, not
confirmed CoreML acceleration — flagged, not overclaimed.

**Scope note**: this makes the already-trained voice *selectable*, not the six-tier automatic
fallback cascade described in T028/T031-T033 (fine-tuned personal → GPT-SoVITS → XTTS zero-shot
→ generic Piper). That's real, separate, larger scope, correctly deferred in `tasks.md`.

## Files touched this session

- `backend/tts/hybrid_tts.py` — guarded `openvoice` import (see above).
- `backend/tts/base.py` — added `piper_personal` factory + registry entry.
- `requirements.txt` — removed the stale `# prioritize MPS support` comment/commented-out index
  URL; backend selection is now documented as `backend/hardware.py`'s job, not an install-time pin.
- `test/voice_similarity.py` — restored (deleted by mistake in `fa0d750`, unrelated to this work).
- `test/xtts_registry_smoke_test.py`, `test/piper_personal_smoke_test.py` — new, this session.
- `specs/001-realtime-cross-platform-translation/tasks.md` — T003, T007, T008, T010 marked
  done with evidence; T006, T011, T012, T028 marked partial with an honest description of what's
  actually verified vs. not; T004, T005, T009 confirmed still open, out of this pass's scope.

## Proven / assumed / unknown

**Proven**: hardware.py's selection logic (unit tests, real venv); Piper dispatch through the
registry (real STT→MT→TTS run, real audio, WER 0.0); XTTS streaming+cloning+language-override
dispatch through the registry (real audio via real model); `piper_personal` is selectable and
produces real audio at the previously-documented RTF.

**Assumed**: that CoreML backend *selection* implies CoreML *execution* for Piper — explicitly
not true yet, flagged in three places (code comment, tasks.md, this doc).

**Unknown**: whether `HybridTTS`/`OmniVoiceTTS`'s fallback-to-generic-voice behavior still works
— untestable on this machine until `openvoice` or `omnivoice` become installable here; whether
the refactor holds up under the 12-user concurrency load test (T019, not run).
