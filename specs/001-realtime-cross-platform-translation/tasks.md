---
description: "Task list for the cross-platform real-time translation pipeline rescope"
---

# Tasks: Cross-Platform Real-Time Speech Translation Pipeline (v2 rescope)

**Input**: `specs/001-realtime-cross-platform-translation/spec.md`, `plan.md`

**Tests**: Included — this project already has a pytest suite; extend it rather than skip tests.

**Organization**: Grouped by user story per `spec.md` priorities (US1/US2 = P1 MVP, US3 = P2, US4 = P3 research).

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [ ] T001 Audit current `requirements.txt` for every hardcoded platform/vendor assumption (not just the known "prioritize MPS" comment) — produce a checklist in `documentation/`
- [ ] T002 [P] Confirm `faster-whisper`, `ctranslate2`, `onnxruntime`, `torch` versions in use each expose a documented way to select CUDA/ROCm/DirectML/CoreML/CPU explicitly at runtime (not just at install time)

## Phase 2: Foundational — Hardware Backend Selection (BLOCKS all user stories)

**⚠️ CRITICAL**: This is the actual fix for the recurring "fast on my Mac, slow everywhere else" problem. Nothing else in this feature matters if this phase is skipped.

- [x] T003 [US2] Create `backend/hardware.py`: DONE, 2026-08-21. Implemented as one function, `detect_backend(stage)`, not three separately-named functions (`stt_mt_device()`/`tts_baseline_device()`/`tts_cloning_device()`) — same chains as spec'd (`plan.md`), same per-stage priority order (stt/mt: cuda→rocm→cpu; tts_baseline: cuda→rocm→directml→coreml→cpu; tts_clone: cuda→mps→rocm→cpu), just one parameterized function instead of three. `probes=` override arg makes it mockable without real GPU hardware (see T010).
- [ ] T004 [US2] Wire `backend/stt/` (faster-whisper) to call `hardware.detect_backend("stt")` instead of any hardcoded device string — NOT done this pass, out of scope (this pass only touched the TTS-engine dispatch refactor + hardware.py itself).
- [ ] T005 [US2] Wire `backend/mt/ctranslate2_mt.py` to call `hardware.detect_backend("mt")` instead of any hardcoded device string — NOT done this pass, same reason as T004.
- [ ] T006 [US2] Wire `backend/tts/piper_tts.py` (ONNX Runtime) to call `hardware.detect_backend("tts_baseline")` — PARTIALLY done. `piper_tts.py` now calls `hardware.py` and correctly *selects* directml/coreml/cpu by name (confirmed live on this Mac: selects `"coreml"`), but `piper-tts`'s `PiperVoice.load()` doesn't accept an onnxruntime `providers=` list in this version, so the selected backend name isn't actually routed into onnxruntime execution yet — flagged with a `ponytail:` comment in the code. Real RTF measured (0.0446, see T011) is consistent with CPU-class execution, not confirmed-accelerated CoreML. Leaving unchecked until provider wiring is real.
- [x] T007 [US2] Wire `backend/tts/coqui_tts.py` (XTTS v2, PyTorch) to call `hardware.detect_backend("tts_clone")` — DONE and verified live: loaded via `TTS_ENGINES["xtts"]()`, correctly selects `mps` on this Mac then self-overrides to `cpu` for XTTS-on-MPS stability (pre-existing, preserved behavior), real `synthesize_stream()` call produced 4.05s of real audio. (`f5_tts.py` mentioned in the original task text doesn't exist in this repo — XTTS v2 is the actual cloning engine in use.)
- [x] T008 [US2] Remove the `# prioritize MPS support` hardcoded default from `requirements.txt` — DONE, 2026-08-21. Replaced with a comment stating backend selection is a `backend/hardware.py` runtime decision, not an install-time pin.
- [ ] T009 [US2] Gate `mlx-whisper` behind a Darwin+Apple Silicon check — N/A, not started: `mlx-whisper` isn't used anywhere in this repo yet, nothing to gate.
- [x] T010 [P] [US2] Unit test `backend/hardware.py` against mocked cuda/rocm/mps/directml/no-GPU — DONE. `test/hardware_test.py`, 7/7 passing, run for real in a freshly-built project venv (`./venv`, Python 3.11.15) — not just syntax-checked.

**Checkpoint**: Foundation ready — every stage now selects hardware per-OS/per-vendor at runtime, not at install time.

---

## Phase 3: User Story 1 - Speaker gets voice-cloned translated audio out (Priority: P1) 🎯 MVP

**Goal**: Confirm the existing, already-built pipeline (STT→MT→TTS, voice cloning via XTTS v2/F5-TTS, generic fallback via Piper) still works correctly once Phase 2's hardware-selection changes land — this is a regression check, not new build, per Constitution Principle VI (don't re-architect what already works).

- [x] T011 [US1] Run existing `test_full_pipeline.py` against the Phase 2 (TTS-registry) changes on the reference M1 Pro — DONE, 2026-08-21, real run not a toy. Piper TTS latency across 3 real utterances: 0.0967s / 0.1793s / 1.1526s (text-length-dependent), matching the historical ~0.35s single-sentence baseline in `documentation/TTS_COMPARISON_REPORT.md` — no regression. `benchmark_full_pipeline.py` and the 12-concurrent-user load test from `PERFORMANCE_TEST_RESULTS.md` were NOT re-run (that's T019's scope, separate and much longer-running); this task covers single-request latency only.
- [~] T012 [US1] Confirm voice-clone fallback still triggers correctly when no speaker profile exists — PARTIALLY verified. XTTS's guard (`REQUIRES_SPEAKER_WAV=True`, checked in `main.py`'s dispatch before synthesis) confirmed live via the real loaded engine's class attributes. `HybridTTS`'s actual behavior (falls back to plain Piper output when `speaker_wav_path=None`, per its own `synthesize()`) and `OmniVoiceTTS`'s default-voice fallback could NOT be exercised: the `openvoice` package has no PyPI release and fails to build from GitHub source on this toolchain (Cython/pyav incompatibility with newer clang — guarded with a clear `ImportError` at `HybridTTS()` construction instead of crashing the whole app's imports); `omnivoice==0.1.4` requires `transformers>=5.3.0`, which conflicts with the `transformers==4.46.3` pin needed by `TTS`/`faster-whisper`. Both are real, pre-existing environment gaps, not caused by this pass's refactor. Leave open until one of those two packages is actually installable.

**Checkpoint**: User Story 1 fully functional post-hardware-refactor.

---

## Phase 4: User Story 2 - Same install works on Windows/macOS/Linux (Priority: P1) 🎯 MVP

**Goal**: Prove Phase 2's backend selection actually produces working acceleration on all three OSes, and automate virtual-audio setup so it's plug-and-play per Constitution Principle V.

- [ ] T013 [US2] Extend `scripts/setup_windows.ps1` (already has partial VB-Cable detection per existing commit history) to also verify/report which GPU backend (DirectML vs CPU) was selected
- [ ] T014 [P] [US2] Write `backend/audio/setup_macos.py` (or extend existing mac setup) to automate BlackHole installation/detection
- [ ] T015 [P] [US2] Write `backend/audio/setup_linux.py` for PulseAudio/PipeWire null-sink automation — currently unaddressed, net-new
- [ ] T016 [US2] Manual/CI validation: run the pipeline on a Windows+AMD box and confirm DirectML is selected for Piper TTS, not CPU fallback (SC-002)
- [ ] T017 [US2] Manual/CI validation: run on a Linux+NVIDIA box and confirm CUDA is selected across all stages (SC-002)

**Checkpoint**: User Stories 1 AND 2 both work — this is the MVP bar for the rescope.

---

## Phase 5: User Story 3 - Concurrency without falling over (Priority: P2)

- [ ] T018 [US3] Implement the shared-model-pool refactor already prescribed in existing project docs — replace per-session model instantiation in `backend/main.py` with a pooled/shared model manager
- [ ] T019 [US3] Re-run concurrency load test (existing methodology per `documentation/PERFORMANCE_TEST_RESULTS.md`); document new ceiling vs. the current documented 12-user ceiling (SC-005)

**Checkpoint**: Concurrency ceiling measured and documented, whether or not it improved — the point is having a number, not guessing.

---

## Phase 6: User Story 4 - Speculative decoding for perceived speed (Priority: P3, thesis research contribution)

- [x] T020 [US4] Implement SSBD (arXiv 2509.21740) around `backend/mt/ctranslate2_mt.py` — done as `backend/mt/ssbd_ctranslate2_mt.py` (`SSBDCTranslate2MT`), using CTranslate2's `score_batch` (single parallel teacher-forced forward pass) for verification and `translate_batch(target_prefix=...)` to resume from the first divergence. Adaptation from the paper's literal formula documented in the module docstring: CTranslate2's public API doesn't expose per-step output distributions during a teacher-forced pass, so a log-prob-threshold divergence check is used in place of the paper's argmax-of-biased-mixture check. A real correctness bug was found and fixed during implementation: "fully accepted" must require the draft to have actually ended on EOS (`return_end_token=True`), not just that every token individually scored above threshold — otherwise short, locally-plausible-but-incomplete drafts get returned verbatim forever as the source keeps growing. Fixed; see commit.
- [x] T021 [US4] A/B benchmark: SSBD-augmented path vs. current path — done as `benchmark_ssbd_mt.py`, real wall-clock on this machine (M1 Pro, CPU, int8), `Helsinki-NLP/opus-mt-en-sk`, three runs (beam=4/short, beam=4/long, beam=1/long). **Result: negative, not positive.** Speedup 0.77x (beam=4, short utterances), 0.94x (beam=4, longer utterances), 0.93x (beam=1/greedy, longer utterances) — SSBD is 6-23% *slower* than the baseline in every configuration tested, not faster. Root cause, confirmed by inspecting per-increment draft reuse: on the long-utterance scenario the model reused **0 of 9-25** previous draft tokens on 4 of 5 increments, regardless of beam size (ruling out beam-search re-ranking as the cause) — `Helsinki-NLP/opus-mt-en-sk` restructures the whole sentence (word order, clause placement) as more source context arrives rather than monotonically extending its prior output, so the "previous translation is a reusable prefix" assumption SSBD depends on does not hold for this model. Translation quality was not separately evaluated since there is no latency win to trade it against.
- [x] T022 [US4] Write-up: `documentation/ssbd_speculative_decoding_findings_2026-08.md` — full findings, explicit proven/assumed/unknown split, real numbers, hypothesis for why this is a general-purpose-sentence-NMT-model property rather than a `Helsinki-NLP/opus-mt-en-sk`-specific quirk (untested: whether a model actually trained for incremental/streaming re-translation, like the paper's Tower+ 2B, would show prefix-stable behavior where this one doesn't — flagged as future work, not tested here). **Recommendation: do not adopt SSBD in the shipped pipeline as implemented; keep as a documented negative-result thesis contribution** — this is legitimate, citable research content (a technique's assumption failing to hold for a specific model class is a real finding), just not a shipped optimization.

**Checkpoint**: All four user stories independently functional and documented.

**2026-08-19 follow-up finding, closes this line of work further than T020-T022 alone did**: read `backend/main.py` directly rather than assuming the pipeline has an incremental-retranslation workload at all. It doesn't. The primary VAD path calls `main_mt_model.translate()` exactly once per finalized utterance (after `SILENCE_TIMEOUT`, 0.3s). The one code path that would re-translate a growing sentence is explicitly commented out at the call site: `# DISABLED: Streaming chunk processing causes Whisper hallucinations on short non-speech sounds`. So SSBD's negative result isn't just "wrong technique for this model" — this pipeline has no re-translation-of-growing-text scenario for any speculative-decoding technique to attach to, as currently built. **Do not pursue MT-stage speculative/incremental decoding further, by any technique** — the workload it would optimize doesn't exist here.

The more relevant, evidence-backed lead found instead: **`ufal/whisper_streaming`** (existing, maintained tool) implements **LocalAgreement** — emits only the longest common prefix agreed across successive incremental STT updates, which is specifically designed to prevent the hallucination-on-short-chunks problem that got this project's own streaming path disabled in the first place. Re-enabling incremental STT via this existing tool would give live partial captions while the user is still speaking — a bigger real-time UX win than MT-stage optimization would have been, and a maintained tool to adopt, not a paper to reimplement. See new Phase 8 below.

---

## Phase 7: User Story 5 - Fast personal-voice TTS tier (Priority: P2)

- [ ] T023 [US5] Check `requirements.txt` for the TTS package pin: replace unmaintained `TTS` (coqui-ai, dead since 2024) with the maintained `coqui-tts` (idiap/coqui-ai-TTS fork on PyPI) if not already
- [ ] T024 [US5] Confirm/document exact `speaker_cache` conditioning-latent caching behavior already in `backend/tts/xtts.py` (or equivalent) matches `documentation/coqui_tts_sections.md` — this is already implemented, do not reimplement, this task is verification only
- [x] T025 [US5] Set up a Piper single-speaker fine-tuning pipeline. DONE: `scripts/finetune_personal_voice.py`, using OHF-Voice/piper1-gpl (rhasspy/piper is archived, development moved). Hands-on verified on this machine: CPU beats MPS for this workload (MPS's constant-padding ops fall back to a slow path - 0.04-0.07 it/s vs 0.25-0.32 it/s on CPU). Toolchain needs its own Python 3.11 venv, separate from the main project (pytorch-lightning/scikit-build/cmake/ninja have no business on the serving backend) - see script docstring for full setup + two real bugs hit and fixed (packaged wheel ships monotonic_align's .pyx but not the compiled .so; PyTorch 2.9+'s new default ONNX exporter fails on this model's data-dependent control flow, must force `dynamo=False`).
- [ ] T026 [US5] **REVISED 2026-08-19**: the user has capped real recording at ~90s (~48s existing clips + up to ~60s more), rejecting the 10-30 min floor. Research confirms that floor is real for Piper/VITS specifically (rhasspy/piper1-gpl's own guidance), not overcautious — so T026 as originally scoped (record 10-30 min for a quality Piper fine-tune) stays a possible *upgrade path* if the user ever records more, but is no longer the default plan. See T031-T033 below for the ~90s-compatible path.
- [x] T027 [US5] Pipeline mechanically proven end-to-end on real (if tiny/toy-scale) data: 3 existing EN clips (37s total) -> trained 3 steps on CPU -> exported to ONNX (forcing legacy exporter) -> loaded via real `PiperVoice` API -> synthesized -> measured RTF 0.0295 (5-run average). This is NOT a quality fine-tune (3 steps, 37s of data) - it proves the mechanism works and the speed claim holds, not that voice similarity is good. A real run needs T026's data first.
- [~] T028 [US5] PARTIALLY done, 2026-08-21: the *reachability* half is done — `en_US-personal-medium.onnx` (the already-trained personal voice) is now selectable via `TTS_ENGINES["piper_personal"]` in `backend/tts/base.py` (`tts_model_choice="piper_personal"` from the client), no `main.py` changes needed thanks to the registry refactor. Verified live: real synthesis, 7.01s of audio from a 116-character sentence, latency 0.3128s → RTF 0.0446 (consistent with the 0.0562 documented in `documentation/personal_voice_bootstrap_2026-08-19.md`). EN-only, same limitation as before. The full four-tier *automatic selection cascade* (fine-tuned personal → GPT-SoVITS → XTTS zero-shot → generic Piper) described below is explicitly NOT done — this only adds one more explicit, manually-selectable tier; it doesn't implement the fallback cascade logic itself. **Now a four-tier design per T031-T033**: fine-tuned personal Piper (fastest, RTF 0.0295, needs 10-30 min — upgrade path) → GPT-SoVITS personal voice (RTF ~0.526 on Apple Silicon per community-reported number, needs only ~1 min — default path) → XTTS zero-shot (RTF 1.72, any language, no per-user setup — fallback) → generic Piper (no clone at all — last resort). Deliberately left for after T031/T032 land.
- [~] T029 [US5] Latency delta MEASURED: fine-tuned Piper RTF 0.0295 vs XTTS RTF 1.72 (`documentation/COQUI_TTS_PERFORMANCE_REPORT.md`) - ~58x. Voice-similarity comparison NOT done (toy 3-step model isn't representative) - re-run once T026/a real training pass exists. GPT-SoVITS's RTF 0.526 (T031) still needs to be independently re-measured on this exact M1 Pro, not just trusted from the M4 community report it's currently sourced from.
- [x] T030 [US5] Documented in `scripts/finetune_personal_voice.py` module docstring (setup, verified numbers, open questions) - trade-off already stated in `spec.md` FR-008/User Story 5 from the prior session's commit.
- [ ] T031 [US5] **NEW 2026-08-19**: Set up GPT-SoVITS (RVC-Boss/GPT-SoVITS) as the default personal-voice engine for the ~90s-of-audio case. README's own stated design goal: "1 min voice data can also be used to train a good TTS model." Community-reported RTF 0.526 on Apple Silicon M4 CPU (GitHub issue #2579) — same chip family as this M1 Pro but not the same chip, re-measure directly on this machine before trusting the number (per T029). Explicitly macOS-supported per its own README, with a documented caveat: GPU-trained models on Mac are lower quality, so Mac training currently defaults to CPU — consistent with the Piper/MPS finding from T025.
- [ ] T032 [US5] Re-run the existing recordings (`speaker_voices/*.wav`, ~48s) plus the new ~60s reading (provided this session) through GPT-SoVITS's few-shot pipeline; measure real voice-similarity and RTF on this machine — this is the first real (non-toy) quality data point for this project's personal-voice tier.
- [ ] T033 [US5] Investigated and explicitly rejected as the default: synthetic-data bootstrap (short recording → zero-shot cloner generates a large synthetic training corpus → fine-tune Piper on that). Real technique (e.g. ZeSTA, arXiv 2603.04219) but requires "domain-conditioned training with real-data oversampling" to avoid measurable speaker-similarity degradation from synthetic-only data — a research project of its own, not the lazy/goal-driven answer given GPT-SoVITS already exists and fits the constraint directly. Not pursuing unless GPT-SoVITS's real quality (T032) turns out to be insufficient.

**Checkpoint**: All five user stories independently functional and documented.

---

## Phase 8: User Story 6 - Incremental STT via LocalAgreement (Priority: P2, net-new 2026-08-19)

- [ ] T034 [US6] Evaluate `ufal/whisper_streaming`'s LocalAgreement policy as a replacement for the currently-disabled streaming chunk path in `backend/main.py` (the `# DISABLED: Streaming chunk processing causes Whisper hallucinations` block) — LocalAgreement is specifically designed to prevent exactly this failure mode by only emitting the longest common prefix agreed across successive updates, rather than committing to unstable early guesses.
- [ ] T035 [US6] If T034 checks out, wire it in behind the existing `is_final=False` code path, re-enabling live partial transcription/translation captions while the user is still speaking, without reintroducing the original hallucination bug — test specifically against the short non-speech sounds that caused the original disable.
- [ ] T036 [US6] Document real before/after: does this actually improve perceived latency/UX, and does the hallucination problem stay fixed under the same conditions that caused the original disable? Evidence-first — don't re-enable and assume it's fine.

**Checkpoint**: Live partial captions working without reintroducing the original bug, or a documented reason why not.

---

## Dependencies & Execution Order

- Phase 1 (Setup) → Phase 2 (Foundational, blocks everything) → Phases 3 & 4 (P1, can run in parallel once Phase 2 lands) → Phase 5 (P2) → Phase 6 (P3, thesis research, can start any time after Phase 2 since it only touches the MT stage) → Phase 7 (P2, TTS-stage only, can run in parallel with Phase 5/6 once Phase 3's baseline cloning path exists)
- Per Constitution Principle II: do not start Phase 6 by re-evaluating S2S/Hibiki again — that question is already closed, cited in `spec.md` Edge Cases.
- Per Constitution Principle II: do not start Phase 7 by re-litigating whether caching fixes the cloning latency — that's already measured and closed (1.9% improvement, see `spec.md` User Story 5). The open question Phase 7 actually resolves is the fine-tuning implementation, not the diagnosis.

## Notes

- Commit after each task or logical group; this feature branch's commits are local until the user explicitly decides to push (git push is a shared/visible action, out of scope for automated execution per the constitution's Operating Mode section).
- Tests exist for this project already — Phase 3/4 tasks are explicitly regression checks against the existing suite, not new test scaffolding, per Constitution Principle VI (don't build what already exists).
