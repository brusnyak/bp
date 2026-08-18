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

- [ ] T003 [US2] Create `backend/hardware.py`: one shared device-detection utility returning the correct backend per stage-family (`stt_mt_device()` → cuda/rocm/cpu; `tts_baseline_device()` → cuda/rocm/directml/coreml/cpu; `tts_cloning_device()` → cuda/mps/rocm/cpu), per the chains in `plan.md`
- [ ] T004 [US2] Wire `backend/stt/` (faster-whisper) to call `hardware.stt_mt_device()` instead of any hardcoded device string
- [ ] T005 [US2] Wire `backend/mt/ctranslate2_mt.py` to call `hardware.stt_mt_device()` instead of any hardcoded device string
- [ ] T006 [US2] Wire `backend/tts/piper_tts.py` (ONNX Runtime) to call `hardware.tts_baseline_device()`, enabling the DirectML execution provider on Windows and CoreML on macOS
- [ ] T007 [US2] Wire `backend/tts/f5_tts.py` / XTTS v2 (PyTorch) to call `hardware.tts_cloning_device()`
- [ ] T008 [US2] Remove the `# prioritize MPS support` hardcoded default and the commented-out CPU-only index URL from `requirements.txt`; document the actual runtime-selected behavior instead
- [ ] T009 [US2] Gate `mlx-whisper` (if/when added) behind an explicit `platform.system() == "Darwin"` + Apple Silicon check inside `hardware.py` — never as the default STT path (Constitution Principle I, VI)
- [ ] T010 [P] [US2] Unit test `backend/hardware.py`'s selection logic against mocked `cuda`/`rocm`/`mps`/`directml`/no-GPU environments (does not require actual GPU hardware to run)

**Checkpoint**: Foundation ready — every stage now selects hardware per-OS/per-vendor at runtime, not at install time.

---

## Phase 3: User Story 1 - Speaker gets voice-cloned translated audio out (Priority: P1) 🎯 MVP

**Goal**: Confirm the existing, already-built pipeline (STT→MT→TTS, voice cloning via XTTS v2/F5-TTS, generic fallback via Piper) still works correctly once Phase 2's hardware-selection changes land — this is a regression check, not new build, per Constitution Principle VI (don't re-architect what already works).

- [ ] T011 [US1] Run existing `test_full_pipeline.py` and `benchmark_full_pipeline.py` against the Phase 2 changes on the reference M1 Pro; confirm no latency regression vs. `documentation/PERFORMANCE_TEST_RESULTS.md` baseline
- [ ] T012 [US1] Confirm voice-clone fallback to generic Piper voice still triggers correctly when no speaker profile exists (existing behavior — regression test only)

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

---

## Dependencies & Execution Order

- Phase 1 (Setup) → Phase 2 (Foundational, blocks everything) → Phases 3 & 4 (P1, can run in parallel once Phase 2 lands) → Phase 5 (P2) → Phase 6 (P3, thesis research, can start any time after Phase 2 since it only touches the MT stage)
- Per Constitution Principle II: do not start Phase 6 by re-evaluating S2S/Hibiki again — that question is already closed, cited in `spec.md` Edge Cases.

## Notes

- Commit after each task or logical group; this feature branch's commits are local until the user explicitly decides to push (git push is a shared/visible action, out of scope for automated execution per the constitution's Operating Mode section).
- Tests exist for this project already — Phase 3/4 tasks are explicitly regression checks against the existing suite, not new test scaffolding, per Constitution Principle VI (don't build what already exists).
