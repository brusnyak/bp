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

- [ ] T020 [US4] Implement SSBD (arXiv 2509.21740) around `backend/mt/ctranslate2_mt.py`: reuse prior streaming-translation output as a speculative draft, verify in one forward pass, resume autoregressive decoding only from first divergence
- [ ] T021 [US4] A/B benchmark: SSBD-augmented path vs. current path, on the same input stream — measure latency delta and translation-quality delta independently (do not conflate the two)
- [ ] T022 [US4] Write up SSBD results in `documentation/` as thesis research content, explicit about what was measured vs. assumed

**Checkpoint**: All four user stories independently functional and documented.

---

## Dependencies & Execution Order

- Phase 1 (Setup) → Phase 2 (Foundational, blocks everything) → Phases 3 & 4 (P1, can run in parallel once Phase 2 lands) → Phase 5 (P2) → Phase 6 (P3, thesis research, can start any time after Phase 2 since it only touches the MT stage)
- Per Constitution Principle II: do not start Phase 6 by re-evaluating S2S/Hibiki again — that question is already closed, cited in `spec.md` Edge Cases.

## Notes

- Commit after each task or logical group; this feature branch's commits are local until the user explicitly decides to push (git push is a shared/visible action, out of scope for automated execution per the constitution's Operating Mode section).
- Tests exist for this project already — Phase 3/4 tasks are explicitly regression checks against the existing suite, not new test scaffolding, per Constitution Principle VI (don't build what already exists).
