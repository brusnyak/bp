# Implementation Plan: Cross-Platform Real-Time Speech Translation Pipeline (v2 rescope)

**Branch**: `001-realtime-cross-platform-translation` | **Date**: 2026-08-19 | **Spec**: `specs/001-realtime-cross-platform-translation/spec.md`

**Input**: Feature specification from `specs/001-realtime-cross-platform-translation/spec.md`

## Summary

Fix the pipeline's hardcoded, Mac-only hardware-acceleration default and formalize a per-stage, runtime-detected backend chain so the existing STT→MT→TTS architecture (validated — S2S is ruled out, see Constitution Principle II) actually delivers on the "works the same everywhere" requirement. Layer SSBD-style speculative decoding on top as the thesis's research contribution once the foundation is fixed.

## Technical Context

**Language/Version**: Python 3.9+ (existing), FastAPI/WebSocket backend, plain HTML/CSS/JS frontend (existing, unchanged by this feature)

**Primary Dependencies**: `faster-whisper` (STT, CTranslate2 backend), CTranslate2 + Opus-MT/Helsinki-NLP (MT), Piper (TTS baseline, ONNX Runtime), Coqui XTTS v2 / F5-TTS (TTS voice cloning, PyTorch), `torch`/`torchaudio`

**Storage**: Existing SQLite (`sql_app.db`) for session/voice metadata — unchanged by this feature

**Testing**: Existing `pytest` suite (`test_backend_pipeline.py`, `test_full_pipeline.py`, `benchmark_latency.py`, `benchmark_full_pipeline.py`) — extend, don't replace

**Target Platform**: Windows 10+, macOS (Apple Silicon + Intel), Linux (glibc-based distros) — this is the platform matrix this feature exists to satisfy

**Project Type**: Web application (FastAPI backend + browser frontend, existing structure)

**Performance Goals**: <1s standard-TTS end-to-end latency, 2.5–3.5s voice-cloning end-to-end latency (existing targets, from `README.md`), on the reference M1 Pro 16GB, for EN↔SK

**Constraints**: No dedicated GPU assumed as baseline; must run on 16GB RAM class hardware; no platform-exclusive dependency (MLX, CUDA-only) as a required-path default

**Scale/Scope**: Single-user thesis demo is the hard requirement (User Stories 1–2); documented-not-necessarily-solved concurrency ceiling is a secondary goal (User Story 3, currently 12 concurrent users per `documentation/PERFORMANCE_TEST_RESULTS.md`)

## Constitution Check

*GATE: checked against `.specify/memory/constitution.md` v1.0.0*

- **Principle I (Cross-Platform)**: This entire feature exists to satisfy this principle — PASS by construction once implemented. Explicit violation risk: reaching for `mlx-whisper` as a default (rejected in FR-006/Constitution VI).
- **Principle II (Evidence-First)**: The S2S-rejection and per-stage backend chain claims in this plan are sourced from `documentation/s2s_translation_research_2026-08.md` and `documentation/PERFORMANCE_TEST_RESULTS.md` — PASS, cited not asserted.
- **Principle III (Slovak Non-Negotiable)**: FR-002, SC-004 — PASS, explicit requirement and success criterion.
- **Principle IV (Real-Time)**: SC-001 — PASS, explicit latency targets carried over from existing README targets, not invented.
- **Principle V (Plug-and-Play)**: FR-005 — PASS, automated per-OS virtual-audio setup is in scope.
- **Principle VI (Simplicity/YAGNI)**: PASS — this plan explicitly keeps the existing Piper/XTTS v2 split and existing pipeline architecture; it does not introduce a new TTS/STT engine or a custom-trained model.

No violations requiring Complexity Tracking justification.

## Project Structure

### Documentation (this feature)

```text
specs/001-realtime-cross-platform-translation/
├── plan.md              # This file
├── spec.md              # Feature specification
└── tasks.md             # Task breakdown (/speckit-tasks output)
```

### Source Code (repository root, existing structure — this feature modifies, does not restructure)

```text
backend/
├── main.py                    # FastAPI app, WebSocket handling — add hardware-backend selection at startup
├── stt/                       # faster-whisper wrapper — backend selection: cuda → rocm → cpu
├── mt/
│   └── ctranslate2_mt.py      # Opus-MT via CTranslate2 — backend selection: cuda → rocm → cpu; SSBD wraps here (User Story 4)
├── tts/
│   ├── piper_tts.py           # ONNX Runtime — backend selection: cuda → rocm → directml → coreml → cpu
│   └── f5_tts.py / xtts.py    # PyTorch — backend selection: cuda → mps → rocm → cpu
└── audio/                     # NEW: per-OS virtual-audio-device setup automation (BlackHole/VB-Cable/PipeWire)

scripts/
└── setup_windows.ps1          # Existing — extend rather than duplicate for the DirectML/VB-Cable path

requirements.txt               # Remove hardcoded "prioritize MPS" comment/default; make hardware backend a runtime decision, not an install-time pin
```

**Structure Decision**: Existing single-backend/single-frontend web-app layout is retained (Constitution Principle VI — no restructuring for its own sake). Changes are concentrated in `backend/stt/`, `backend/mt/`, `backend/tts/` (add a shared hardware-detection utility each stage calls) and a new `backend/audio/` module for OS-specific virtual-device automation. No frontend changes required for this feature.

## Complexity Tracking

*No Constitution violations — table intentionally empty.*
