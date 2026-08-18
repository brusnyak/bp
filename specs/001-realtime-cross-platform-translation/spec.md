# Feature Specification: Cross-Platform Real-Time Speech Translation Pipeline (v2 rescope)

**Feature Branch**: `001-realtime-cross-platform-translation`

**Created**: 2026-08-19

**Status**: Draft

**Input**: Rescope of the existing STT→MT→TTS pipeline to be genuinely cross-platform (Windows/macOS/Linux), keep Slovak as a hard requirement, drop the S2S (speech-to-speech) direction after hands-on evaluation, and add speculative decoding as the thesis's novel contribution.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Speak in a meeting, get translated speech out in your own voice (Priority: P1) 🎯 MVP

A conference speaker enables the tool, speaks in English (or their configured source language) into their normal microphone during a call/meeting/presentation. Other participants hear the translation, synthesized in the speaker's own cloned voice, through the meeting app's audio — the speaker did not have to change any meeting-app settings beyond selecting the virtual mic once.

**Why this priority**: This is the entire product. Without it nothing else matters.

**Independent Test**: Run the pipeline end-to-end on a single machine with a live mic and virtual audio output selected in a real meeting app (Zoom/Meet/Teams); verify translated, voice-cloned audio arrives on the other end within the real-time latency budget (Success Criteria below).

**Acceptance Scenarios**:

1. **Given** the app is initialized and a voice sample has been recorded, **When** the user speaks a sentence in English, **Then** the other participants hear a Slovak translation in the user's own voice within the target latency.
2. **Given** the user has not recorded a voice sample, **When** they speak, **Then** the system falls back to a generic Piper voice rather than failing.

---

### User Story 2 - Same install works identically on Windows, macOS, and Linux (Priority: P1) 🎯 MVP

A user on any of the three major OSes installs the tool and it runs at usable speed without them needing to know their own GPU vendor, install CUDA/ROCm manually, or edit config files.

**Why this priority**: Co-equal P1 with User Story 1 — the recurring problem this rescope exists to fix is the tool being fast on the author's Mac and slow/broken elsewhere. A translation tool that only works well for its author isn't a deliverable product.

**Independent Test**: Run the same install steps on a Windows machine with an AMD GPU, a Linux machine with an NVIDIA GPU, and the reference M1 Pro Mac; verify each selects a working hardware backend automatically and produces translated audio without manual intervention.

**Acceptance Scenarios**:

1. **Given** a Windows machine with a DirectX-12-capable AMD GPU, **When** the app starts, **Then** the TTS stage uses DirectML acceleration automatically, not CPU-only.
2. **Given** a machine with no discrete GPU at all, **When** the app starts, **Then** every stage falls back to CPU cleanly, with an explicit (not silent) indication in the UI that it's running in a slower/CPU mode.
3. **Given** the current `requirements.txt`'s hardcoded "prioritize MPS support" default, **When** this feature ships, **Then** that default no longer exists — hardware selection is runtime-detected per stage, not hardcoded to one vendor.

---

### User Story 3 - Handle concurrent speakers without falling over (Priority: P2)

Multiple simultaneous conference sessions (or multiple speakers) can run against the backend without one session's load degrading another's, up to the concurrency the reference hardware can sustain.

**Why this priority**: Already a documented, measured bottleneck (`documentation/PERFORMANCE_TEST_RESULTS.md` shows the pipeline saturating at 12 concurrent users on the reference M1 Pro 16GB) — but it's not blocking for a single-user thesis demo, so P2 not P1.

**Independent Test**: Load-test with N simulated concurrent sessions on the reference machine; verify graceful degradation (documented ceiling, no crash) rather than silent failure past the ceiling.

**Acceptance Scenarios**:

1. **Given** the shared-model-pool refactor already prescribed in the project's own docs, **When** it's implemented, **Then** per-session model instances no longer duplicate the full model set in memory.

---

### User Story 4 - Translation feels faster via speculative decoding (Priority: P3, thesis research contribution)

The system reduces perceived/actual translation latency using a self-speculative biased decoding (SSBD) approach layered on the existing Opus-MT calls, without requiring new training data or a GPU cluster.

**Why this priority**: This is the thesis's novel research angle, not required for the product to function — the pipeline works without it. Scoped as P3 explicitly so it doesn't block the MVP.

**Independent Test**: A/B the existing Opus-MT translation path against the SSBD-augmented path on the same input stream; measure latency delta and translation-quality delta independently.

**Acceptance Scenarios**:

1. **Given** a streaming input requiring re-translation as more audio arrives, **When** SSBD is enabled, **Then** it reuses the prior output as a draft and verifies in one forward pass rather than fully re-decoding from scratch.

---

### Edge Cases

- What happens when no GPU of any kind is present and the machine is genuinely low-spec? → Must degrade to CPU-only with a visible warning, not silently run at unusable latency with no explanation.
- What happens when the user speaks a language/pair the system doesn't support? → Must surface a clear "unsupported language pair" state, not silently produce garbage output.
- What happens on Windows with an AMD GPU for the TTS voice-cloning stage specifically (PyTorch has no ROCm-on-Windows path)? → Known, accepted limitation: falls back to CPU for that stage only; must be documented, not hidden.
- What happens when Hibiki-class S2S temptation resurfaces later (a future maintainer proposes it again)? → Point to `documentation/s2s_translation_research_2026-08.md` (RTF 1.37, ~4.1GB RAM/stream, no Slovak support) — this is a settled decision per Constitution Principle II, not an open question, unless genuinely new evidence exists.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST perform STT → MT → TTS on live microphone audio and output translated, voice-cloned speech to a virtual audio device.
- **FR-002**: System MUST support English↔Slovak as a first-class, always-supported language pair.
- **FR-003**: System MUST select hardware acceleration per pipeline stage at runtime (not hardcoded), using the chain: STT/MT (CTranslate2) = `cuda → rocm(Linux+AMD) → cpu`; TTS baseline (Piper/ONNX) = `cuda → rocm(Linux) → directml(Windows) → coreml(mac) → cpu`; TTS cloning (PyTorch) = `cuda → mps(mac) → rocm(Linux+AMD) → cpu`.
- **FR-004**: System MUST fall back to a generic (non-cloned) voice when no speaker voice sample is available, rather than failing.
- **FR-005**: System MUST automate virtual-audio-device setup per OS (BlackHole/VB-Cable/PulseAudio-PipeWire) so the end user does not manually install or configure audio routing beyond one device selection in their meeting app.
- **FR-006**: System MUST NOT default to any Apple-Silicon-only dependency (e.g. `mlx-whisper`) as the primary path on any stage — such dependencies MAY exist as an opt-in, auto-detected fast path only.
- **FR-007**: System MUST expose real-time latency metrics (existing latency breakdown UI) so performance claims remain measurable, not assumed.

### Key Entities

- **Speaker Voice Profile**: A recorded/uploaded voice sample plus metadata (language, display name) used for voice-cloned TTS output. Already implemented (`speaker_voices.json`).
- **Translation Session**: One active WebSocket connection's pipeline state — source/target language, selected TTS model, per-session model instances (candidate for the shared-pool refactor in User Story 3).
- **Hardware Backend Selection**: Per-stage runtime decision (STT/MT backend, TTS backend) based on detected OS + GPU vendor, replacing the current hardcoded MPS-first default.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: End-to-end latency (mic input to translated audio output) stays within the near-real-time budget already targeted in the project's own docs (<1s standard TTS path, 2.5–3.5s voice-cloning path) on the reference M1 Pro 16GB, for the EN↔SK pair.
- **SC-002**: The same codebase, unmodified, runs on a Windows machine and a Linux machine and produces translated audio without manual per-OS code changes — only per-OS setup-script execution.
- **SC-003**: `requirements.txt`/config no longer hardcodes "prioritize MPS" or any single-vendor GPU path as the default.
- **SC-004**: Slovak translation quality/latency is not regressed relative to the current (pre-rescope) pipeline, measured against `documentation/PERFORMANCE_TEST_RESULTS.md` baselines.
- **SC-005**: Concurrent-session ceiling is measured and documented after the shared-model-pool refactor, with a number to compare against the current documented ceiling of 12 users.

## Assumptions

- Target users have a normal consumer machine (no dedicated GPU assumed as the baseline case); GPU acceleration is a bonus path, not a requirement.
- The existing Piper + XTTS v2 TTS split stays architecturally as-is (per Constitution Principle VI) — this rescope is about hardware-backend selection and platform support, not swapping the TTS engine.
- S2S (Hibiki-class) models are out of scope for this feature and are not revisited without new evidence (Constitution Principle II, `documentation/s2s_translation_research_2026-08.md`).
- SSBD (User Story 4) is a research/thesis-value addition, not a blocking requirement for the MVP defined by User Stories 1–2.
