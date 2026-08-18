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

### User Story 5 - Cloned voice runs at default-voice speed, not as a slow overlay (Priority: P2)

The user records their own voice once; thereafter, for any language they've provided reference audio in, their cloned voice behaves as a fast default TTS voice (same speed class as Piper's generic voice, not the current 2.5–3.5s zero-shot cloning path).

**Why this priority**: Directly requested (2026-08-19) after the user correctly identified the current cloning latency as a possible architecture flaw rather than an unremovable cost — confirmed true, see findings below. P2, not P1: it's a quality-of-experience fix on top of an already-functioning P1 cloning path (US1's fallback-to-generic-voice acceptance scenario still holds), not a blocker for the MVP.

**Research findings (evidence-first, per Constitution Principle II)**:

- **Proven, from this project's own data** (`documentation/COQUI_TTS_PERFORMANCE_REPORT.md` §2.3): speaker-embedding/conditioning-latent caching is *already implemented* (`speaker_cache` dict around `get_conditioning_latents()`, see `documentation/coqui_tts_sections.md` lines ~304-308). Measured effect: 8.31s → 8.15s raw synthesis, a 1.9% improvement. This proves the reference-audio re-encoding step is **not** the source of the 2.5–3.5s perceived latency — that cost was already engineered out and barely moved the number.
- **Proven, corroborated externally**: the actual cost is XTTS v2's GPT-based autoregressive decoder generating audio tokens step-by-step — architectural, not a caching miss. Independent community consensus confirms cloned-voice XTTS v2 stays slow regardless of caching, even though the same model's non-cloned/base streaming path can hit <200ms. Piper (RTF ~0.05) vs XTTS (RTF ~1.72) is a ~34x gap that caching cannot close.
- **Verdict**: this is a genuine system-level architecture ceiling of zero-shot autoregressive voice cloning, not a fixable inefficiency in this project's implementation. The user's instinct to ask "is this an architecture flaw" was correct.
- **Real fix found, proven feasible**: Piper (VITS-based) supports fine-tuning a dedicated single-speaker model from an existing checkpoint on as little as ~5 minutes of target-voice audio (up to ~1hr for best quality) — standard, documented, community-practiced, realistic on modest hardware since it's fine-tuning, not training from scratch. A fine-tuned Piper voice runs at Piper's native speed (no zero-shot conditioning at inference time at all).
- **Unresolved trade-off, flagged not silently decided**: a Piper voice fine-tuned on English recordings does not inherit XTTS's cross-lingual transfer — VITS-style models are bound to the phoneme/language coverage of their training data. Getting a fast, fine-tuned "your voice, in Slovak" requires recording reference audio *in Slovak*, not just English. This is a real product decision, not an engineering detail: it trades XTTS's "record once in any language, get any-language output" promise for speed, and only for languages actually recorded.
- **Recommended resolution**: extend the existing two-tier TTS design (Piper generic / XTTS cloned) into three tiers: (1) Piper generic — fastest, no cloning; (2) Piper personally-fine-tuned — fast, cloned, only for languages the user has recorded reference audio in; (3) XTTS zero-shot — slow (2.5–3.5s), cloned, any language, retained as the fallback for languages without a fine-tuned model. This keeps US1's fallback behavior intact while giving the user's stated goal (fast default cloned voice) for EN and SK specifically, once they record ~5-60 min of reference audio in each.
- **Also flagged, out of this story's scope but worth a task**: `coqui-ai/TTS` (the original PyPI package) has been unmaintained since Coqui.ai shut down in 2024; the community-maintained fork is `idiap/coqui-ai-TTS`, installable as `coqui-tts`. Check which package `requirements.txt` currently pins.

**Independent Test**: Fine-tune a Piper model on ~10 minutes of the user's own EN and SK reference audio; A/B its latency and perceived voice-similarity against the current XTTS zero-shot path for the same sentences.

**Acceptance Scenarios**:

1. **Given** the user has recorded reference audio in a supported language and a fine-tuned Piper model exists for it, **When** they speak in that language, **Then** output uses the fast fine-tuned voice, not the XTTS zero-shot path.
2. **Given** the user has not recorded/fine-tuned for a given target language, **When** translation targets that language, **Then** the system falls back to XTTS zero-shot cloning (current behavior, unchanged) rather than failing or silently using a generic voice.

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
- **FR-008**: System MUST support a fine-tuned, personal-voice Piper model as a fast-path TTS tier for languages the user has provided reference audio in, falling back to XTTS zero-shot cloning for any language without one (see User Story 5).

### Key Entities

- **Speaker Voice Profile**: A recorded/uploaded voice sample plus metadata (language, display name) used for voice-cloned TTS output. Already implemented (`speaker_voices.json`).
- **Translation Session**: One active WebSocket connection's pipeline state — source/target language, selected TTS model, per-session model instances (candidate for the shared-pool refactor in User Story 3).
- **Hardware Backend Selection**: Per-stage runtime decision (STT/MT backend, TTS backend) based on detected OS + GPU vendor, replacing the current hardcoded MPS-first default.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: End-to-end latency (mic input to translated audio output) stays within the near-real-time budget already targeted in the project's own docs on the reference M1 Pro 16GB, for the EN↔SK pair: <1s standard TTS path; **<1s for the personal fine-tuned voice path (User Story 5) once a language has been fine-tuned**; 2.5–3.5s remains the accepted ceiling only for XTTS zero-shot cloning on languages without a fine-tuned model.
- **SC-002**: The same codebase, unmodified, runs on a Windows machine and a Linux machine and produces translated audio without manual per-OS code changes — only per-OS setup-script execution.
- **SC-003**: `requirements.txt`/config no longer hardcodes "prioritize MPS" or any single-vendor GPU path as the default.
- **SC-004**: Slovak translation quality/latency is not regressed relative to the current (pre-rescope) pipeline, measured against `documentation/PERFORMANCE_TEST_RESULTS.md` baselines.
- **SC-005**: Concurrent-session ceiling is measured and documented after the shared-model-pool refactor, with a number to compare against the current documented ceiling of 12 users.

## Assumptions

- Target users have a normal consumer machine (no dedicated GPU assumed as the baseline case); GPU acceleration is a bonus path, not a requirement.
- The existing Piper + XTTS v2 TTS split stays architecturally as-is (per Constitution Principle VI) — this rescope is about hardware-backend selection and platform support, not swapping the TTS engine. User Story 5 adds a third tier (personal fine-tuned Piper voices) on top of this split rather than replacing either existing engine — still no new TTS engine introduced.
- User Story 5's fine-tuned-voice fast path is per-language: it requires the user to record reference audio in each target language they want it for (confirmed EN and SK at minimum). Languages without a fine-tuned model keep using XTTS zero-shot at current latency — this is a stated, accepted trade-off, not a gap to silently paper over.
- S2S (Hibiki-class) models are out of scope for this feature and are not revisited without new evidence (Constitution Principle II, `documentation/s2s_translation_research_2026-08.md`).
- SSBD (User Story 4) is a research/thesis-value addition, not a blocking requirement for the MVP defined by User Stories 1–2.
