# BP (Real-Time Speech Translation) Constitution

## Core Principles

### I. Cross-Platform, No Exceptions
Every required-path dependency MUST run on Windows, macOS, and Linux without per-OS hand-tuning. A platform-exclusive library (e.g. MLX, which is Apple-Silicon-only) MAY exist as an optional, runtime-detected fast path, but MUST NOT be the default for any pipeline stage. Hardware acceleration is selected per-stage at runtime (`cuda → rocm → directml/mps/coreml → cpu`, chain differs by stage — see `documentation/` for the current per-stage chain), never hardcoded to one vendor.

### II. Evidence-First, No Re-Litigating Settled Decisions
A claim about performance, latency, memory, or capability MUST be backed by a measurement recorded in the project's own docs (`documentation/PERFORMANCE_TEST_RESULTS.md`, `documentation/s2s_translation_research_2026-08.md`, etc.) before it drives a decision. Once a decision has evidence behind it (e.g. "S2S models are ruled out for this project"), do not reopen it without new evidence — extend the record instead of repeating the investigation.

### III. Slovak Is Non-Negotiable
English↔Slovak is a hard requirement, not a stretch goal. Any architecture change, model swap, or dependency upgrade MUST preserve working Slovak support before it ships. If a candidate technology doesn't support Slovak, it is out of scope for the core pipeline, however good it is for other languages.

### IV. Real-Time Is the Product
The system's value proposition is near-real-time (or faster) translation on commodity hardware, no dedicated GPU assumed as the baseline. Any change MUST be evaluated against end-to-end latency and memory footprint on the reference machine (M1 Pro, 16GB, the current dev/test box) before being accepted, not just on isolated component benchmarks.

### V. Plug-and-Play, Not Manual Setup
The user-facing bar is: install once, select the virtual mic once in your meeting app, and it works — no ongoing manual reconfiguration. OS-specific complexity (virtual audio routing: BlackHole/mac, VB-Cable/Windows, PulseAudio-PipeWire/Linux) MUST be hidden behind an automated per-OS setup path, never documented as a manual README step the end user has to perform themselves.

### VI. Simplicity Over Novelty (YAGNI)
Prefer adapting and correctly configuring existing, working open-source components (faster-whisper, CTranslate2/Opus-MT, Piper, XTTS v2) over building or training new models from scratch. A custom-trained model is only in scope if a concrete evidence gap forces it — the default assumption is that off-the-shelf, correctly wired components are enough. Novel research contributions (e.g. speculative decoding for translation) should wrap around existing components, not replace them wholesale.

## Operating Mode

Decide and execute well-evidenced, reversible steps directly rather than re-litigating settled decisions in conversation — this project moved to Spec Kit specifically so scope lives in versioned files, not chat history. Still stop and ask before genuinely destructive/irreversible actions or anything with shared/social visibility (pushing to the remote, submitting the thesis).

## Governance

This constitution supersedes ad hoc scope discussions for this project. Amendments require updating this file with a reason and a version bump — not a new conversation that quietly overrides it. Feature specs, plans, and tasks under `specs/` MUST be checked against these principles before implementation; violations require explicit justification in the relevant plan's Complexity Tracking section.

**Version**: 1.0.0 | **Ratified**: 2026-08-19 | **Last Amended**: 2026-08-19
