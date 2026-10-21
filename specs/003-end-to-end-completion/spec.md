# Feature Specification: End-to-End Completion — any-language, swap-any-engine, plug-and-play

**Feature Branch**: `003-end-to-end-completion`
**Created**: 2026-10-18
**Status**: Draft
**Supersedes**: `001-realtime-cross-platform-translation`, `002-pwa-pip-demo` (archived to `specs/_archive/`; their still-open work is fused below — see §Source map)
**Input**: Owner direction 2026-10-18 — "the project has to be able to handle any language, cloning and more, run on server, semi-production-grade ready… easy CLI plug-and-play, able to swap whatever whenever for a better version… the rest for Path B and beyond should be set." Thesis-complete **and** product-grade, because the handler may want a collaboration on their own custom system.

## Context

Two earlier specs each drifted: 001 (cross-platform pipeline) left portability, per-OS
audio routing and the shared model pool open; 002 (PWA + PiP demo) was a demo sprint.
This spec is the single completion line. It turns the bachelor prototype into a system
that (a) supports any language pair through a declared, tested "language pack", (b) lets
every pipeline stage be swapped for a better engine behind one interface, (c) installs
and runs with one CLI command on any OS, (d) can run as a server for multiple users, and
(e) can be used in a real call (system audio in, virtual mic out, floating captions).
The cascade stays the thesis contribution; a unified speech-to-speech engine is optional,
pluggable, and never required (evidence: `documentation/s2s_landscape_2026-10-18.md`).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Add and verify a language pair without code (Priority: P1)

A maintainer wants `en↔pl` working. They declare a language pack (STT model, MT
direction(s), TTS voice, script hints) in one config/manifest file, run
`bp language add pl`, and the system downloads/points at the models, runs a smoke test
against a fixed clip, and reports WER/chrF/latency. If the test passes, `pl` appears as
**verified**; if it fails, it appears as **experimental** with the failing metric.

**Why this priority**: everything the supervisor's "any language" request means reduces
to this one flow. Without it, every language is bespoke code.

**Independent Test**: Add `cs` (files already exist: `bootstrap_cz/`,
`ct2_models/Helsinki-NLP--opus-mt-en-cs`) with zero code changes; `bp language test cs`
returns a pass/fail record.

**Acceptance Scenarios**:

1. **Given** a language pack file for a pair, **When** `bp language add <pair>` runs,
   **Then** models are resolved (local or downloaded), a smoke test runs, and a JSON
   record with WER/chrF/latency + `verified|experimental` is written.
2. **Given** a verified language, **When** the live UI language dropdown opens, **Then**
   verified pairs show as tested and unverified ones as "experimental", matching the
   existing UI rule from `model_landscape_2026-10.md`.
3. **Given** a pair whose MT model is missing, **When** it is selected, **Then** the
   system refuses cleanly with the missing component named (no silent fallback to a
   wrong language).

---

### User Story 2 - Swap any engine behind one interface (Priority: P1)

A maintainer swaps the STT engine (e.g. faster-whisper → Voxtral Realtime), or TTS
(Piper → OmniVoice/CosyVoice3), or MT, or VAD, by changing config. Each stage has one
interface; each engine declares its capabilities; a gate test must pass before an engine
is selectable in production.

**Why this priority**: "no reliance on any one thing" + "swap whenever for a better
version" is the core durability requirement.

**Independent Test**: Register a second TTS engine in the registry, point the config at
it, run one utterance; output audio is produced and the latency record names the engine.

**Acceptance Scenarios**:

1. **Given** the engine registry, **When** a stage's engine is changed in config,
   **Then** the pipeline runs with the new engine without code edits.
2. **Given** a candidate engine, **When** it cannot pass the stage's gate test,
   **Then** it cannot be set as the default (the gate blocks it) but can still be run
   in an explicit experimental mode.
3. **Given** any engine, **When** it is used, **Then** the session record names the
   exact engine + version so every measurement is attributable.

---

### User Story 3 - One-command install and run, any OS, CPU floor (Priority: P1)

A new machine: clone, one command, done. No GPU required for the baseline. Linux is
covered too (currently unverified). The command produces a runnable server, a sample
voice, certs, and a passing self-test.

**Why this priority**: plug-and-play is Constitution principle V and the thing a
collaborator or examiner hits first.

**Independent Test**: Fresh clone on macOS + Windows + Linux; `python scripts/setup.py`
then `bp run`; `bp selftest` passes on all three; recorded times match the sheet in
`documentation/linux_setup_and_test.md`.

**Acceptance Scenarios**:

1. **Given** a clean machine with Python 3.11, **When** the setup command runs,
   **Then** it ends with a passing self-test and no manual steps.
2. **Given** Linux, **When** audio routing is configured, **Then** a PipeWire null-sink
   is created and selected automatically (not documented as a user chore).

---

### User Story 4 - Run as a server for multiple users, semi-production (Priority: P2)

An operator runs the backend on a shared box. Auth is real (no unauthenticated
WebSocket), sessions are isolated, models are pooled (one copy per stage, not per
session), and health/metrics endpoints exist for monitoring.

**Why this priority**: the supervisor's "run on server" and any hosted demo need this.
`T018` (shared model pool) and the security audit items live here.

**Independent Test**: Two authenticated clients translate concurrently; model memory is
loaded once (measured RSS delta < ~1 model); `/health` reports stage readiness.

**Acceptance Scenarios**:

1. **Given** the server deployed, **When** an unauthenticated client opens `/ws`,
   **Then** the connection is rejected.
2. **Given** N concurrent sessions, **When** load rises, **Then** per-stage resource use
   is bounded by configuration and the measured capacity in `load_tests_2026-10.md`
   is reproduced or exceeded.

---

### User Story 5 - Use it in a real call (Path B) (Priority: P2)

A user joins a Zoom/Meet/Teams call. The system captures the other party's audio
(loopback), shows floating captions over any window, and can put the user's translated
voice into the call (virtual mic). One-time routing setup, automated per OS.

**Why this priority**: without this, the tool has no user path beyond a demo (see the
commercial review). It is the difference between "prototype" and "usable".

**Independent Test**: On the reference Mac, a recorded two-app call is translated with
captions shown over the call window and translated audio presented to a virtual mic.

**Acceptance Scenarios**:

1. **Given** BlackHole/VB-Cable/PipeWire set up once, **When** a call plays audio,
   **Then** captions appear within the measured latency target and audio can be routed
   into the call.
2. **Given** the overlay, **When** another app is focused, **Then** captions stay visible.

---

### User Story 6 - Use it as a library/SDK in someone else's system (Priority: P3)

A collaborator integrates the pipeline into their own app with no UI: import the
pipeline, configure stages, stream in, stream out. A thin REST/WS contract + Python API.

**Why this priority**: the collaboration hook; also the most realistic non-thesis value.

**Independent Test**: A ~30-line script drives one utterance end-to-end and prints text
+ writes audio, importing only the public package surface.

---

### User Story 7 - Plug a unified speech-to-speech engine (Priority: P3)

An operator selects a unified engine (local Qwen3-Omni, or a cloud adapter) instead of
the cascade, for higher quality on a GPU box, without rewriting the app.

**Why this priority**: future-proofing; keeps the project current after defense.

**Independent Test**: A unified engine implementing the stage interface produces
translated audio + text; the cascade remains the default.

---

### Edge Cases

- Missing MT direction for a pair (e.g. `sk→cs` with no model): refuse, name it, do not
  pivot through English silently unless the pack opts in.
- A language whose script differs (non-Latin): script/`initial_prompt` hints must be part
  of the pack, and the smoke test must catch diacritic/orthography failures.
- A CC-BY-NC model (Seamless/NLLB) selected for a shipped path: the licence gate blocks it.
- Offline machine with no cache: setup must say what it needs and fail clearly.

## Requirements *(mandatory)*

### Functional

- **FR-001**: A language is defined by a declarative **language pack** (source/target
  code, STT model ref, MT model ref(s), TTS voice ref, optional `initial_prompt`/script
  hints, optional license). Adding a language = adding/editing a pack, not code.
- **FR-002**: `bp language add|list|test|remove` CLI exists; `test` runs a fixed clip
  through the real pipeline and writes a record (`WER`, `chrF`, per-stage latency,
  `verified|experimental`).
- **FR-003**: Every pipeline stage (VAD, STT, MT, TTS) sits behind one interface with an
  **engine registry**; engines are selected by config key, not by import sites.
- **FR-004**: Each engine declares a **capability manifest** (languages, device needs,
  streaming support, licence). The UI/CLI reads it; nothing is advertised that the engine
  cannot do.
- **FR-005**: A **licence gate** marks engines `commercial-ok | non-commercial | unknown`;
  non-commercial engines cannot be selected in the default (shippable) profile.
- **FR-006**: A **stage gate test** per engine must pass before it can be a default;
  failure keeps it experimental-only.
- **FR-007**: The system runs **offline-first**; no cloud call is on the default path.
  A cloud adapter (Gemini Live Translate / Azure Live Interpreter) may exist as an
  opt-in engine, never the default.
- **FR-008**: **Auth** is required for all mutating/streaming endpoints; the WebSocket
  is authenticated; registration is closed by default.
- **FR-009**: Models are **pooled** — one instance per stage per process unless config
  says otherwise; sessions share, not duplicate.
- **FR-010**: **Health/metrics** endpoints report per-stage readiness, engine in use,
  and live load.
- **FR-011**: **Path B**: per-OS audio routing (loopback capture + virtual mic) is set up
  by one command per OS; a floating caption window is available.
- **FR-012**: **SDK**: a documented public surface (Python API + REST/WS contract) lets a
  foreign system drive the pipeline without the bundled UI.
- **FR-013**: Every measurement names the exact engine + version + device (attribution),
  reusing the existing session-record format.
- **FR-014**: A unified S2S engine can implement the cascade's interface and be selected
  by config (FR-003) for both offline (local model) and online (API adapter) cases.

### Key Entities

- **Language pack** — the declarative unit of "a supported language/pair".
- **Engine** — one implementation of a stage (VAD/STT/MT/TTS/unified), with manifest.
- **Capability manifest** — machine-readable engine facts (languages, device, licence).
- **Eval record** — one gate/`language test` result: metrics + engine + device + date.
- **Session** — one user's run: config, engines, per-utterance latency, transcripts.

## Success Criteria *(mandatory)*

- **SC-001**: A new Latin-script language pair is added and tested with **no code
  changes**, in under 30 minutes of operator time (excluding model download).
- **SC-002**: `python scripts/setup.py && bp selftest` passes on **macOS, Windows and
  Linux** from a clean clone, CPU-only, no manual steps.
- **SC-003**: Swapping the STT engine (e.g. to Voxtral Realtime) or the TTS engine
  (e.g. to OmniVoice) requires only a config change and passes the stage gate.
- **SC-004**: Two authenticated clients run concurrently on one process with models
  loaded once; the measured capacity in `load_tests_2026-10.md` is met or beaten.
- **SC-005**: A live Zoom/Meet call on the reference Mac shows captions over the window
  and can present translated audio to the call, after a one-time automated setup.
- **SC-006**: Every advertised language shows either a passing eval record or an explicit
  "experimental" label — no untested claim is shown as tested.
- **SC-007**: A ~30-line external script drives one utterance end-to-end.

## Non-Functional / Constraints

- **Cross-platform, CPU floor** (Constitution I): baseline must run on Windows/macOS/
  Linux without a GPU; accelerators are runtime-detected, per stage, never hardcoded.
- **Real-time** (Constitution IV): changes are judged on end-to-end latency on the
  reference machine; the current measured band is EN→SK ~0.7 s / SK→EN ~1.4 s per turn.
- **Evidence-first** (Constitution II): no performance claim without a recorded
  measurement naming engine + device.
- **Slovak** (Constitution III, to be amended): EN↔SK stays the reference pair and must
  never regress; the principle broadens from "only Slovak" to "Slovak is the reference;
  any pair meeting the support bar is supported".
- **Licence hygiene**: no CC-BY-NC model on a shippable path (see
  `documentation/s2s_landscape_2026-10-18.md` §8). Repo needs a `LICENSE`.
- **Memory**: reference box 16 GB; new engines must declare and respect VRAM/RAM needs.
- **No per-command permission sprawl; no secrets in repo or logs** (auto-security rules).

## Source map (what moved from 001/002)

| Origin | Item | Now |
|---|---|---|
| 001 T004/T005/T006 | Wire per-stage hardware backend detection (STT/MT/TTS) | FR-003, US2 (task T003 here) |
| 001 T013/T014/T015 | Per-OS audio routing (VB-Cable/BlackHole/PipeWire) | FR-011, US5 (tasks T014–T016) |
| 001 T018 | Shared model pool | FR-009, US4 (task T010) |
| 001 T019 | Concurrency ceiling | superseded by `load_tests_2026-10.md`; re-verify in T011 |
| 001 T021–T026, T031–T033 | Voice-clone tier research | kept as `model_landscape` verdicts; TTS clone upgrade in US2/T007 |
| 001 T035 | Streaming partial captions wiring | US2/T006 (captions) |
| 001 T023/T024 | coqui-tts pin / XTTS cache verify | housekeeping T018 |
| 002 T002/T004/T013 | PWA verify, offline bundle, install prompt | US3 + T017 |
| 002 T006 | VAD soundcheck | T019 |
| 002 T007/T010 | Demo script + recordings | T017/T019 |

## Out of scope

- New model training (Piper fine-tune workflow stays as-is unless US2 needs it).
- A SaaS/billing product. This spec targets a great thesis + a collaborator-usable system.
- Beating cloud accuracy on spontaneous speech; the goal is a trustworthy local system,
  not a leaderboard win.

## Assumptions

- Python 3.11 is available; `uv` may be present (used for speed, not required).
- The reference machine stays the M1 Pro; a T4-class GPU box may exist for TTS load tests.
- Opus-MT remains the default MT (a model per direction) until a licence-clean
  multilingual MT (NLLW/EuroLLM) passes its gate.

## Dependencies

- `documentation/s2s_landscape_2026-10-18.md` (reuse candidates + licence traps).
- `documentation/model_landscape_2026-10.md` (per-component verdicts).
- `documentation/load_tests_2026-10.md` (capacity numbers to meet).
- Constitution `.specify/memory/constitution.md` (needs the III amendment noted above).
