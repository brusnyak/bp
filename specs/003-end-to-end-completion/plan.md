# Implementation Plan: End-to-End Completion

**Spec**: `spec.md` · **Branch**: `003-end-to-end-completion` · **Created**: 2026-10-18
**Constitution**: `.specify/memory/constitution.md` (I–VI) — one amendment required (§Complexity Tracking)

## Summary

Turn the prototype into an any-language, swap-any-engine, one-command, server-capable
system, with Path B (real-call capture/output) as the usable-product track. The cascade
stays the default and the thesis contribution; unified S2S engines are optional adapters.
Do it in four milestones, each independently demoable and gated by the existing tests +
a new stage gate.

## Technical approach

### 1. Declarative languages (FR-001, FR-002)
- New `languages/` directory of pack files: `languages/packs/<src>-<tgt>.yaml|json`
  (plain JSON to avoid a YAML dep). Fields: codes, `stt` model ref, `mt` model ref(s),
  `tts` voice ref, optional `initial_prompt`/script, `license`.
- Resolver resolves a model ref to either a local path or a HuggingFace id (download on
  demand through the existing conversion pipeline in `scripts/convert_models.py`).
- `scripts/bp.py` gains `language add|list|test|remove`. `test` reuses the existing eval
  harness (`scripts/eval_stt.py`, `scripts/bench_stt_*`, `scripts/demo_conversation.py`)
  and writes `processed/language_eval/<pair>.json`.
- Verified/experimental status is derived from whether a passing eval record exists.

### 2. Stage interfaces + engine registry (FR-003…FR-006, FR-013)
- Define abstract bases: `VADEngine`, `STTEngine`, `MTEngine`, `TTSEngine`, plus a
  `UnifiedEngine` (FR-014). Existing classes (`faster_whisper_stt`, `ctranslate2_mt`,
  `piper_tts`, `omni_tts`, `hybrid_tts`, VAD) get wrapped, not rewritten.
- `backend/registry.py`: name → factory + capability manifest. Manifest fields:
  `stages`, `languages`, `devices` (cpu/cuda/mps/directml), `streaming` (bool),
  `license` (`commercial-ok|non-commercial|unknown`), `resource` (RAM/VRAM hint).
- Config selects engines by name (`session_config`); `backend/main.py` resolves via the
  registry. Keep the existing label==engine invariant (a past real bug).
- Licence gate: non-commercial engines are blocked in the default profile, allowed only
  with `--allow-noncommercial`.
- Stage gate: `bp engine gate <stage> <engine>` runs a fixed clip and asserts a per-stage
  threshold (STT WER, MT chrF, TTS RTF+similarity, VAD close-timing). Result →
  `processed/engine_gate/<stage>_<engine>.json`.

### 3. One-command install + server hardening (FR-007…FR-010)
- Extend `scripts/setup.py`: language-pack-driven model fetch, `bp selftest`, Linux path.
- Auth: close registration by default; require JWT on `/ws` and mutating routes; keep the
  snapshot/restore fixture in `test/backend_api_tests.py` (never wipe real voices).
- Model pool: one instance per stage per process; sessions share. Replace per-session
  instantiation in `backend/main.py` (this is old T018, now in scope).
- `/health` + `/metrics`: per-stage readiness, engine + version, live session count.

### 4. Path B + SDK (FR-011, FR-012, US5, US6)
- Per-OS routing scripts: `scripts/audio/setup_macos.py` (BlackHole),
  `setup_windows.ps1` (VB-Cable), `setup_linux.py` (PipeWire null-sink). Idempotent;
  verify + report the selected device.
- Capture: system loopback as a WebSocket audio source; output: expose a virtual-mic sink.
- Overlay: a small always-on-top caption window (Tauri/Electron shell or a separate
  PiP-style window), reusing the existing subtitle renderer.
- SDK: `bp` importable package; a documented `Pipeline` facade + REST/WS contract doc
  under `documentation/`.

## Reuse (never start dry — `documentation/s2s_landscape_2026-10-18.md` §5)

- **Study/benchmark, not fork:** SimulStreaming (AlignAtt policy) and WhisperLiveKit for
  the streaming-STT behaviour. Adopt the *pattern* (AlignAtt + local agreement) behind
  our `STTEngine` interface; do not vendor the repos.
- Keep Piper + Opus-MT (MIT/per-model) as licence-clean defaults; keep Seamless/NLLB as
  offline references only (CC-BY-NC).

## Project structure (additions)

```
languages/packs/*.json            # language packs (FR-001)
backend/registry.py               # engine registry + manifests (FR-003/004/005)
backend/engines/                  # thin wrappers implementing the stage interfaces
scripts/audio/setup_{macos,linux}.py, setup_windows.ps1   # Path B routing (FR-011)
scripts/bp.py                     # + language / engine gate / selftest subcommands
processed/language_eval/*.json    # per-pair eval records (FR-002)
processed/engine_gate/*.json      # per-engine gate records (FR-006)
documentation/sdk.md              # public surface (FR-012)
```

## Milestones (each independently demoable)

- **M1 — Languages + registry** (US1, US2): packs, CLI, registry, licence gate, stage
  gates for the *current* engines. Demo: add `cs`, swap TTS to OmniVoice, gate passes.
- **M2 — Plug-and-play + server** (US3, US4): setup rewrite, Linux verify, auth, pooling,
  health. Demo: clean clone → `bp selftest` on three OSes; two clients, one model load.
- **M3 — Path B** (US5): routing scripts, loopback capture, virtual-mic output, overlay.
  Demo: a recorded call translated live with captions over the window.
- **M4 — SDK + unified engine** (US6, US7): public facade, docs, one unified adapter
  (local Qwen3-Omni or a cloud adapter) behind the interface.

## Gates (run before claiming anything done)

1. `make test` — existing suite green (38 tests; fix the two stale items first, T001/T002).
2. `bp selftest` — on macOS + Windows + Linux (record times in `linux_setup_and_test.md`).
3. `bp engine gate <stage> <engine>` — for every engine touched.
4. `bp language test <pair>` — for every pair claimed verified.
5. Live WS rehearsal — `test/interrupt_smoke_test.py` against `make run`.
6. `make demo-check` — pre-flight.

## Complexity Tracking

| Item | Why | Constitution tension | Resolution |
|---|---|---|---|
| Language packs + registry | "any language, swap anything" is explicit owner direction | VI (YAGNI): adds indirection | Indirection is the requirement here; keep wrappers thin, no new deps. |
| Path B audio routing | no user path without it | V (plug-and-play) supports it | automate per-OS; no manual README steps. |
| **Principle III amendment** | Slovak stays reference, but "any language" is now in scope | III currently says Slovak is the only non-negotiable pair | Amend III to "Slovak is the reference pair; any pair meeting the support bar is supported". Requires owner sign-off + version bump. |

## Risks

- **Scope**: four milestones is a lot. M1/M2 are the thesis-critical set; M3/M4 are the
  product set. If time is short, ship M1+M2 and keep M3/M4 specified (this is exactly the
  track split the owner asked for).
- **Licence**: a tempting model (Seamless/NLLB) is non-commercial; the gate exists for this.
- **Windows/CPU quality**: cloned voice needs a GPU; keep the Piper CPU baseline honest
  and labelled.
