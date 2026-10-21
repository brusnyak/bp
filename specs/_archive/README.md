# Archived specs

These are kept for history, not for execution. The single active spec is
`specs/003-end-to-end-completion/`.

## Why archived (2026-10-18, owner)

001 and 002 had drifted and their open tasks were being confused with current scope.
Per owner direction, the still-open, still-relevant work was **fused** into
`specs/003-end-to-end-completion/spec.md` (§Source map) rather than tracked in two places.

| Archived | What it was | Still-open work now lives in 003 as |
|---|---|---|
| `001-realtime-cross-platform-translation` | Cross-platform pipeline, hardware backends, per-OS audio routing, voice tiers, streaming captions | FR-003/US2 (backend wiring), FR-011/US5 (audio routing), FR-009/US4 (model pool), US2/T006 (captions) |
| `002-pwa-pip-demo` | PWA control room + PiP subtitles + Monday demo sprint | US3 (install/verify), US5/T022 (overlay), phase-3 tasks |

Closed as not-applicable (recorded so they are not re-opened):
- 001 T009 — `mlx-whisper` gating: no usage in repo, nothing to gate.
- 001 T019 — concurrency ceiling: superseded by `documentation/load_tests_2026-10.md`.
- 001 T023 — coqui `TTS` pin: moot. The one-command setup never installs coqui; `requirements.txt`
  carries only `piper-tts==1.3.0` (2026-10-18 check). `backend/tts/coqui_tts.py` imports `torch` +
  `TTS.api` at module level and is guarded at the call site, so XTTS is an opt-in extra on machines
  that install it themselves. If it is ever needed, the maintained fork is idiap `coqui-tts` — but
  XTTS v2 is non-commercial (CPML), so it stays out of the default profile regardless (see T008 gate).
- 001 T024 — verify `speaker_cache` conditioning-latent caching in xtts: not carried. Verification-only,
  and XTTS is non-commercial, non-default, and not installed on the reference `.venv` (`bp engine list`
  reports `xtts --`). Re-open only if XTTS is actually put on the critical path.

Do not branch from, edit, or cite these files as current scope. Read 003.
