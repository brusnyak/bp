# Documentation index

26 files (2026-10-01). Start with the first group; the rest is lookup by task.
`documentation/README.md` is a stale overview (names SeamlessM4T/F5-TTS as live
components — both retired); trust PLAN.md + the files below over it.

## Start here

| File | What |
|---|---|
| `handler_update_2026-09.md` | Covering note for the handler, quotes only measured rows |
| `compute_capacity_report_2026-09.md` | Mac vs NVIDIA laptop vs university server, MEASURED / PROJECTED labelled |
| `demo_runbook_2026-09-28.md` | Stage demo pre-flight, script, fallback ladder |
| `monday_test_checklist.md` | Dry-run checklist |

## Voice work (the active thread)

| File | What |
|---|---|
| `verification_omni_piper_sk_2026-09-30.md` | New-voice verdict: SUPERSEDED by ear — both personal voices kill-graded for trembling (owner 2026-10-01); intelligibility numbers stand |
| `model_landscape_2026-10.md` | Current landscape: STT/MT/TTS/S2S verdicts + runnable matrix + eval order (replaces 6 files below). §10 = 2026-10-02 addendum (X-Voice lead) |
| `piper_training_runbook.md` | No-repeat training sheet: epoch targets, corpus rules, resume chain, fan-noise options |
| `recording_session_2026-09-28.md` | Mic recording protocol that produced the v2b takes |
| `recording_script_sk_v2.md` / `recording_script_v2.md` / `reading_script_bilingual.md` | What the owner read |

## Pipeline evidence

| File | What |
|---|---|
| `demo_report_2026-09.md` (+ `demo_report_skeleton.md`) | Measured demo numbers |
| `progress_since_the_break.md` | Narrative progress log |
| `concurrency_accuracy_2026-08-21.md` | Concurrency vs accuracy notes |

## Setup / run / admin

| File | What |
|---|---|
| `footprint_audit_2026-09-28.md` | Ranked disk/venv cut list (open: delete legacy venvs) |
| `linux_setup_and_test.md` | Third-OS test sheet (not executed) |
| `security_audit_2026-09.md` | Auth/secrets review |
| `supabase_auth_db_guide.md` | Auth + DB guide |
| `project_overview.md` | Generic overview (see README staleness note above) |

## CLI (`scripts/bp.py` — the script index; `bp --help` is authoritative)

| Command | Script | When |
|---|---|---|

Unwrapped on purpose: one-shot setup (`setup.py`, `convert_*`, `fetch_whisper.py`,
`gen_cert.py`), diagnostics (`*_probe.py`, `streaming_audit.py` in
`scripts/archive/`), retired spikes (`stt_parakeet_spike.py`,
`eval_*.py`, `benchmark_*.py` — numbers already in PLAN.md / landscape report).

| `README.md` | STALE overview (names retired components) — read this index instead |
| `fundamentals_deep.md` | Companion to fundamentals: verifiable numbers, system python only |
| `reading_script_bilingual.md` | Recording source text; parsed by `bp stt` for STT refs — do not reformat |
| `voice_and_app_direction_2026-09.md` | Owner requirements (one voice, multilingual, any OS) |

## Thesis

| File | What |
|---|---|
| `thesis_draft.md` | Draft; one `[DOPLNIŤ]` marker left (AI-declaration skeleton) |
| `fundamentals.md` / `fundamentals_deep.md` | Theory chapters (fundamentals §9 connectors, §10 ground basics + exercises) |

## Research spikes (closed, read-only)

S2S, simultaneous-translation, hybrid-TTS/OpenVoice, Coqui perf, realtime-rearchitecture,
SK/DE bootstrap, UX walkaround, TTS comparison, performance test results — all in
`hybrid_tts_openvoice_2026-08-21.md`, `COQUI_TTS_PERFORMANCE_REPORT.md`,
`realtime_pipeline_rearchitecture_2026-08-22.md`, `sk_de_bootstrap_findings_2026-08-19.md`,
`ux_walkaround.md`, `TTS_COMPARISON_REPORT.md`, `PERFORMANCE_TEST_RESULTS.md`,
`personal_voice_bootstrap_2026-08-19.md`, `coqui_tts_sections.md`.
The six superseded research files were deleted 2026-10-01 (content in git history);
 their verdicts live in `model_landscape_2026-10.md` §8. Retired one-shot scripts live in `scripts/archive/` and dead test scripts in `test/archive/` (runnable, unindexed).
