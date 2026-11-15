# Documentation index — START HERE

**Single entry point:** [`MASTER_README.md`](MASTER_README.md) — project overview, current state, model landscape, Kaggle T4 results, E2E streaming tests, ordered actions, and how to continue next session. Read this first.

---

## Start here (single source of truth)

| Read | When |
|---|---|
| [`MASTER_README.md`](MASTER_README.md) | **Single source of truth** — project overview, current state, model landscape, Kaggle T4 results, E2E streaming test scripts, ordered actions, next-session handoff. **Start here every session.** |

---

## Reference material (read only when MASTER_README.md points you here)

| Read | When |
|---|---|
| [`PLAN_2026-11-10.md`](PLAN_2026-11-10.md) | **Current plan and decisions** (2 s goal, Slovak-input result, voice options, ordered actions, project decisions). Wins over older pages where they disagree |
| [`HANDOFF_2026-11-10.md`](HANDOFF_2026-11-10.md) | Decisions, measured state, Kaggle kernels, ordered next steps (2026-11-10) |
| [`evidence/2026-11-10/`](evidence/2026-11-10/) | Receipts for the plan: strategy matrix on generic audio, TTS latency probes |
| [`STATUS.md`](STATUS.md) | Where the project stands: what works, what is measured but not built, what is not built, decisions |
| [`LANGUAGES.md`](LANGUAGES.md) | Best configuration per language and hardware, with measured quality and timing |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Method, results, negative results and limits; how to rerun on Kaggle |
| [`NEAR_REALTIME_GOAL.md`](NEAR_REALTIME_GOAL.md) | The near real-time goal, the Google, Microsoft and OpenAI behaviour it copies, and the success test |
| [`REALTIME_REVIEW_2026-11-09.md`](REALTIME_REVIEW_2026-11-09.md) | Are we measuring the right things; what Piper and OmniVoice need to behave like the big three; better instruments |
| [`REALTIME_OPTIONS_2026-11-08.md`](REALTIME_OPTIONS_2026-11-08.md) | Hosted services and open models for near real-time, what was read and what was not |
| [`DEMO.md`](DEMO.md) | A presentation test from a clean clone, including the room demo |
| [`reading_script_multilingual.md`](reading_script_multilingual.md) | What to read when recording English, Slovak or German for scoring (`scripts/score_reading.py`) |
| [`KAGGLE_SETUP.md`](KAGGLE_SETUP.md) | Running benchmark kernels on a Kaggle T4 |
| [`linux_setup_and_test.md`](linux_setup_and_test.md) | Linux install notes and test ladder |
| [`MULTI_USER_SERVER.md`](MULTI_USER_SERVER.md), [`MULTI_SPEAKER.md`](MULTI_SPEAKER.md) | Older design notes for shared use (they describe the unvalidated `/ws/meeting` scaffold; room mode as built is in `backend/room.py` and `STATUS.md`) |
| [`sdk.md`](sdk.md), [`cli_quickref.md`](cli_quickref.md) | Interfaces and commands |
| [`model_landscape_2026-10.md`](model_landscape_2026-10.md), [`s2s_landscape_2026-10-18.md`](s2s_landscape_2026-10-18.md) | Engine research with `MEASURED`/`VENDOR` labels (October; superseded by BENCHMARKS where they overlap) |
| [`LICENSE_INVENTORY_2026-11-06.md`](LICENSE_INVENTORY_2026-11-06.md) | Licences of engines, models and voices |
| [`VERIFICATION_RESULTS_2026-11-06.md`](VERIFICATION_RESULTS_2026-11-06.md), [`load_tests_2026-10.md`](load_tests_2026-10.md) | Earlier verification and load measurements (read with the dates in mind) |
| [`fundamentals.md`](fundamentals.md), [`fundamentals_deep.md`](fundamentals_deep.md), [`technical_report.md`](technical_report.md), [`piper_training_runbook.md`](piper_training_runbook.md), [`supabase_auth_db_guide.md`](supabase_auth_db_guide.md) | Background, technical report text, voice training, auth |

## New test scripts (this session)

| Read | When |
|---|---|
| [`scripts/live/e2e_streaming_test.py`](../scripts/live/e2e_streaming_test.py) | **New** — full STT→MT→TTS pipeline via WebSocket, per-stage latency measurement |
| [`scripts/live/e2e_compare_tts.py`](../scripts/live/e2e_compare_tts.py) | **New** — compare multiple TTS engines against Piper/OmniVoice baselines |

## Evidence (every quoted number has a receipt here)

| Read | When |
|---|---|
| [`evidence/2026-11-07/`](evidence/2026-11-07/) | STT/MT/chain/route measurements (base for check-docs) |
| [`evidence/2026-11-09/`](evidence/2026-11-09/) | Voxtral route tests |
| [`evidence/2026-11-10/`](evidence/2026-11-10/) | Strategy matrix, TTS probes, room test |

`make check-docs` verifies every marked number against its receipt (57/57 pass).

---

## Archived (kept for history, not maintained)

| Read | When |
|---|---|
| [`archive/`](archive/) | Old architecture and streaming specs, runbooks, handoffs |
| [`archive/2026-11-07-stale/`](archive/2026-11-07-stale/) | September specs with contradictory latency budgets, old plan and file index |

**Do not read archive files for current state — they contain contradictory numbers and superseded decisions.**
