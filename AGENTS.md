# AGENTS.md — BP (real-time EN↔SK speech translation)

Project-level memory for every agent working in this repo. Read `PLAN.md` (state +
next steps), `DESIGN.md` (UI contract) and `.specify/memory/constitution.md`
(binding principles) before changing anything. Global conventions: `~/memory-vault`
(`pages/agent-rules.md`, `pages/agent-memory.md`); session journal: vault
`journals/YYYY-MM-DD.md`; work items: `pages/action-queue.md`.

## What this is

Bachelor-thesis prototype (FEI-184458-129117, STU, Slovak): near-real-time
speech translation for online conferences, EN↔SK, local-first, no cloud in the
pipeline. Stages: WebRTC VAD → faster-whisper STT → Opus-MT (CTranslate2) →
Piper/XTTS/Hybrid TTS → browser playback + subtitles + latency charts.
Backend FastAPI + WebSocket, UI plain HTML/CSS/JS. Owner: Yegor Brusnyak.

## Machines and branches (check `git log` before assuming either is current)

| Machine | Path | Role |
|---|---|---|
| Mac, M1 Pro 16GB | `~/Documents/STU/BP` (main checkout, `main`) | reference box: all measurements in `PLAN.md` are from here; has `venv`, `.venv-stt` (Parakeet), `.venv-train` (Piper fine-tune), `models/` (OpenVoice), `.env`, HF cache |
| Windows 11, AMD, CPU-only, no admin | clone of `https://github.com/brusnyak/bp` (GitHub user `I-BRUS`) | portability proof: `requirements-windows.txt`, `SETUP_WINDOWS.md`, `setup_windows.ps1`; no XTTS/Coqui (no Windows wheels), Piper-only |

Branch state and history (verified 2026-09-28):
- `main` = the 6 local commits ending at the shipped SK voice + the merge of the
  Windows CPU-setup work (`e55461c`, merged `79a8476`).
- `origin/main` had diverged (Windows laptop pushed PR #1 from `I-BRUS`); that side is
  bookmarked locally as `windows/amd-cpu-setup` and merged on `merge/windows-amd-cpu`.
- `voice-lab`, `worktree-agent-a342c9d7f70fec900` are fully merged ancestors — dead ends,
  do not branch from them.
- Windows deltas stay in `requirements-windows.txt`; `requirements.txt` keeps macOS/Linux
  installs (Coqui/omnivoice/OpenVoice live there) — do not "unify" them.

## Run it

| Command | What |
|---|---|
| `make run` | backend, `https://localhost:8000` (self-signed cert in `certs/`) |
| `make demo-check` | demo pre-flight: assets + live server (`scripts/demo_preflight.py --server`) |
| `make lab` | Voice Lab static page, `http://localhost:8080/ui/voice-lab/lab.html`, no backend |
| `make test` | piper pipeline + VAD + hardware + auth + API tests |
| `python3 scripts/update_voice_lab_library.py --no-test` | regenerate `ui/voice-lab/library.json` (never hand-edit) |
| `venv/bin/python test/interrupt_smoke_test.py` | live WS rehearsal, needs `make run` |
| `venv/bin/python scripts/e2e_ensk_new_voice.py` | full EN→SK offline run, writes `processed/e2e_ensk.json` |

Demo: `documentation/demo_runbook_2026-09-28.md` (+ `monday_test_checklist.md`,
`handler_update_2026-09.md`). Thesis: `documentation/thesis_draft.md`.

## Test reality (measured 2026-09-28, M1 Pro)

- `pytest` collects only `test/*_test.py` style files: `hardware_test.py` (7),
  `vad_tests.py` (4), `backend_auth_tests.py` (6), `backend_api_tests.py` (3).
  `make test` = 20 tests + the piper pipeline script, all green.
- `test/full_pipeline_test.py` is a **broken leftover**: it imports
  `backend.tts.f5_tts`, removed in August, so it breaks `pytest test/` collection.
  Exclude it or delete it; do not silently "fix" it into a test.
- `test/interrupt_smoke_test.py`, `backend_vad_stt_test.py`, `concurrent_*` are live
  scripts that need a running server, not pytest files.
- `test/backend_api_tests.py` used to wipe `speaker_voices/speaker_voices.json` and
  unlink `*.wav` on teardown (real data loss on a demo machine). It now snapshots and
  restores; keep it that way if you touch that fixture.

## Hard rules

- `.env` never enters git (Google client id + `JWT_SECRET`). `.env` may be symlinked
  from another checkout for a local run; never from a shared/cloud path.
- Personal voice models are committed binaries on purpose (`backend/tts/piper_models/*.onnx`);
  never retrain or overwrite them as part of unrelated work.
- No cloud TTS/translation in the pipeline (constitution III + thesis local-first rule);
  cloud services may only appear as QC references.
- SK output default is the owner's own fine-tuned voice (`piper` → remaps to
  `piper_sk_personal`); keep `session_config["tts_model_choice"]` in sync with the
  stored engine name or synthesis silently never runs (this was a real bug, fixed 2026-09-28).
- Evidence-first: numbers in docs must come from a recorded measurement, and a
  decision with evidence behind it is not re-litigated (constitution II).
- Ask before pushing to the remote or anything shared/visible (constitution, Operating Mode).

## Next steps

1. Merge/ff `main` into the Mac checkout and re-run `make test` + the rehearsal there
   (`merge/windows-amd-cpu` → `main` is a fast-forward).
2. Re-run `scripts/update_voice_lab_library.py --no-test` on the Mac after that merge to
   pick up the locally-untracked `*_v2b` takes.
3. Demo prep per the runbook; then the handler update message.
4. Open engineering items: per-language STT routing (Parakeet EN / whisper-small SK) in
   the backend, `test/full_pipeline_test.py` deletion, SK STT accuracy rung (turbo/small).
