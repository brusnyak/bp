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
| Mac, M1 Pro 16GB | `~/Documents/STU/BP` (main checkout, `main`) | reference box: all measurements in `PLAN.md` are from here; has `.venv` (post-`lite`; legacy `venv` still on disk), `.venv-stt` (Parakeet), `.venv-train` (Piper fine-tune), `models/` (OpenVoice), `.env`, HF cache |
| Windows 11, AMD, CPU-only, no admin | fork `I-BRUS/bp`, branch `lite` (GitHub user `I-BRUS`) | portability proof: one-command `scripts/setup.py`, torch-free runtime, Piper-only (no XTTS/Coqui wheels) |

Branch state and history (verified 2026-09-28, after the `lite` landing):
- Integration tip = `merge/windows-amd-cpu` = `cline/63e9f` = `main`: the 6 local commits
  ending at the shipped SK voice, PR #1 merge (`e55461c`), demo readiness + footprint audit,
  then the `lite` merge (`001b4c3`) and the POSIX voice-path fix (`a68b1db`). Check
  `git log --oneline -3` for the current hash.
- `lite` (I-BRUS fork, 6 commits over `origin/main`) landed 2026-09-28: torch-free runtime,
  one-command setup, Slovak-tuned `whisper-small` STT default (SK→EN ~3 s vs 9–11 s),
  measured demo report, 75-sentence recording kit, and the weight (piper onnx, ct2 models,
  recordings, certs) moved out of git into local-only/ignored files.
- `origin/main` is still at `f6904a5` (PR #1 only) — our line is **not pushed**; pushing
  needs the owner's go-ahead.
- `voice-lab`, `worktree-agent-a342c9d7f70fec900` are fully merged ancestors — dead ends,
  do not branch from them.

## Landing branches (protocol — still in force)

New work lands as a **branch on the remote**, never as a direct rewrite of `main`:

1. Contributor: commit on a `win/*` (or similar) branch, push, open a PR (that is how
   `79a8476` and the `lite` branch arrived).
2. Mac: `git fetch`, merge that branch into the integration line, resolve conflicts,
   then run the three gates — `make test` (38 expected), `.venv/bin/python
   test/interrupt_smoke_test.py` (needs `make run`), `make demo-check`.
3. Only after the gates pass: ff `main` at `~/Documents/STU/BP` onto the tip.

Setup model after `lite` (2026-09-28): **one** cross-platform `requirements.txt`
(+ `requirements-dev.txt` for tests, `requirements-convert.txt` for the one-time model
conversion) and one setup path — `python3.11 scripts/setup.py --dev` (the Mac's default
`python3` is 3.14, which the script rejects by design; `PYTHON=python3.11 make install`
works too). ~92 s on a warm Mac, ~850 s cold on the Windows laptop. The old
`setup_windows.ps1` / `SETUP_WINDOWS.md` / `requirements-windows.txt` were deleted by
`lite` — do not resurrect them. Plans behind that split: `documentation/footprint_audit_2026-09-28.md`
(ranked cut list) and `documentation/linux_setup_and_test.md` (third-OS test sheet).

## Run it

| Command | What |
|---|---|
| `python3.11 scripts/setup.py --dev` (= `make install`) | one-command setup: `.venv` + deps + `.env` + certs + voices + MT/STT model conversion |
| `make run` | backend, `https://localhost:8000` (self-signed cert in `certs/`) |
| `make demo-check` | demo pre-flight: assets + live server (`scripts/demo_preflight.py --server`) |
| `make lab` | Voice Lab static page, `http://localhost:8080/ui/voice-lab/lab.html`, no backend |
| `make test` | 38 tests: hardware, VAD, MT, API, auth, security, config, ratings |
| `python3 scripts/update_voice_lab_library.py --no-test` | regenerate `ui/voice-lab/library.json` (never hand-edit) |
| `.venv/bin/python test/interrupt_smoke_test.py` | live WS rehearsal, needs `make run` |
| `.venv/bin/python scripts/e2e_ensk_new_voice.py` | full EN→SK offline run, writes `processed/e2e_ensk.json` |

Demo: `documentation/demo_runbook_2026-09-28.md` (+ `monday_test_checklist.md`,
`handler_update_2026-09.md`). Thesis: `documentation/thesis_draft.md`.

## Test reality (measured 2026-09-28, M1 Pro)

- `make test` = explicit pytest over `hardware_test` (7), `vad_tests` (4),
  `backend_auth_tests` (6), `backend_api_tests` (3) — the original 20 — plus
  `mt_model_tests`, `security_tests`, `config_tests` from `lite`: **33 tests, green on
  macOS 2026-09-28 (merge gate) and on the Windows laptop (33/33)**.
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
- Personal voice models and voice recordings are **local-only, never committed**
  (`backend/tts/piper_models/*.onnx`, `speaker_voices/` — both gitignored after `lite`).
  A fresh clone falls back to the generic Piper voice (`backend/tts/base.py` `_piper`).
  The shipped SK voice (`sk_SK-personal-male-medium.onnx`, latest copy 2026-09-28 07:04)
  **cannot be re-downloaded** — back it up outside git before any `make clean`.
- One cross-platform `requirements.txt` (+ `-dev` / `-convert`); the Windows-specific
  files are gone for good. Never re-add `setup_windows.ps1` / `requirements-windows.txt`.
- No cloud TTS/translation in the pipeline (constitution III + thesis local-first rule);
  cloud services may only appear as QC references.
- SK output default is the owner's own fine-tuned voice (`piper` → remaps to
  `piper_sk_personal`); keep `session_config["tts_model_choice"]` in sync with the
  stored engine name or synthesis silently never runs (this was a real bug, fixed 2026-09-28).
- Evidence-first: numbers in docs must come from a recorded measurement, and a
  decision with evidence behind it is not re-litigated (constitution II).
- Ask before pushing to the remote or anything shared/visible (constitution, Operating Mode).

## Next steps (owner-approved order, 2026-09-28)

1. [x] **Pushed `main` to `origin/main`** (tip: 920fd24 including Windows/AMD CPU setup, lite, and SK->EN evaluation matrix).
2. **Accelerate SK→EN turnaround latency**:
   - Target: close the ~2x latency gap (SK→EN ~1.4s vs EN→SK ~0.7s).
   - Analysis: MT and TTS are already sub-150ms. The gap is 100% in Slovak STT (`whisper-small-sk` at ~1.3s vs EN `base` at ~0.55s).
   - Test `beam_size=1` (greedy) vs `beam_size=2` / `5` in `faster-whisper` on Slovak speech with `scripts/eval_sk_direction.py` to check speedup vs accuracy trade-off.
   - Benchmark `initial_prompt` with common Slovak diacritics / orthography.
   - Profile `cpu_threads` and chunking thresholds.
3. Voice cloning speed + quality: record the 75-sentence set
   (`scripts/build_recording_set.py` → `scripts/record_reading.py`), longer Piper fine-tune,
   QC in the Voice Lab (WER thirds / F0 / HNR / Praat panel).
4. Updated recording transcript for the owner (SK + CZ — include the ElevenLabs
   instant-voice-cloning reference texts, which double as colloquial STT/MT test material).
5. Linux: execute `documentation/linux_setup_and_test.md` and wire
   `documentation/ci.yml.example` into `.github/workflows/ci.yml`.
6. Voice Lab analytics → demo-ready: E2E + S2S charts from `scripts/demo_conversation.py`
   and the latency/STT JSONs, speed + quality panels, and the showcase page
   (done / worked-on / future work).
7. Housekeeping: delete `test/full_pipeline_test.py`; re-run
   `python3 scripts/update_voice_lab_library.py --no-test` for the `*_v2b` takes.
8. **GPU path (2026-10-02, evidence in `documentation/model_landscape_2026-10.md` §10.2–10.4):** free T4 (Colab + Kaggle) runs
   OmniVoice cloning at RTF 0.14–0.17 (long) / ~0.7 s per phrase and X-Voice at 0.46–0.80; owner ear: clones good, Seamless =
   generic voice. Next: (a) meeting simulation with OmniVoice as TTS on a GPU (`scripts/meeting_sim.py`, timeline input →
   translation → output); (b) live measurement on the presentation machine (needs a GPU box; Kaggle/Colab cannot run the
   live mic path); (c) capture numeric ear grades (log in on :8000 or export JSON). Piper fine-tuning only matters for CPU-only
   machines. Kaggle CLI + private dataset `yegorby/bp-gpu-bench-refs` + Modal token were created for this; revoke when done.
