# AGENTS.md — BP (real-time EN↔SK speech translation)

> Current state (2026-11-07): start with `documentation/STATUS.md` (where the project stands), `documentation/LANGUAGES.md` and `documentation/BENCHMARKS.md`; every quoted number is checked by `make check-docs`. Older state (2026-11-04): `PRODUCT.md`, `documentation/implementation-2026-11-04/RESULTS.md` and `documentation/archive/2026-11-07-stale/HANDOFF_NEXT_SESSION.md`. Earlier dated counts, readiness claims, media-access behavior and Kaggle failure explanations below are historical. Current contracts require authenticated explicit preparation, offline assets, target-language TTS, terminal Stop and reviewed full-route evidence. Full cloned-speech/LHC/commercial acceptance remains pending.

Project-level memory for every agent working in this repo. Read `documentation/archive/2026-11-07-stale/PLAN.md` (state +
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
| Mac, M1 Pro 16GB | `~/Documents/STU/BP` (main checkout, `main`) | reference box: all measurements in `documentation/archive/2026-11-07-stale/PLAN.md` are from here; has `.venv` (post-`lite`; legacy `venv` still on disk), `.venv-stt` (Parakeet), `.venv-train` (Piper fine-tune), `models/` (OpenVoice), `.env`, HF cache |
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
- 2026-11-03 update: the line **is pushed**; `main` = `origin/main` = `faf68ff` (CI matrix
  + SDK/thesis-numbers commit, cloned fresh 2026-10-30). Hand-written dirty batch
  (UPDATES.md 2026-10-30: SK→EN quality tiers env `BP_SK_STT_MODEL`, JWT on WS, `bench/`)
  sits uncommitted, awaiting owner decision. Demo voice restored from
  `~/Documents/STU/BP-voice-backup` + `GOOGLE_CLIENT_ID/SECRET` appended to `.env` →
  preflight "ALL REQUIRED CHECKS PASSED" (2026-11-03).
- `voice-lab`, `worktree-agent-a342c9d7f70fec900` are fully merged ancestors — dead ends,
  do not branch from them.

## Landing branches (protocol — still in force)

New work lands as a **branch on the remote**, never as a direct rewrite of `main`:

1. Contributor: commit on a `win/*` (or similar) branch, push, open a PR (that is how
   `79a8476` and the `lite` branch arrived).
2. Mac: `git fetch`, merge that branch into the integration line, resolve conflicts,
   then run the three gates — `make test` (87 expected, verified 2026-11-06), `.venv/bin/python
   test/interrupt_smoke_test.py` (needs `make run`), `make demo-check`.
3. Only after the gates pass: ff `main` at `~/Documents/STU/BP` onto the tip.

Setup model after `lite` (2026-09-28): **one** cross-platform `requirements.txt`
(+ `requirements-dev.txt` for tests, `requirements-convert.txt` for the one-time model
conversion) and one setup path — `python3.11 scripts/setup.py --dev` (the Mac's default
`python3` is 3.14, which the script rejects by design; `PYTHON=python3.11 make install`
works too). ~92 s on a warm Mac, ~850 s cold on the Windows laptop. The old
`setup_windows.ps1` / `SETUP_WINDOWS.md` / `requirements-windows.txt` were deleted by
`lite` — do not resurrect them. Plans behind that split: `documentation/archive/2026-11-07-stale/footprint_audit_2026-09-28.md`
(ranked cut list) and `documentation/linux_setup_and_test.md` (third-OS test sheet).

## Run it

| Command | What |
|---|---|
| `python3.11 scripts/setup.py --dev` (= `make install`) | one-command setup: `.venv` + deps + `.env` + certs + voices + MT/STT model conversion |
| `make run` | backend, `http://localhost:8000` (plain HTTP; `BP_HTTPS=1` for HTTPS with the self-signed cert in `certs/`) |
| `make demo-check` | demo pre-flight: assets + live server (`scripts/demo_preflight.py --server`) |
| `make lab` | Voice Lab static page, `http://localhost:8080/ui/voice-lab/lab.html`, no backend |
| `make test` | 87 tests: hardware, VAD, MT, API, auth, security, config, ratings |
| `python3 scripts/update_voice_lab_library.py --no-test` | regenerate `ui/voice-lab/library.json` (never hand-edit) |
| `.venv/bin/python test/interrupt_smoke_test.py` | live WS rehearsal, needs `make run` |
| `.venv/bin/python scripts/e2e_ensk_new_voice.py` | full EN→SK offline run, writes `processed/e2e_ensk.json` |

Demo: `documentation/archive/2026-11-07-stale/demo_runbook_2026-09-28.md` (+ `documentation/archive/2026-11-07-stale/monday_test_checklist.md`). Thesis: `documentation/thesis_draft.md`.

## Architecture v2 — streaming redesign (in progress)

Ground-up redesign for near real-time streaming + multi-speaker meetings. New modules:

| Module | File | Purpose |
|--------|------|---------|
| Hardware auto-config | `backend/hardware_scan.py` | Scans CPU/RAM/GPU at startup, selects STT/MT/TTS/VAD config per machine |
| Meeting session manager | `backend/session_manager.py` | Multi-speaker meeting rooms, per-speaker voice, per-listener language |
| Streaming pipeline | `backend/streaming/pipeline.py` | True STT→MT→TTS streaming (0.7s windows, local-agreement, fragment emit) |
| Streaming spec | `documentation/archive/2026-11-07-stale/STREAMING_SPEC.md` | Latency budget, config env vars, test plan |
| Multi-speaker design | `documentation/MULTI_SPEAKER.md` | Room model, translation matrix, API endpoints |
| Architecture overview | `documentation/archive/2026-11-07-stale/ARCHITECTURE_V2.md` | Full v2 design, deployment modes, LHC fit |

Key change vs v1: v1 waits for silence/3s segment before STT→MT→TTS (~6-8s first word). v2 streams 0.7s audio windows, commits words via local-agreement, emits translated fragments as they're ready (~0.9s first word on CPU, ~0.5s on GPU T4).

WebSocket endpoints: existing `/ws` (single-user, backward compatible) + `/ws/meeting` (multi-speaker room). HTTP: `/api/hardware` (auto-config), `/api/meeting/create|list|{room_id}`.

Kaggle full-route test: `scripts/gpu_bench/kaggle_full_route/kaggle_full_route.py` — pushes to Tesla T4, measures p50/p95 first-audio for 90s continuous SK→EN speech.

Voices: `speaker_voices/` has 7 owner recordings (EN/SK/CS with transcripts). Backup at `~/Documents/STU/BP-voice-backup/` matches. Piper models in `backend/tts/piper_models/` (incl. `sk_SK-personal-male-medium`, `en_US-personal-v2`). CT2 models in `ct2_models/` (whisper-base, whisper-small-sk, all opus-mt pairs).

## Test reality (measured 2026-09-28, M1 Pro)

- `make test` = explicit pytest over `hardware_test` (7), `vad_tests` (4),
  `backend_auth_tests` (6), `backend_api_tests` (3) — the original 20 — plus
  `mt_model_tests`, `security_tests`, `config_tests` from `lite`: **87 tests, green on
  macOS 2026-11-06 (verified)**.
- `test/full_pipeline_test.py` deleted (was a broken F5 leftover importing removed `backend.tts.f5_tts`).
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
   - `whisper-small-sk` is NOT a safe SK→EN default (Czech-like hallucinations, WER 0.25–0.32 > 0.2 gate, measured `bench/results` 2026-11-02). `initialize_all_models` now sets `session_config["stt_experimental_sk"]=True` for SK source unless `BP_SK_STT_MODEL` is set; the UI must flag SK→EN as experimental.
   - Quality path: `BP_SK_STT_MODEL=kinit/whisper-large-v3-turbo-sk` (~6–13% WER per SloPal 2026, ~800 MB). Verify Mac RTF acceptable before shipping SK→EN live.
   - Machine-aware model selection (run anywhere): `BP_DEVICE=cpu|cuda|mps|auto` (STT), `BP_COMPUTE_TYPE=int8|float16|float32`, `BP_MT_DEVICE=cpu|cuda` (CT2 MT). Defaults: STT auto/int8, MT cpu/int8.
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
8. **GPU path (T4 / ≤8 GB VRAM target):** `backend/registry.py` declares optional GPU engines — Canary AST (SK→EN, CC-BY-4.0), Kokoro (EN TTS, RTF 0.03, 1.33 GB VRAM on T4), Qwen3-TTS / Chatterbox (EN zero-shot cloning, 4–8 GB VRAM, MIT), SeamlessM4T v2 (S2S, CC-BY-NC non-commercial, quality reference only). Canary AST measured on Kaggle T4 2026-11-03 (SK ASR WER 0.0477 <0.2 ✅, but SK→EN AST chrF++ ~52 below commercial bar) — speed path, not quality path; NOT promoted to live route. CPU baseline (Piper + faster-whisper) intact for torch-free Windows.
9. **Kaggle T4 validation:** Canary AST done 2026-11-03 (not promoted). OmniVoice live-speech / long-sentence / presentation-pipeline test submitted to Kaggle (bp-omnivoice-matrix-20261104, RUNNING). Kokoro EN ear-grading pending. Revoke Kaggle/Modal tokens when done.
10. **Load tests (T4, `documentation/load_tests_2026-10.md`):** OmniVoice clone RTF 0.18–0.34 (batch/QC path only — NOT live). Live path = Piper SK/EN on CPU (RTF ~0.04) + optional Kokoro EN GPU (RTF 0.03). New split: Kokoro EN + Piper SK on T4 (load test pending).
11. **Project state (2026-10-02, owner):** ratings imported (`scripts/grade_library.py --apply --import voice-ratings.json`). Everything left needs better compute: live mic run on GPU machine, faster GPU numbers, duration cap vs playback backlog.
12. **Lightened + dry-setup measured (2026-10-02, `documentation/archive/2026-11-07-stale/setup_friction_2026-10.md`):** project 9.4 GB -> 2.9 GB; fresh clone = 89 s / 705 MB .git; `setup.py --dev` 428 s, tests 38/38, ~9 min total. Owner voice backup outside git: `~/Documents/STU/BP-voice-backup`.

## Session 2026-10-20 — Night Mode Findings

### SK→EN STT Trade-off (measured)
| Config | WER | STT Time | RTF | Notes |
|--------|-----|----------|-----|-------|
| small-sk, beam=5, prompt | 0.255 | 18.4s | 0.14 | Current default (quality) |
| small-sk, beam=1, prompt | **0.323** | **15.8s** | **0.12** | Fast mode — viable for demo |
| base, beam=5, prompt | 0.714 | 8.9s | 0.07 | Unusable for SK |
| large-v3-turbo, beam=1, prompt | 0.828 | 80.8s (30s chunks) | 0.60 | Too slow on CPU |

**Root cause**: `whisper-small-sk` fine-tune produces Czech-like hallucinations ("šivé okážky" for "živej ukážke"). Not a config issue — the model itself is flawed.

**Decision**: Keep `beam_size=5` as default for quality. Add `BP_STT_BEAM_SIZE=1` as "fast mode" env var. Document honestly in thesis.

### PiP Styling Updated
- Canvas size 640×280 (was 640×220)
- Dark glass background with vignette, brand accent bar, text shadows
- 3 lines transcript + 3 lines translation, divider line
- Softer green (#5eead4) for translation, white for source

### Documentation Created
- `documentation/project_status_2026-10-20.md` — comprehensive status, swap ladder, CLI ref
- `documentation/cli_quickref.md` — one-page CLI cheat sheet

### Updated Next Steps (Priority Order)

**P0 — Shippable for "picky old man" (occasional lecture/meeting use)**
1. T022 Floating caption overlay (text overlay, not just PiP canvas)
2. T010 Wire `registry.select()`/`instantiate()` into `main.py` (config-driven engine swap)
3. T031 Add `/health` + `/metrics` HTTP routes (server observability)
4. T023b Document SK→EN trade-off, make `beam_size` configurable per language pack

**P1 — Polish + Voice**
5. T029 RVC-MLX model prep: train owner voice via Applio (rent GPU ~1h), register as `rvc-mlx`
6. T030 `bp engine gate` CI integration (thresholds exist, need automation)
7. T028b Move live-test scripts out of `test/` (backend_vad_stt_test.py, piper_pipeline_test.py)

**P2 — SDK + Collaboration**
8. T024 Public `Pipeline` facade + `documentation/sdk.md` (REST/WS contract)
9. T025 Unified engine adapter (Qwen3-Omni local or cloud adapter)
10. T026 Supervisor handbook: add language, add engine, embed SDK

**P3 — Thesis + Evidence**
11. T027 Update thesis draft with measured numbers, remove `[DOPLNIŤ]` markers
12. T028 Re-run three demo gates + `interrupt_smoke_test.py`, date the runbook

**P4 — Platform (when project complete)**
13. Linux x86_64: run `documentation/linux_setup_and_test.md`, wire CI
14. Kaggle/Modal cleanup: revoke tokens from GPU benchmarks

## Session 2026-10-21 — Final Verification Complete

### All P0 Blockers Resolved ✅

| Issue | Fixed | Evidence |
|-------|-------|----------|
| SK→EN "verified" but hallucinated Czech | Now **experimental** (WER 0.27 > 0.2 threshold) | Auto-fetches reference, computes WER, enforces quality gate |
| `bp language` commands missing | Wired into `bp.py` | `bp language list|test|add|remove|show` all work |
| `setup.py` didn't run `bp selftest` | Added at end of setup | `python3.11 scripts/setup.py --dev` runs selftest, passes |
| `e2e` test broken (wrong file) | Fixed path to `me_en_script.m4a` | `bp e2e` runs (needs audio file) |
| Live test scripts in `test/` | Moved to `scripts/live/` | `make test` = 87/87 clean |
| `/health` + `/metrics` missing | Added @router.get endpoints | `GET /health` (public), `GET /metrics` (auth) |
| Config-driven engine swap | Registry `instantiate()` used for all stages | STT/MT/TTS/VAD engine names from session config |
| Per-pack `beam_size` | Backend reads `config.get("beam_size")` | Frontend wiring pending |
| Licence gate | Fail-closed: non-commercial blocked | XTTS, NLLB blocked unless explicit override |

### Verification Results (2026-10-21)

| Test | Result |
|------|--------|
| `make test` | 87/87 pass |
| `bp selftest` | PASSED (5 warnings for optional engines) |
| `bp engine gate` (4/4) | All pass (STT, MT, TTS, VAD) |
| `bp language list` | 3 verified (en-sk, en-cs), 1 experimental (sk-en) |
| Live pipeline (WebSocket) | EN→SK ✅, SK→EN (model flaw), EN→CS ✅ |
| TTS engine swap | `piper` ↔ `piper_sk_personal` both work |
| Demo preflight | ALL REQUIRED CHECKS PASSED |
| Engine registry | 15 engines registered, licence gate works |

### Known Limitation: SK→EN Direction
- `whisper-small-sk` fine-tune produces Czech-like hallucinations
- Transcript: "viete si v tejto šivejokáške..." vs "vitajte v tejto živej ukážke..."
- WER 0.27 (beam=5) / 0.32 (beam=1) — above 0.2 threshold
- Root cause: model flaw, not config. Needs GPU model (Voxtral/Parakeet) or RVC-MLX voice layer.

### Ready for "Picky Old Man" Use
- One-command setup: `python3.11 scripts/setup.py --dev` → ends with `bp selftest`
- Plug-and-play language packs: `bp language add <pair>` → auto-resolves models, runs smoke test
- Engine swap via config: change `stt_engine`, `mt_engine`, `tts_engine` in pack
- Health/metrics for server ops: `/health` (public), `/metrics` (auth)
- 38 pass; demo preflight passes
