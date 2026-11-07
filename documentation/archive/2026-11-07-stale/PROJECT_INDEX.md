# BP Project Index — Complete Asset & File Inventory

**Generated:** 2026-11-05  
**Purpose:** Single source of truth for every asset in the repo — what exists, where, what it's used for, and whether it's active or stale.

---

## 1. Entry Points & Configuration

| File | Purpose | Status |
|------|---------|--------|
| `app.py` | FastAPI application factory, WebSocket endpoints (`/ws`, `/ws/meeting`), HTTP routes (`/api/hardware`, `/api/meeting/*`) | **Active** (v2) |
| `Makefile` | Install, run, test, demo-check targets | **Active** |
| `requirements.txt` | Core Python dependencies | **Active** |
| `requirements-dev.txt` | Test dependencies | **Active** |
| `requirements-convert.txt` | Model conversion dependencies | **Active** |
| `.env` | Secrets (JWT_SECRET, GOOGLE_CLIENT_ID) — gitignored, never committed | **Active** |
| `.env.example` | Template for `.env` | **Active** |
| `scripts/setup.py` | One-command setup: venv, deps, .env, certs, voices, MT/STT conversion | **Active** |

---

## 2. Backend Modules

### 2.1 Active (v2 + legacy used by app.py)

| File | Purpose | Used By |
|------|---------|---------|
| `backend/main.py` | FastAPI router: auth, voices, ratings, WebSocket `/ws`, model pool, pipeline init | **app.py** |
| `backend/hardware_scan.py` | **v2**: OS/CPU/RAM/GPU scan → auto-select STT/MT/TTS/VAD config | **app.py** (v2) |
| `backend/session_manager.py` | **v2**: Meeting rooms, per-speaker/listener sessions | **app.py** (v2) |
| `backend/streaming/pipeline.py` | **v2**: Streaming STT→MT→TTS pipeline (0.7s windows, local-agreement) | **app.py** (v2) |
| `backend/pipeline.py` | Public Pipeline facade (SDK entry point, non-streaming) | Tests, SDK |
| `backend/registry.py` | Engine registry + capability manifest + licence gate | **main.py**, tests |
| `backend/stt/faster_whisper_stt.py` | Faster-whisper STT wrapper | **main.py**, **streaming/pipeline.py** |
| `backend/mt/ctranslate2_mt.py` | Opus-MT via CTranslate2 | **main.py**, **streaming/pipeline.py** |
| `backend/tts/base.py` | TTS engine registry (Piper, OmniVoice, XTTS, Hybrid) | **main.py** |
| `backend/tts/piper_tts.py` | Piper TTS engine | **base.py** |
| `backend/tts/omni_tts.py` | OmniVoice TTS engine (GPU) | **base.py** |
| `backend/tts/hybrid_tts.py` | Hybrid TTS engine | **base.py** |
| `backend/tts/coqui_tts.py` | XTTS v2 engine (non-commercial) | **base.py** |
| `backend/tts/text_chunking.py` | Clause-level text splitting for streaming TTS | TTS engines |
| `backend/tts/download_piper_models.py` | Download Piper voices by model ID | **scripts/setup.py** |
| `backend/tts/rvc_mlx.py` | RVC-MLX voice conversion (Mac MLX) | Declared in registry only |
| `backend/utils/auth.py` | JWT, password hashing, rate limiting | **main.py** |
| `backend/utils/db_manager.py` | SQLAlchemy DB, User model, session | **main.py**, app.py |
| `backend/utils/audio_utils.py` | Audio load/save/resample | **main.py** |
| `backend/stt/incremental.py` | Local-agreement word commit + fragmenter | **streaming/pipeline.py** |
| `backend/stt/streaming_captions.py` | Vendored whisper_streaming captioner (additive captions only) | **main.py** |
| `backend/stt/whisper_streaming_vendor/` | Vendored whisper-online + Silero VAD | **streaming_captions.py** |

### 2.2 Stale / Unused (candidates for removal)

| File | Purpose | Status | Reason |
|------|---------|--------|--------|
| `backend/database.py` | Stub — says "moved to backend/utils/db_manager.py" | **STALE** | Zero imports |
| `backend/hardware.py` | Per-stage hardware-backend detection | **DUPLICATE** | Superseded by `hardware_scan.py` (v2) |
| `backend/hardware_detect.py` | Hardware/OS detection + self-configuration | **STALE** | Zero imports anywhere |
| `backend/mt/nllb_mt.py` | NLLB-200 MT (non-commercial) | **UNUSED** | Declared in registry, never instantiated in live path |
| `backend/mt/hybrid_mt.py` | Hybrid MT (Opus-MT + NLLB fallback) | **UNUSED** | Declared in registry, never used live |
| `backend/mt/ssbd_ctranslate2_mt.py` | SSBD speculative decoding wrapper | **ARCHIVE** | Only referenced by `scripts/archive/` |
| `backend/tts/rvc_mlx.py` | RVC-MLX voice conversion | **DECLARED ONLY** | Factory in registry, `factory=None` → not built |

### 2.3 Tests

| File | Purpose | Status |
|------|---------|--------|
| `test/hardware_test.py` | Hardware detection tests | **Active** |
| `test/hardware_scan_tests.py` | **v2**: Hardware auto-config tests | **Active** |
| `test/vad_tests.py` | WebRTC VAD + live pipeline tests | **Active** |
| `test/mt_model_tests.py` | MT model tests | **Active** |
| `test/backend_api_tests.py` | API smoke tests (live models) | **Active** |
| `test/backend_auth_tests.py` | Auth, guest, rate limit tests | **Active** |
| `test/security_tests.py` | Security regression tests | **Active** |
| `test/config_tests.py` | SK STT model selection tests | **Active** |
| `test/readiness_regression_test.py` | Readiness regression | **Active** |
| `test/language_evidence_test.py` | Language evidence tests | **Active** |
| `test/incremental_tests.py` | Incremental fragmenter unit tests | **Active** |
| `test/session_playback_test.cjs` | Node playback lifecycle tests | **Active** |
| `test/archive/` | Old/broken tests (full_pipeline_test, etc.) | **STALE** |
| `test/concurrent_performance_test.py` | Concurrent user simulation | **Unused** |
| `test/i18n_dict_test.py` | i18n JSON validation | **Unused** |
| `test/model_testing_framework.py` | WER/BLEU/METEOR framework | **Unused** |
| `test/voice_similarity.py` | Voice similarity scoring | **Unused** |

---

## 3. Frontend (UI)

| Directory | Purpose | Status |
|-----------|---------|--------|
| `ui/home/` | Landing page | **Active** |
| `ui/live-speech/` | Main translation UI (mic, captions, playback, PiP) | **Active** |
| `ui/auth/` | Login/register/Google auth | **Active** |
| `ui/enroll/` | Voice enrollment/recording | **Active** |
| `ui/extra/` | About, FAQ, privacy, terms, cookies, feedback | **Active** |
| `ui/voice-lab/` | Private Voice Lab (QC, charts, library) | **Active** (private) |
| `ui/test-stt/` | STT testing page | **Active** |
| `ui/i18n/` | SK, DE translations | **Active** |
| `ui/vendor/` | Chart.js, fonts | **Active** |
| `ui/images/` | Brand assets | **Active** |

---

## 4. Assets

### 4.1 Speaker Voices (`speaker_voices/`) — gitignored, local-only

| File | Duration | Language | Transcript | Purpose |
|------|----------|----------|------------|---------|
| `me_en_script.m4a` | ~12s | EN | Full demo script | EN source for EN→SK |
| `me_en_rainbow.m4a` | ~41s | EN | "When the sunlight strikes raindrops..." | EN test |
| `me_en_rainbow_b.m4a` | ~46s | EN | Same as rainbow | EN test B |
| `me_sk_script.m4a` | ~134s | SK | Slovak demo script | SK source for SK→EN |
| `me_sk_trhove.m4a` | ~104s | SK | Slovak market story | SK source (longer) |
| `me_sk_trhove_b.m4a` | ~109s | SK | Same as trhove | SK source B |
| `me_cs_staromestske.m4a` | ~25s | CS | Czech Old Town Square story | CS source |

**Backup:** `~/Documents/STU/BP-voice-backup/speaker_voices/` — identical copy.

### 4.2 Piper Models (`backend/tts/piper_models/`) — gitignored

| Model | Language | Purpose |
|-------|----------|---------|
| `sk_SK-personal-male-medium.onnx` | SK | Owner's cloned SK voice (default SK output) |
| `en_US-personal-v2.onnx` | EN | Owner's cloned EN voice |
| `sk_SK-lili-medium.onnx` | SK | Generic SK fallback (CC0) |
| `en_US-john-medium.onnx` | EN | Generic EN fallback (public domain) |
| `en_US-kristin-medium.onnx` | EN | Generic EN (CC0) |
| `en_US-norman-medium.onnx` | EN | Generic EN |
| `en_US-ryan-medium.onnx` | EN | **Non-commercial** (CC BY-NC-SA) — not shipped |
| `cs_CZ-jirka-medium.onnx` | CS | Generic CS (CC0) |

### 4.3 CT2 Models (`ct2_models/`) — gitignored

| Model | Purpose |
|-------|---------|
| `whisper-base/` | EN STT |
| `whisper-small-sk/` | SK STT (fine-tuned, flawed — Czech hallucinations) |
| `Helsinki-NLP--opus-mt-en-sk/` | EN→SK MT |
| `Helsinki-NLP--opus-mt-sk-en/` | SK→EN MT |
| `Helsinki-NLP--opus-mt-en-cs/` | EN→CS MT |
| `Helsinki-NLP--opus-mt-cs-en/` | CS→EN MT |
| `Helsinki-NLP--opus-mt-en-hi/` | EN→HI MT |
| `Helsinki-NLP--opus-mt-hi-en/` | HI→EN MT |
| `Helsinki-NLP--opus-mt-en-pl/` | EN→PL MT |
| `Helsinki-NLP--opus-mt-pl-en/` | PL→EN MT |
| `Helsinki-NLP--opus-mt-en-zh/` | EN→ZH MT |
| `Helsinki-NLP--opus-mt-zh-en/` | ZH→EN MT |
| `Helsinki-NLP--opus-mt-fr-en/` | FR→EN MT |
| `Helsinki-NLP--opus-mt-ja-en/` | JA→EN MT |

### 4.4 Language Packs (`languages/packs/`)

| Pack | Direction | STT Model | TTS Voice | Status |
|------|-----------|-----------|-----------|--------|
| `en-sk.json` | EN→SK | base | sk_SK-personal-male | **Reference** |
| `sk-en.json` | SK→EN | small-sk | en_US-john-medium | **Bottleneck** (SK STT flawed) |
| `en-cs.json` | EN→CS | base | — | Evaluated |
| `cs-en.json` | CS→EN | base | — | Evaluated |
| `en-de.json` | EN→DE | base | — | Declared |
| `en-hi.json` | EN→HI | base | — | Declared |
| `en-pl.json` | EN→PL | base | — | Declared |
| `en-zh.json` | EN→ZH | base | — | Declared |
| `fr-en.json` | FR→EN | base | — | Declared |
| `hi-en.json` | HI→EN | base | — | Declared |
| `ja-en.json` | JA→EN | base | — | Declared |
| `pl-en.json` | PL→EN | base | — | Declared |
| `zh-en.json` | ZH→EN | base | — | Declared |

---

## 5. Scripts

| File | Purpose | Status |
|------|---------|--------|
| `scripts/demo_audio.py` | Offline clip through pipeline (demo rehearsal) | **Active** |
| `scripts/e2e_ensk_new_voice.py` | Full EN→SK e2e with timing | **Active** |
| `scripts/demo_preflight.py` | Demo pre-flight check | **Active** |
| `scripts/update_voice_lab_library.py` | Regenerate Voice Lab library.json | **Active** |
| `scripts/language_packs.py` | Load/resolve language packs | **Active** |
| `scripts/convert_models.py` | Convert HF models to CT2 | **Active** |
| `scripts/bp.py` | CLI (corpus/qc/stt/e2e/library/script) | **Active** |
| `scripts/setup.py` | One-command setup | **Active** |
| `scripts/vad_close_probe.py` | VAD trailing-silence probe | **Active** |
| `scripts/live/paced_replay.py` | **v2**: 1x paced full-route client, delay metrics | **Active** |
| `scripts/live/summarize_replay.py` | **v2**: Summarize paced_replay JSON | **Active** |
| `scripts/live/interrupt_smoke_test.py` | Live WS rehearsal (needs running server) | **Active** |
| `scripts/gpu_bench/kaggle_full_route/` | **v2**: Kaggle T4 full-route test (PUSHED, RUNNING) | **Active** |
| `scripts/gpu_bench/kaggle_route/` | Previous Kaggle route test (COMPLETE) | **Historical** |
| `scripts/gpu_bench/kaggle_session/` | Previous Kaggle session test (COMPLETE) | **Historical** |
| `scripts/gpu_bench/kaggle_canary/` | Previous Canary validation | **Historical** |
| `scripts/archive/` | Old benchmarks (SSBD, full_pipeline, charts) | **STALE** |
| `scripts/model_harness/` | Model harness (unclear purpose) | **Review** |

---

## 6. Documentation

### 6.1 Current (v2, this session)

| File | Purpose |
|------|---------|
| `ARCHITECTURE_V2.md` | **v2**: Full architecture, deployment modes, LHC fit |
| `STREAMING_SPEC.md` | **v2**: Streaming STT→MT→TTS spec, latency budget |
| `MULTI_SPEAKER.md` | **v2**: Multi-speaker meeting design, API endpoints |
| `PROJECT_INDEX.md` | **v2**: This file — complete asset inventory |

### 6.2 Historical (dated, reference only)

| File | Date | Status |
|------|------|--------|
| `HANDOFF_NEXT_SESSION.md` | 2026-11-04 | Historical |
| `HANDOFF_2026-11-04_session2_spec_plan.md` | 2026-11-04 | Historical |
| `PLAN_2026-11-03.md` | 2026-11-03 | Historical |
| `PROJECT_MATRIX_2026-11-04.md` | 2026-11-04 | Historical |
| `REVIEW_AND_UPDATE_PLAN_2026-11-04.md` | 2026-11-04 | Historical |
| `TEMPORARY_NOTES_2026-11-03.md` | 2026-11-03 | Historical |
| `TEMP_NAVIGATION_MAP.md` | 2026-11-04 | Historical |
| `backend-fixes-2026-11-04.md` | 2026-11-04 | Historical |
| `documentation_audit_2026-11-03.md` | 2026-11-03 | Historical |
| `review-design-evidence-2026-11-04.md` | 2026-11-04 | Historical |
| `review-product-2026-11-04.md` | 2026-11-04 | Historical |
| `research_findings_2026-11-03.md` | 2026-11-03 | Historical |
| `kaggle_validation_2026-11-03.md` | 2026-11-03 | Historical |
| `claims.md` | 2026-11-04 | Historical |
| `licences.md` | 2026-11-04 | Historical |
| `presentation_runbook_2026-11-04.md` | 2026-11-04 | Historical |

### 6.3 Implementation results (this session's repair batch)

| Directory | Purpose |
|-----------|---------|
| `implementation-2026-11-04/` | Repair results, Kaggle receipts, roast, designer output |

### 6.4 Older docs (2026-09 to 2026-10)

| File | Purpose |
|------|---------|
| `demo_runbook_2026-09-28.md` | Demo runbook |
| `monday_test_checklist.md` | Demo checklist |
| `evaluation_2026-10-18.md` | Evaluation |
| `load_tests_2026-10.md` | Load tests |
| `linux_setup_and_test.md` | Linux test sheet |
| `setup_friction_2026-10.md` | Setup friction report |
| `footprint_audit_2026-09-28.md` | Footprint audit |
| `voice_and_app_direction_2026-09.md` | Voice/direction |
| `personal_voice_bootstrap_2026-08-19.md` | Voice bootstrap |
| `piper_training_runbook.md` | Piper training |
| `recording_session_2026-09-28.md` | Recording session |
| `reading_script_bilingual.md` | Bilingual reading script |
| `verification_omni_piper_sk_2026-09-30.md` | Omni/Piper SK verification |
| `model_landscape_2026-10.md` | Model landscape |
| `s2s_landscape_2026-10-18.md` | S2S landscape |
| `compute_capacity_report_2026-09.md` | Compute capacity |
| `concurrency_accuracy_2026-08-21.md` | Concurrency accuracy |
| `project_overview.md` | Project overview |
| `project_status_2026-10-20.md` | Project status |
| `progress_since_the_break.md` | Progress report |
| `thesis_draft.md` | Thesis draft |
| `cli_quickref.md` | CLI quickref |
| `ci.yml.example` | CI example |
| `supabase_auth_db_guide.md` | Auth guide |
| `teams_auth_ticket_2026-10-14.md` | Teams auth ticket |
| `VAD_TUNING_GUIDE.md` | VAD tuning |
| `security_audit_2026-09.md` | Security audit |
| `ROAST_REPORT_2026-10-21.md` | Roast report |
| `INDEX.md` | Documentation index |

---

## 7. Kaggle Assets

| Asset | ID | Status |
|-------|-----|--------|
| `bp-gpu-bench-refs` | Reference datasets (en.wav, sk.wav, cs.wav) | Private, exists |
| `bp-sk-audio-20261105` | SK test audio (me_sk_script.m4a, me_sk_trhove.m4a) | **Just created** |
| `bp-full-route-code-v2` | v2 code package (app.py, backend/, languages/, scripts/) | **Just created** |
| `bp-full-stt-mt-tts-streaming-route-t4` | Kernel — **RUNNING on T4** | **Just pushed** |
| `bp-route-paced-t4-20261104` | Previous route test | COMPLETE |
| `bp-omnivoice-bounded-session-replay-20261104` | Previous TTS test | COMPLETE |
| `bp-canary-sk-cs-validation` | Previous Canary test | COMPLETE |

---

## 8. Root-Level Files

| File | Purpose | Status |
|------|---------|--------|
| `AGENTS.md` | Project memory for agents | **Active** (updated with v2) |
| `PRODUCT.md` | Product contract | **Active** |
| `PLAN.md` | Dev plan | **Active** |
| `DESIGN.md` | UI contract | **Active** |
| `README.md` | Project README | **Active** |
| `NOTICE.md` | Licence notice | **Active** |
| `LICENSE` | MIT licence | **Active** |
| `UPDATES.md` | Update log | **Active** |
| `VAD_TUNING_GUIDE.md` | VAD tuning guide | **Active** |
| `sql_app.db` | SQLite database (gitignored) | **Active** |

---

## 9. Stale / Bloat Summary (Candidates for Removal)

| Category | Files | Action |
|----------|-------|--------|
| Stale backend modules | `backend/database.py`, `backend/hardware_detect.py` | Remove (zero imports) |
| Duplicate hardware detection | `backend/hardware.py` | Deprecate in favor of `hardware_scan.py` |
| Unused MT engines | `backend/mt/nllb_mt.py`, `backend/mt/hybrid_mt.py` | Keep declared in registry, but not in live path |
| Archive tests | `test/archive/` | Move to archive or remove |
| Unused test files | `test/concurrent_performance_test.py`, `test/i18n_dict_test.py`, `test/model_testing_framework.py`, `test/voice_similarity.py` | Review for removal |
| Archive scripts | `scripts/archive/` | Move to archive or remove |
| Historical docs | `documentation/*2026-10*`, `documentation/*2026-09*` | Keep as dated reference, mark clearly |
| Kaggle prior output | `documentation/implementation-2026-11-04/kaggle-prior-output/` | Gitignored, keep locally |
| Kaggle route/session private | `documentation/implementation-2026-11-04/kaggle-*-private/` | Gitignored, keep locally |

