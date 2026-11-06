# BP — Documentation Navigation Map (temporary, 2026-11-04)

> Current repair state (2026-11-04): start with `PRODUCT.md`, `documentation/implementation-2026-11-04/RESULTS.md` and `documentation/HANDOFF_NEXT_SESSION.md`. Earlier dated counts, readiness claims, media-access behavior and Kaggle failure explanations below are historical. Current contracts require authenticated explicit preparation, offline assets, target-language TTS, terminal Stop and reviewed full-route evidence. Full cloned-speech/LHC/commercial acceptance remains pending.

**Purpose:** single index so scope, state, and next steps live in files, not chat
memory. Constitution: `.specify/memory/constitution.md` (v1.0.0, ratified 2026-08-19).

---

## Core principles (constitution)

| # | Principle | Meaning for this project |
|---|-----------|--------------------------|
| I | Cross-Platform, No Exceptions | Required-path deps run on Win/mac/Linux; per-stage runtime hardware chain `cuda→rocm→directml/mps/coreml→cpu` (`backend/hardware.py`) |
| II | Evidence-First | Every perf/latency/memory/capability claim backed by a measured record; settled decisions not re-litigated without new evidence |
| III | Slovak Non-Negotiable | EN↔SK is a hard requirement; anything without SK support is out of scope for core pipeline |
| IV | Real-Time Is the Product | Near-real-time on commodity hardware (M1 Pro 16GB baseline, no GPU assumed) |
| V | Plug-and-Play | Install once, select virtual mic once, it works; OS virtual-audio hidden behind automated setup |
| VI | Simplicity Over Novelty | Adapt existing components (faster-whisper, CTranslate2/Opus-MT, Piper, XTTS v2); no from-scratch training unless evidence forces it |

---

## Where each source of truth lives

| Topic | File | State |
|-------|------|-------|
| Constitution | `.specify/memory/constitution.md` | Current v1.0.0 |
| Project memory / rules | `AGENTS.md` | Current — read before changing anything |
| Dev plan (next steps, run commands) | `PLAN.md` | Current |
| Working plan / phased scope / commercial positioning | `documentation/PLAN_2026-11-03.md` | Current — Phase 1 done, Phase 1.8 (doc cleanup) next |
| UI contract | `DESIGN.md` | Current |
| Spec 001 (cross-platform pipeline) | `specs/001-realtime-cross-platform-translation/` | Current |
| Spec 002 (PWA/PiP demo) | `specs/002-pwa-pip-demo/` | Current |
| Spec 003 (end-to-end completion) | `specs/003-end-to-end-completion/` | Current |
| Language packs (declarative STT/MT/TTS per pair) | `languages/packs/*.json` | 13 packs; cs-en + sk-en need review |
| Engine registry (select/instantiate, fail-closed licence gate) | `backend/registry.py` | Current |
| Hardware backend detection | `backend/hardware.py` | Current — per-stage chains |
| FastAPI + WebSocket server | `backend/main.py` | Current |
| Pipeline facade | `backend/pipeline.py` | Current |
| bp CLI | `scripts/bp.py` | Current — `language|engine|stt|e2e|selftest|bench|grades` |
| Setup (one-command) | `scripts/setup.py` | Current — ends with `bp selftest` |
| Test suite | `test/` | 38 tests pass |
| Kaggle results (collected) | `/Users/yegor/bp-benchmark-2026-11-04/confirm-canary-v1/`, `confirm-nemo-v1/` | COMPLETE — 14 canary + 2 nemo cases, n=80 each |
| Live speech UI | `ui/live-speech/live.html` | Current — virtual mic + output device select EXISTS |
| Voice Lab | `ui/voice-lab/lab.html` | Current |
| GitHub Pages handler page | `bp-report-pages-2026-11-03/docs/index.html` | To update LAST |

---

## Files needing cleanup / consolidation (Phase 1.8)

These are stale or partially stale per `documentation_audit_2026-11-03.md`:

**ALL CLEANED (this session, 2026-11-04):**
1. ✅ `documentation/model_landscape_2026-10.md` — Canary AST now MEASURED (chrF++ ~52, below commercial bar); OmniVoice = batch/QC only; X-Voice and OmniVoice GPU now measured
2. ✅ `documentation/load_tests_2026-10.md` — Added Kokoro/Piper split; OmniVoice = batch/QC only
3. ✅ `documentation/monday_test_checklist.md` — Added CZ accuracy row; updated SK→EN with Canary results
4. ✅ `documentation/demo_runbook_2026-09-28.md` — Removed `SETUP_WINDOWS.md` ref; updated honest gap with Canary results
5. ✅ `documentation/linux_setup_and_test.md` — Removed deleted Windows file references
6. ✅ `documentation/evaluation_2026-10-18.md` — sk-en marked experimental; OmniVoice = batch/QC only
7. ✅ `AGENTS.md` §8–9 — Canary AST Kaggle done (not promoted); OmniVoice job submitted; load test split clarified
8. ✅ `documentation/sdk.md` — Already correct (omnivoice 🔒 unknown, needs GPU)
9. ✅ `documentation/project_status_2026-10-20.md` — CZ row fixed; OmniVoice clarified; Canary results added
10. ✅ `thesis_draft.md` — No `[DOPLNIŤ]` marker found; clean

**Deleted already (this session):** `documentation/handler_update_2026-09.md` (replaced by GitHub Pages `docs/index.html`)

---

## Language packs — current content (needs review for CZ-primary pivot)

| Pack | STT | MT | TTS voice | Notes |
|------|-----|-----|-----------|-------|
| `cs-en` | `small-sk`, `language=cs` (Czech-as-proxy) | `opus-mt-cs-en` (NOT converted yet) | `en_US-ryan-medium` | MT direction not converted; SK-tuned model used as Czech proxy |
| `sk-en` | `small-sk`, `language=sk` | `opus-mt-sk-en` (double-dash model id) | `en_US-ryan-medium` | Reference pair; SK STT is latency bottleneck |
| `en-cs` | `base`, `language=en` | `opus-mt-en-cs` | `cs_CZ-jirka-medium` | First non-reference pair; all components present |

---

## What's proven vs open (evidence, from collected Kaggle results)

**Proven (measured, n=80 each, confirm-canary-v1/ + confirm-nemo-v1/):**
- Canary-1B ASR SK beam=5: WER 0.0477 (< 0.2 gate ✅)
- Canary-1B ASR CS beam=5: WER 0.0922 (< 0.2 gate ✅)
- Canary-1B ASR EN beam=5: WER 0.0498
- Parakeet SK STT: WER 0.0690
- Canary-1B AST SK→EN beam=5: chrF 0.2401 (NO English reference → NOT verified quality)
- Canary-1B AST CS→EN beam=5: chrF 0.2373 (NO English reference → NOT verified quality)

**Open (NOT proven):**
- SK→EN / CZ→EN translation quality (no English reference; chrF ~0.24 is not a quality gate pass)
- OmniVoice on live speech / long sentences / presentation pipeline (only batch RTF measured: 0.40–0.77 short phrases on T4 — not real-time)
- No Slovak zero-shot voice cloning exists (XTTS/Chatterbox/Qwen3-TTS/Kokoro all exclude SK)

**Already exists (I was wrong to claim otherwise):**
- Virtual mic / output device selection: `ui/live-speech/live.html` lines 158-208
- Per-stage hardware auto-detection: `backend/hardware.py` `detect_backend()`

---

## Next steps in order (follow PLAN.md + PLAN_2026-11-03.md, single source of truth)

1. Run Kaggle tests for SK→EN and CZ→EN translation quality with English reference review
2. Test OmniVoice on live speech / long sentences / presentation pipeline on Kaggle
3. Pivot to CZ as primary (cs-en pack fixes: convert MT, fix STT proxy, fix TTS voice)
4. Doc cleanup Phase 1.8 (stale files above)
5. GitHub Pages handler page update (`bp-report-pages-2026-11-03/docs/index.html`) — LAST
6. Roast/critic/review/test/configure → commercially ready
