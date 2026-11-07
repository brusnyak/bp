# BP — Handoff for Next Session (2026-11-04)

## Where things stand

**Handler file removed.** `documentation/handler_update_2026-09.md` deleted. Replaced by GitHub Pages page at `bp-report-pages-2026-11-03/docs/index.html` (self-contained HTML with inline model data, filters, charts, evidence index). Zero remaining `handler_update`/`handler-report` references in main repo (verified via grep).

**Commercial readiness: NOT ready.** README now states this explicitly with a gap table:
- SK→EN translation: chrF++ ~52 mean (measured n=80) — below commercial bar (~70+), not promoted to live route
- CZ→EN translation: chrF++ ~50 mean (measured n=80) — below commercial bar
- SK zero-shot voice cloning: no open model exists; Piper fine-tune only (CPU, RTF ~0.04)
- OmniVoice: batch/QC only (RTF 0.40–0.65 short phrases on T4), NOT live path
- Virtual mic auto-setup per OS exists in UI but needs per-OS testing
- No "suggest best setup" output yet from hardware detection

**Proven (measured):**
- Canary-1B ASR: SK WER 0.0477 (<0.2 ✅), CS 0.0922, EN 0.0498 (n=80 each, Kaggle T4)
- Canary AST SK→EN chrF++ mean 52.49, CS→EN 50.05 (n=80 each, English references present but flagged `semantic_review_pending`)
- Beam size 1 vs 5: no meaningful difference on Canary AST
- EN→SK verified (~0.7s turn), EN→CZ verified (~0.8s turn), live on CPU

**Not proven:**
- SK→EN / CZ→EN translation quality (no verified English reference; chrF++ ~50 is not a quality pass)
- OmniVoice on live microphone speech / long sentences / presentation pipeline (only batch RTF measured)
- Kokoro EN + Piper SK load test on T4 (pending)

## Files changed this session

**Modified:**
- `README.md` — commercial-readiness gap table, cs-en experimental, SK→EN Canary chrF++ note, OmniVoice=batch/QC only, honest assessment section
- `AGENTS.md` §8-9 — Canary AST Kaggle done (not promoted), OmniVoice job submitted, load split clarified
- `documentation/archive/2026-11-07-stale/PLAN.md` — removed handler_update reference from handler deliverable section
- `documentation/INDEX.md` — removed handler_update row
- `documentation/model_landscape_2026-10.md` — Canary AST measured, OmniVoice=batch/QC only, X-Voice/OmniVoice GPU measured
- `documentation/load_tests_2026-10.md` — Kokoro/Piper split added, OmniVoice=batch/QC only
- `documentation/archive/2026-11-07-stale/monday_test_checklist.md` — CZ accuracy row added, SK→EN updated with Canary results
- `documentation/archive/2026-11-07-stale/demo_runbook_2026-09-28.md` — removed `SETUP_WINDOWS.md` ref, updated honest gap with Canary results
- `documentation/linux_setup_and_test.md` — removed deleted Windows file references
- `documentation/evaluation_2026-10-18.md` — sk-en experimental, OmniVoice=batch/QC only
- `documentation/project_status_2026-10-20.md` — CZ row fixed, OmniVoice clarified, Canary results added
- `languages/packs/cs-en.json` — TTS voice `en_US-ryan-medium` → `cs_CZ-jirka-medium`, updated notes
- `scripts/convert_models.py` — added `cs-en` to PAIRS list

**Deleted:**
- `documentation/handler_update_2026-09.md`

**Created:**
- `documentation/TEMP_NAVIGATION_MAP.md` — single index of all sources of truth
- `documentation/archive/2026-11-07-stale/HANDOFF_NEXT_SESSION.md` (this file)

**Pre-existing uncommitted changes (NOT mine — do not touch unless you know what they are):**
- `app.py`, `backend/main.py`, `backend/registry.py`, `reports/index-report.html`, `scripts/bp.py`, `scripts/live/interrupt_smoke_test.py`, `test/config_tests.py`, `test/vad_tests.py`
- Untracked: `UPDATES.md`, `bench/`, `documentation/PLAN_2026-11-03.md`, `documentation/TEMPORARY_NOTES_2026-11-03.md`, `documentation/documentation_audit_2026-11-03.md`, `documentation/handoff-2026-11-03-pipeline-pivot.md`, `documentation/kaggle_validation_2026-11-03.md`, `documentation/research_findings_2026-11-03.md`, `documentation/evidence/`, `scripts/gpu_bench/kaggle_canary/`, `scripts/model_harness/`, `test/language_evidence_test.py`

## Kaggle status

**COMPLETE:** `yegorby/bp-confirm-canary-matrix-20261104` v1, `yegorby/bp-confirm-nemo-matrix-20261104` v1. Outputs collected to `/Users/yegor/bp-benchmark-2026-11-04/confirm-canary-v1/` + `confirm-nemo-v1/`.

**ERROR:** `yegorby/bp-omnivoice-matrix-20261104` v2 — OmniVoice long-speech test failing on Kaggle Python 3.13 (dependency conflicts in `ensure_omnivoice()` / `omnivoice_eval.py`). Model weights load fine (100%) but `create_voice_clone_prompt`/`generate` fails. Need to debug `scripts/gpu_bench/kaggle_matrix/omnivoice_eval.py` — consider switching to the `kaggle_final.py` pattern (uses shipped `backend/tts/omni_tts.py` directly) or add the full `gpu_bench.py` PKGS dependency list. Two prior versions also errored (v1: direct GitHub clone approach, v2: pip install approach). Not a live-path blocker (OmniVoice = batch/QC only).

## CLI/MCP assessment

**MCP NOT relevant** for thesis or commercial scope. Project already has everything via CLI + Python API:
- `scripts/bp.py` — 15+ subcommands (language, engine, stt, e2e, selftest, bench, grades, library, qc, corpus, etc.)
- `backend/registry.py` — `select()`/`instantiate()`/`catalog()`, fail-closed licence gate
- `backend/pipeline.py` — `Pipeline` facade
- `backend/hardware.py` — per-stage backend chains
- `backend/hardware_detect.py` — `recommend_engines()` (OS + VRAM-based)
- `languages/packs/*.json` — declarative packs

An MCP server would add protocol maintenance without new capability. Skip for now.

## Next steps (in order)

1. **Debug/fix Kaggle OmniVoice job** (`omnivoice_eval.py`, Python 3.13 deps) OR accept OmniVoice = batch/QC only and move on. Not a live-path blocker.
2. **Run `scripts/convert_models.py`** (needs your yes — creates throwaway venv, installs torch+transformers ~1GB, converts `opus-mt-cs-en` to CTranslate2 int8). Required for cs-en MT to work locally. User previously said lightweight repo preferred (convert on load, not committed to git).
3. **Pivot to CZ primary** — cs-en pack fixed (TTS voice → Czech), MT conversion is the only blocker.
4. **Product/design UI review** — consider running `designer`, `painter`, or `frontend-design` skill on `ui/home/home.html`, `ui/live-speech/live.html`, `ui/voice-lab/lab.html`, `ui/enroll/enroll.html`, `ui/auth/auth.html`. Customer journey: discovery → website → repo → setup → live use.
5. **GitHub Pages handler page update** (LAST) — `bp-report-pages-2026-11-03/docs/index.html` already exists as self-contained HTML; may need updating with the honest commercial-readiness assessment from README.
6. **Doc cleanup verified complete** — all 9 stale files from Phase 1.8 cleaned (see navigation map).
7. **Thesis** — update with measured numbers (Canary chrF++ ~52 below commercial bar, OmniVoice batch/QC only), remove any remaining `[DOPLNIŤ]` markers (none found in thesis_draft.md).

## Roast (for next session)

**Weakest point:** Claiming "commercially ready" when SK→EN/CZ→EN translation quality is unverified (chrF++ ~50) and OmniVoice isn't live-tested on mic audio. **Cheapest failure:** A production claim that collapses on first live SK/CZ mic. **Smallest fix:** Keep explicit "experimental" gates in UI for SK→EN/CZ→EN; don't promote Canary to live route until chrF++ >70 with human-reviewed references; state "NOT commercially ready" prominently until quality gate passes.

## Context files to read next session

- `documentation/TEMP_NAVIGATION_MAP.md` — full index of sources of truth
- `AGENTS.md` — project memory (read before changing anything)
- `documentation/archive/2026-11-07-stale/PLAN.md` — live dev plan
- `documentation/PLAN_2026-11-03.md` — working plan, phased scope
- `.specify/memory/constitution.md` — v1.0.0, cross-platform / evidence-first / Slovak non-negotiable / real-time / plug-and-play / simplicity

## Machines

- Main: `/Users/yegor/Documents/STU/BP` (Mac M1 Pro 16GB, main branch)
- Kaggle results: `/Users/yegor/bp-benchmark-2026-11-04/`
- Kaggle matrix worktree: `/Users/yegor/bp-kaggle-matrix-2026-11-04/` (untracked `scripts/gpu_bench/kaggle_matrix/` — do not git clean/reset)
- ECC reference: `/Users/yegor/Developer/ecc-reference` (separate, read-only)

🟢 Handoff file written; all handler references removed; README updated with honest commercial-readiness; doc cleanup verified complete; Kaggle OmniVoice job ERRORING (non-blocking)