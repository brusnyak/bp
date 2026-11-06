# Documentation Audit + Scope Reconsideration (2026-11-03)

**Purpose:** review every documentation file, mark it current/stale/partially-stale,
and decide what must change to reach "student code" state — current, accurate, usable by
a zero-knowledge person checking out the repo and getting it running.

**Context:** Slovak + Czech are mutually intelligible West Slavic. Czech is essentially
free to add (Whisper supports CZ natively; BuzzASR/czech WER ~12% vs SK ~21–32%;
Opus-MT CZ↔EN chrF ~0.72 vs SK↔EN ~0.66; Piper `cs_CZ-jirka-medium` already in repo;
SeamlessM4T v2 supports CZ S2S same as SK). This widens scope from EN↔SK to
EN↔{SK,CZ} with a pluggable architecture, and strengthens the "any language" claim.

---

## A. Czech language research (new)

| Stage | Czech support | Quality | Repo status |
|-------|--------------|---------|-------------|
| STT | Whisper large-v3 native; BuzzASR/czech fine-tune WER ~12% (FLEURS/CV) | Good | Not wired as pack |
| MT | Opus-MT CZ↔EN chrF ~0.72 (tatoeba), ~0.66 (flores); CZ↔SK pair exists | Stronger than SK | `opus-mt-cs-en` exists on HF, not converted locally |
| TTS | Piper `cs_CZ-jirka-medium` (medium) | Medium | ✅ Already in `backend/tts/piper_models/` |
| S2S | SeamlessM4T v2: Czech (ces) Sp,Tx,Sp,Tx — full S2S | Reference | Not in pipeline |

**SK-CZ similarity:** CZ data is more abundant than SK; models transfer between them.
Adding CZ is low-effort, high-value, de-risks the "any language" claim, and gives the
handler a working two-direction demo (SK + CZ).

---

## B. Documentation audit — every file

| File | State | Verdict | What needs to change |
|------|-------|---------|----------------------|
| `README.md` (root) | **Stale** | Names retired SeamlessM4T/F5-TTS as live; manual `pip install` flow; non-existent scripts (`download_piper_models.py`, `convert_opus_mt_to_ct2.py`); wrong test files; no `setup.py --dev`, `bp` CLI, language packs, health/metrics, cross-platform. | **Rewrite** as accurate clone→setup→run→use walkthrough matching actual stack. |
| `documentation/README.md` | **Current** | ✅ Just rewritten — accurate stack, quick start, SK→EN warning, voice-cloning limits, enterprise comparison. | Done. |
| `documentation/INDEX.md` | **Current** | ✅ Just updated — marks README.md current. | Done. |
| `documentation/research_findings_2026-11-03.md` | **Current** | ✅ Just created — enterprise S2S, voice cloning, Qwen/Canary/Parakeet, Slovak relevance, T4 VRAM fit. | Done. |
| `documentation/model_landscape_2026-10.md` | **Stale** | §10 calls OmniVoice "current bulk generator" (pivot drops fine-tuning focus); Kokoro "dead for SK" but pivot wants Kokoro as EN TTS; missing Qwen3-TTS/Chatterbox rows; no CZ rows; Canary AST marked UNKNOWN (untested). | Add CZ rows (STT/MT/TTS/S2S); clarify Kokoro = EN TTS only; add Qwen3-TTS/Chatterbox rows; update Canary AST status pending Kaggle. |
| `documentation/load_tests_2026-10.md` | **Stale** | OmniVoice-only cascade load tests; no Kokoro/Piper split; references `processed/gpu_bench/load_T4/`. | Re-orient to Kokoro EN + Piper SK split; add T4 numbers per engine. |
| `documentation/linux_setup_and_test.md` | **Partially stale** | Run 1 results valid (250s setup, 33 tests, all=cpu). References old manual flow and `SETUP_WINDOWS.md` (deleted). | Remove references to deleted Windows files; confirm lite flow is the only path. |
| `documentation/demo_runbook_2026-09-28.md` | **Partially stale** | Core demo flow valid. References `SETUP_WINDOWS.md` (deleted), Windows AMD numbers, OmniVoice clone. | Remove deleted-file refs; add CZ demo direction; update SK→EN honest gap (WER 0.2143). |
| `documentation/monday_test_checklist.md` | **Stale** | References `en_script_reading.m4a`/`sk_script_reading.m4a` (actual: `me_en_script.m4a`/`me_sk_script.m4a`); references Hybrid voice + `scores.json`/`qc-venv` (obsolete); SK STT baseline 0.63 (now 0.2143 clean / ~0.32 mic). | Fix clip filenames; remove Hybrid/qc-venv; update baselines; add CZ row. |
| `documentation/project_status_2026-10-20.md` | **Partially stale** | SK→EN marked experimental (good). Mentions OmniVoice as bulk generator. No CZ row. | Add CZ row (verified); clarify OmniVoice = bulk/generator only, not live path. |
| `documentation/sdk.md` | **Mostly current** | Matches `pipeline.py`, `registry.py`, `/health`+`/metrics`, WS protocol. TTS engine list missing Kokoro/Chatterbox/Qwen3-TTS. | Add Kokoro/Chatterbox/Qwen3-TTS as optional GPU TTS engines. |
| `documentation/ci.yml.example` | **Current** | Matches `setup.py --dev --no-npm`. | None. |
| `documentation/evaluation_2026-10-18.md` | **Partially stale** | Says 47/47 tests (now 38); mentions OmniVoice cached wavs + removed `.venv-omni` env. | Update test count; remove OmniVoice refs; add CZ eval row. |
| `documentation/setup_friction_2026-10.md` | **Current** | Measured friction, still valid (9 min clone→green). | None. |
| `documentation/linux_setup_and_test.md` | **Partially stale** | See above. | See above. |
| `documentation/s2s_landscape_2026-10-18.md` | **Current** | Good landscape; correctly notes Teams/Zoom no SK, Gemini/DeepL yes SK. | Could add CZ row to the Slovak/CZ comparison. |
| `documentation/compute_capacity_report_2026-09.md` | **Current** | Measured numbers, projections labelled. Valid. | Could add CZ fine-tune projection. |
| `documentation/verification_omni_piper_sk_2026-09-30.md` | Historical | Voice kill-graded (trembling), numbers stand. | None (historical record). |
| `documentation/piper_training_runbook.md` | ? | Not yet read. | Read and assess. |
| `documentation/security_audit_2026-09.md` | ? | Not yet read. | Read and assess. |
| `documentation/thesis_draft.md` | Has `[DOPLNIŤ]` | One marker left; AI-declaration skeleton. | Fill marker; update SK→EN honest numbers. |
| `documentation/demo_report_2026-09.md` | Historical | Measured demo numbers (2026-09). | None (historical record). |
| `documentation/progress_since_the_break.md` | Historical | Narrative progress log. | None. |
| `documentation/concurrency_accuracy_2026-08-21.md` | Old | Pre-lite era. | Archive or mark historical. |
| `documentation/footprint_audit_2026-09-28.md` | Historical | Ranked cut list; open items (delete legacy venvs). | Check if open items done. |
| `documentation/teams_auth_ticket_2026-10-14.md` | Historical | Auth ticket. | None. |
| `documentation/personal_voice_bootstrap_2026-08-19.md` | Historical | Voice bootstrap. | None. |
| `documentation/fundamentals.md` / `fundamentals_deep.md` | Theory chapters | Fundamentals §9 connectors, §10 ground basics. | None (thesis theory). |
| `AGENTS.md` §8–9 | **Stale** | OmniVoice GPU path, load-test next steps, Kaggle/Modal cleanup — superseded by pivot. | Update to GPU-oriented model harness + CZ direction. |
| `PLAN.md` | Current | Live dev plan. Next steps need re-scoping for CZ + GPU harness. | Add CZ direction + model harness phase. |
| `DESIGN.md` | Current | UI contract. | None. |

**Files needing immediate work:** root `README.md`, `model_landscape_2026-10.md`, `load_tests_2026-10.md`, `monday_test_checklist.md`, `demo_runbook_2026-09-28.md`, `linux_setup_and_test.md`, `evaluation_2026-10-18.md`, `AGENTS.md` §8–9, `sdk.md` (minor), `project_status_2026-10-20.md` (minor).

---

## C. Scope reconsideration (SK-CZ similarity + GPU orientation)

### What changes

1. **EN↔SK → EN↔{SK, CZ}** as the initial verified pair set. CZ is low-effort (Whisper CZ native, Opus-MT CZ↔EN strong, Piper jirka voice present, SeamlessM4T v2 CZ S2S). This makes "any language" structurally credible: a language is "supported" when STT+MT+TTS exist and `bp language test <pair>` passes.

2. **GPU-first deployment target (T4 / ≤8 GB VRAM or better RTX), not CPU-only.**
   - Baseline still runs CPU-only (Piper + faster-whisper) for the torch-free Windows laptop.
   - Optional GPU engines behind the registry: Canary AST (SK→EN, collapses STT+MT), Kokoro (EN TTS, RTF 0.03, 1.33 GB), Qwen3-TTS / Chatterbox (EN voice cloning, 4–8 GB), SeamlessM4T v2 (S2S quality reference, CC-BY-NC, ~6 GB).
   - The model test harness (new `scripts/model_harness/`) bounds every candidate to 8 GB VRAM and reports WER/chrF/RTF/VRAM/latency on fixed SK/CZ/EN clips.

3. **"Your own voice" scope** (thesis already includes this): For SK output, no open ZS-TTS supports SK. Path = Piper fine-tune (CPU, works, low intelligibility per engine_ab) + optional voice-conversion layer (RVC/Applio, train once on GPU, apply to any TTS). For EN output, ZS cloning works (Chatterbox/Qwen3-TTS/XTTS on GPU).

4. **Commercial-readiness for zero-knowledge users:** accurate docs (root README rewrite), one-command setup, clear verified-vs-experimental status, no stale claims, `/health`+`/metrics` documented, `bp language` CLI wired, `selftest` in `setup.py`.

### What stays the same
- Cascade (STT→MT→TTS) is the thesis contribution — inspectable, measurable per stage, CPU-runnable.
- Local-first, no cloud in pipeline; cloud only as QC reference.
- Fail-closed licence gate (non-commercial engines blocked by default).
- FastAPI + WebSocket + plain HTML/CSS/JS UI.

### Phased plan (revised)

**Phase 1 — Foundation (no compute, local edits):**
1. Rewrite root `README.md` (accurate onboarding).
2. Flip `sk-en` to `"experimental"`; require WER < 0.3 for `"verified"` in `language_packs.py`.
3. Add CZ language pack (`cs_CZ-jirka-medium` TTS + Opus-MT cs-en) + `bp language test cs` passes.
4. Wire `bp language` subcommand into `scripts/bp.py`; `bp selftest` at end of `setup.py`; fix e2e path (`me_en_script.m4a`); per-pack `beam_size`.
5. Register Canary AST + Kokoro as optional GPU engine skeletons in `registry.py` (fail-closed, declared not implemented).
6. Clean stale docs: `model_landscape_2026-10.md`, `load_tests_2026-10.md`, `monday_test_checklist.md`, `demo_runbook_2026-09-28.md`, `linux_setup_and_test.md`, `evaluation_2026-10-18.md`, `AGENTS.md` §8–9.

**Phase 2 — Model test harness (GPU validation on Kaggle T4, 1 day):**
7. Build `scripts/model_harness/` — one CLI: `bp bench --stage stt|mt|tts|s2s --model <name> --device cuda --vram-limit 8`, bounded to 8 GB VRAM, fixed SK/CZ/EN clips, outputs WER/chrF/RTF/VRAM/latency + ear-grade wavs.
8. Run Canary AST on SK clips; if WER < 0.2 on real mic → promote to optional GPU engine; else accept SK→EN experimental-only.
9. Ear-grade Kokoro EN TTS on T4.

**Phase 3 — Ops + release:**
10. `documentation/ops.md` for `/health`+`/metrics`; `Pipeline` facade already exists (`backend/pipeline.py`); `documentation/sdk.md` minor update.
11. Move live test scripts to `scripts/live/`; Linux x86_64 + CI; revoke Kaggle/Modal tokens; fresh-clone test; tag release (your yes before push).

---

## D. Questions before I execute

1. **Proceed with Phase 1 (items 1–6)?** All local edits, reversible, no compute. Highest leverage: root README rewrite + `sk-en`→experimental + CZ pack.
2. **CZ as verified second direction?** Ship CZ alongside SK in the demo/runbook/checklist.
3. **Model harness as new `scripts/model_harness/` directory?** Or extend existing `bench/` + `scripts/gpu_bench/`?
4. **Canary/Kokoro as optional GPU skeletons now, or wait for Kaggle result first?**
5. **Thesis title stays as-is** (approved by faculty); "own voice" scope already in thesis outline — confirmed no title change.

Constitution reminder: I won't commit or push without your explicit yes. All changes stay uncommitted for review.

🟢 Tests 38/38 pass; selftest PASSED; research complete; documentation audit complete; scope reconsidered for CZ + GPU.
