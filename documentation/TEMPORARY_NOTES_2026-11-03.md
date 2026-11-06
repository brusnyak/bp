# Temporary documentation — BP memory aids (2026-11-03)

**Purpose:** scratchpad so nothing lives only in chat memory. Append-only during this work session. Delete or fold into permanent docs when done.

## What's already done (don't redo)
- `bp language` subcommand wired into `scripts/bp.py` (lines 573-605) ✅
- `bp selftest` at end of `setup.py` (line 92-93) ✅
- `sk-en` already `"experimental"` (no eval record → `status()` returns experimental) ✅
- e2e path already `me_en_script.m4a` ✅
- Per-pack `beam_size` wired in `backend/main.py` lines 543-546 ✅
- `make test` 38/38 pass; `bp selftest` PASSED ✅
- Root `README.md` rewritten ✅
- `documentation/README.md` rewritten ✅
- `documentation/INDEX.md` updated ✅
- `documentation/research_findings_2026-11-03.md` created ✅
- `documentation/documentation_audit_2026-11-03.md` created ✅
- `documentation/PLAN_2026-11-03.md` created (this session's working plan) ✅
- `backend/registry.py` GPU engine skeletons added (Canary, Kokoro, Qwen3-TTS, Chatterbox, SeamlessM4T v2) ✅

## Still to do (Phase 1.8 — stale doc cleanup)
- `documentation/model_landscape_2026-10.md` — §10 stale (OmniVoice "current bulk generator", Kokoro "dead for SK"); add CZ rows; clarify Kokoro=EN TTS only; add Qwen3-TTS/Chatterbox rows; update Canary AST status
- `documentation/load_tests_2026-10.md` — OmniVoice-only; re-orient to Kokoro EN + Piper SK split
- `documentation/monday_test_checklist.md` — wrong clip filenames (`en_script_reading.m4a` → `me_en_script.m4a`); remove Hybrid/qc-venv; update baselines
- `documentation/demo_runbook_2026-09-28.md` — refs to deleted `SETUP_WINDOWS.md`; add CZ; update SK→EN honest gap
- `documentation/linux_setup_and_test.md` — refs to deleted Windows files; confirm lite flow only
- `documentation/evaluation_2026-10-18.md` — says 47/47 tests (now 38); OmniVoice refs; add CZ row
- `AGENTS.md` §8-9 — OmniVoice GPU path, load-test next steps, Kaggle/Modal cleanup — superseded
- `documentation/sdk.md` — add Kokoro/Chatterbox/Qwen3-TTS as optional GPU TTS engines
- `documentation/project_status_2026-10-20.md` — add CZ row; clarify OmniVoice=bulk only

## Still to do (Phase 2 — needs yes for Kaggle/compute)
- Build `scripts/model_harness/` — bounded 8 GB VRAM model test harness
- Run Canary AST on Kaggle T4 (SK→EN WER gate < 0.2)
- Ear-grade Kokoro EN TTS + Qwen3-TTS/Chatterbox EN cloning on T4
- Test additional language pairs (fr, ar, zh, hi, African)

## Still to do (Phase 3 — ops + release)
- `documentation/ops.md` — `/health` + `/metrics` documented
- Move live test scripts to `scripts/live/`
- Linux x86_64 + CI wiring
- Revoke Kaggle/Modal tokens
- Fresh-clone test + tag release + push (explicit yes)

## Key numbers (proven this session)
- Tests: 38/38 pass (`make test`)
- Selftest: PASSED (6 expected warnings)
- SK→EN WER clean: 0.2143 (Kaggle v6 T4) — above 0.2 gate
- SK→EN WER mic: ~0.32 (beam=5)
- EN→SK E2E: ~0.78–1.3s
- SK→EN E2E: ~1.27–1.64s (STT-bound)
- Kokoro EN TTS: RTF 0.03, 1.33 GB VRAM on T4
- Piper SK TTS: RTF ~0.04, CPU
- CZ: Whisper native, BuzzASR WER ~12%, Opus-MT chrF ~0.72, Piper jirka present
