# Handoff: BP Pipeline Pivot — Next Session

**Source:** Roast: Pipeline Reconsideration + Kaggle v6 results (T4, 16GB, 278s)
**Project:** `~/Documents/STU/BP` (main = `origin/main` = `faf68ff`)
**Status:** Uncommitted changes exist (AGENTS.md, app.py, backend/main.py, registry.py, test/vad_tests.py, live-script.js, live.html, live-style.css)

---

## 1. Current State (Proven)

### What Works
- **EN→SK**: fully functional live pipeline over WebSocket with JWT auth
  - STT: whisper-base EN WER 0.0588 (clean clip), ~0.55s
  - MT: Opus-MT CT2 sub-150ms
  - TTS: Piper SK, ~0.17s
  - Total: ~0.78–1.3s end-to-end (live smoke test passed 2026-11-03)
- **Tests**: `make test` = 38/38 pass (Mac M1 Pro)
- **Setup**: one-command `python3.11 scripts/setup.py --dev` (~92s warm Mac)
- **Architecture**: FastAPI + WebSocket backend, plain HTML/CSS/JS UI, engine registry (`backend/registry.py`)

### What Is Broken
- **SK→EN STT**: fundamentally broken with open-source Whisper
  - `kinit/whisper-large-v3-turbo-sk` (best Slovak fine-tune, MIT, CV24 WER 9.29%): **WER 0.2143 on clean clip** (Kaggle v6)
  - `whisper-small-sk` (incumbent): **WER 0.2143 on clean clip** (identical — suspicious)
  - `whisper-large-v3` base: **WER 0.3636**
  - All above 0.2 quality gate; real mic will be worse
- **TTS**: no verified Slovak voice on GPU; Piper fine-tune takes 8-9h on M1 Pro

---

## 2. Paid Cloud vs Local (Evidence, Oct 2026 Pricing Pages)

| Service | STT | Translation | TTS | Approx/hour |
|---------|-----|-------------|-----|-------------|
| Google Cloud STT V2 | $0.016/min | separate | separate | ~$1.00+ |
| Azure Speech | $1.00/hour | $2.50/hour (≤2 langs) | separate | ~$3.50+ |
| AWS Transcribe | $0.01/min ($0.60/hour) | $15/M chars (~$0.75/hour) | separate | ~$1.35+ |
| **BP local** | free | free | free | **hardware + electricity only** |

**Local wins on cost, loses on SK→EN quality.** Commercial case: privacy/GDPR + no per-minute fees, but SK→EN must be fixed.

---

## 3. Voice Cloning Comparison

| Approach | Training | Inference RTF | VRAM | Best for |
|----------|----------|---------------|------|----------|
| Piper fine-tune | 8-9h (M1 Pro) | 0.04 (CPU) | minimal | CPU-only, SK voices |
| OmniVoice zero-shot | none | 0.18–0.34 (T4) | 5.8GB | GPU, cross-lingual (worse than Kokoro) |
| Kokoro pre-clone | none | **0.03** (T4) | **1.33GB** | **GPU, EN output** (winner) |

**Recommendation:** Kokoro for EN output (GPU), Piper for SK output (CPU). Fine-tuning only matters for CPU-only machines needing a specific SK voice.

---

## 4. Pivot Proposal (Not Yet Approved)

**Drop fine-tuning focus.** Pivot to:

1. **SK→EN**: Canary-1b-v2 AST (NVIDIA, CC-BY-4.0) — direct speech translation, collapses STT+MT into one step. Supports Slovak. **Not yet tested** — Kaggle kernel failed on missing `lightning`/`pytorch-lightning` dep.
2. **SK→EN fallback**: `kinit/whisper-large-v3-turbo-sk` with `initial_prompt` tuning — WER 0.2143 on clean clip, needs to cross under 0.2 on real mic.
3. **EN output TTS**: Kokoro (RTF 0.03, 1.33GB VRAM on T4)
4. **SK output TTS**: Piper (CPU, RTF 0.04)

**Questions for owner:**
1. Confirm pivot: Canary AST for SK→EN + Kokoro for EN TTS + Piper for SK TTS?
2. Authorize Kaggle retry of Canary AST with `pytorch-lightning`?
3. Proceed with documentation cleanup (remove stale OmniVoice/XTTS/F5/Coqui references)?

---

## 5. Next Actions (Ordered)

| # | Role | Effort | Dependency | Acceptance |
|---|------|--------|------------|------------|
| 1 | Builder | 1 day | Kaggle creds | Retry Canary AST on T4 with `pytorch-lightning`; test `kinit/parakeet-tdt-0.6b-v3-sk` and `kinit/whisper-medium-sk` |
| 2 | Builder | 2 hours | none | Remove stale OmniVoice/XTTS/F5 doc references; document Kokoro + Piper split |
| 3 | Owner | Decide | Kaggle results | Confirm Canary AST path or accept SK→EN experimental-only |

---

## 6. Kaggle Kernel State

- Kernel: `yegorby/bp-tts-gpu-compare` (v6 pushed, completed)
- Results: `/tmp/kaggle_output5/results.json` (local copy)
- Kaggle token: in `~/.env` as `KAGGLE_TOKEN=[REDACTED]` (do not print)
- Kernel needs `pytorch-lightning` for Canary AST (failed on v6)
- Kokoro TTS confirmed: RTF ~0.03, VRAM 1.33GB on T4
- `kinit/whisper-large-v3-turbo-sk`: WER 0.2143 on clean SK clip (2.99GB VRAM)
- `whisper-small-sk`: WER 0.2143 on clean SK clip (3.61GB VRAM)
- `whisper-large-v3` base: WER 0.3636 on clean SK clip

---

## 7. Files to Clean Up (Documentation)

Stale references to remove/replace in `documentation/`:
- OmniVoice, XTTS, Coqui, F5 references (torch-free Windows runtime can't run them)
- Add: Kokoro as GPU TTS path, Piper as CPU TTS path
- Files: `documentation/model_landscape_2026-10.md`, `documentation/load_tests_2026-10.md`, `documentation/linux_setup_and_test.md`, `AGENTS.md` §8-9, §146-149

---

## 8. FLIP Condition

Verdict becomes "Ship with caveats" if Canary AST or tuned `kinit/whisper-large-v3-turbo-sk` crosses under 0.2 WER on real Slovak mic audio.

---

**Not yet committed.** Nothing pushed to `origin/main`. Owner approval required before any push (constitution + AGENTS.md hard rule).
