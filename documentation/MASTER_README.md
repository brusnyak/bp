# BP — Master Documentation (single source of truth)

**Project:** Real-time EN↔SK speech translation (bachelor's thesis, STU FEI)
**Repo:** `~/Developer/bp-main` | **Branch:** `backend-quality-2026-11-07` | **Owner:** Yegor Brusniak
**Status:** 31 commits ahead of `origin/main`, uncommitted (push needs owner yes)
**Last updated:** 2026-11-12

---

## 0. Start here — where we stand

**Verdict:** One speaker, one direction runs live on a laptop (EN→SK/DE/CS). The 2 s median-lag goal is met on **generated English** on the M1 Pro Mac (0.5–1.8 s), but **Slovak input under 2 s is not viable on this Mac** — Whisper turbo is a ~0.7 s/call floor (30 s padded window), Nemotron scores 28% on Slovak. **Better compute is the gate.**

**Open problems:**
1. Slovak input accuracy/drift under 2 s (needs GPU or streaming-native recogniser)
2. Voice decision: Piper (generic, fast) vs. cloned voice (better quality, needs GPU)
3. Room mode real-model test pending
4. Real-speech (non-generated) E2E test pending

**Working today (each verified):**
| Capability | Evidence |
|------------|----------|
| Default install on Linux (clean container) | `linux_setup_and_test.md` |
| Real server on Kaggle T4, EN→DE & EN→SK, zero server errors | `route_test.json` |
| Live route EN→SK on Mac, MLX turbo | `evidence/2026-11-07/` |
| Per-language quality & timing (T4) | `LANGUAGES.md`, `BENCHMARKS.md` |
| Room mode built & unit-tested (no real models yet) | `test/room_tests.py` |
| `make test`: 141 pass, 1 known flake | `test/` |
| `make check-docs`: 57/57 claims verified | `scripts/check_readme_claims.py` |

---

## 1. Model landscape — every TTS/STT/MT model considered

All model research is consolidated in `documentation/model_landscape_2026-10.md` (Oct 2026) plus `documentation/TTS_LANDSCAPE_REVIEW_2026-11-12.md` (this session's TTS review).

### STT — speech recognition
| Model | Slovak? | License | Best measured WER | Verdict |
|-------|---------|---------|-------------------|---------|
| whisper-small-sk (incumbent) | native SK fine-tune | MIT | 0.13 FLEURS-sk / 0.26 owner mic | **Keep as SK default on CPU** |
| whisper base (EN) | n/a | MIT | 0.077 | Keep for EN |
| large-v3-turbo | 99 langs incl SK | MIT | 0.41–0.52 owner mic | Rejected (owner mic too high) |
| Parakeet-TDT v3 | 25 EU langs incl SK | CC-BY-4.0 | 0.58–0.63 owner mic | Speed proven, SK accuracy not |
| Nemotron 3.5 streaming 0.6B | SK broad tier | OpenMDW | 0.75–0.86 SK owner mic | **Rejected for SK** |
| Canary-1B / Flash / 180M | 25 EU langs incl SK | NVIDIA CC-BY-4.0 | 0.0477 SK FLEURS | Speed path, AST chrF++ ~52 below commercial bar |

### MT — machine translation
| Model | SK↔EN? | License | Best chrF | Verdict |
|-------|--------|---------|-----------|---------|
| Opus-MT sk-en / en-sk | dedicated pair | CC-BY-4.0 | 62.1 (sk→en) | **Keep — speed crown unchallenged** |
| NLLB-200-distilled-600M | 200+ langs incl SK | CC-BY-NC | ~59 | Non-commercial trap, benchmark only |
| MADLAD-400-3B-CT2 | 400+ langs incl SK | Apache-2.0 | 36.0 BLEU (news) | Top MT experiment, untested on SK noisy STT |

### TTS — text to speech (incl. voice cloning)
| Model | Slovak? | License | RTF | Streaming? | Verdict |
|-------|---------|---------|-----|------------|---------|
| **Piper** (incumbent live) | `sk_SK-lili` base only | MIT | 0.03–0.05 CPU | ✅ Yes | **Live path fallback — generic quality, working baseline** |
| Piper + own fine-tune (`me_omni_piper_sk`) | SK | MIT | 0.03–0.05 | ✅ Yes | Trained, WER 0.19 vs 0.54 old |
| **OmniVoice** | 600+ (zero-shot clone) | Apache-2.0 | 0.40–0.65 T4 | ❌ No (batch/QC) | Batch/QC only, not live |
| **XTTS v2** (Coqui) | **No Slovak**; Czech yes, English yes (17 langs) | Non-commercial (CPML) | 0.52-0.63 T4 | ✅ Yes (first chunk 0.45-0.48 s) | **Measured 2026-11-15 (T4, 3 repeats):** EN word error 0%, CS 4-10%, 1.9-2.5 GB GPU. Usable for EN/CS cloning; not for Slovak. Evidence `evidence/2026-11-15/kaggle_xtts_t4_results.json` |
| **X-Voice** (F5 lineage) | **Yes: sk, cs, en measured** (30 langs) | Code MIT, weights CC-BY-NC | Stage1 0.37-0.47 (16 steps), 0.7-0.89 (32 steps) T4 | ❌ No (whole phrase: 2.2-2.7 s for a 3-5 s sentence at 16 steps) | **Measured 2026-11-15 (T4, 3 repeats):** word error 0-11% EN/SK/CS, < 1.3 GB GPU. Best cross-lingual clone quality of the three, too slow per phrase on a T4 for the 2 s goal; paper reports RTF 0.073 on an RTX 4090. Evidence `evidence/2026-11-15/kaggle_xvoice_t4_results.json` |
| **Pocket TTS** (Kyutai 100M) | No official SK | MIT | 0.63 T4 | ✅ Yes | No native SK, too slow on T4, cloning needs HF auth |
| **Higgs Audio v3** (Boson 4B) | Likely (100+ langs) | Non-commercial | — | ? | ❌ Failed on Kaggle (flashinfer compile timeout >5 min) |
| **Fish Audio S2-Pro** | Unverified (80+ langs) | Non-commercial | — | ? | ❌ Discarded — needs ≥24GB VRAM, T4 has 16GB |
| Kokoro 82M | EN only | Apache-2.0 | 0.03 T4 | ✅ Yes | EN TTS only, no SK |
| Qwen3-TTS | EN only | Apache-2.0 | — | ? | EN zero-shot cloning, no SK |

**Bottom line:** No open model streams Slovak at usable accuracy today. Slovak input stays on Whisper with phrase/clause cutting. Live-path TTS = Piper (generic, fast). Voice-cloning hopes rest on untested CUDA models (XTTS, X-Voice, Higgs Audio) — all failed on Kaggle T4 due to environment issues.

---

## 2. Kaggle T4 standalone tests — this session

All kernels pushed to private Kaggle account `yegorby`, Tesla T4 GPU, one poller at a time.

| Kernel | Slug | Model | Status | Result |
|--------|------|-------|--------|--------|
| Pocket TTS | `yegorby/bp-kyutai-pocket-tts-t4` | Kyutai Pocket TTS 100M | ✅ complete | RTF 0.63 on T4, EN catalog voice works, no native SK, cloning needs HF auth |
| X-Voice | `yegorby/bp-x-voice-t4` | X-Voice (F5 lineage, SK+CZ native) | ✅ complete (v9) | Stage1/Stage2 x 16/32 steps x en/sk/cs, see table above |
| XTTS v2 | `yegorby/bp-xtts-voice-clone-t4` | Coqui XTTS v2 | ✅ complete (v15) | EN/CS measured, SK unsupported |
| Higgs Audio v3 | `yegorby/bp-higgs-audio-v3-t4` | Boson Higgs Audio v3 4B | ❌ error | flashinfer CUDA compile timeout >5 min |
| Fish Audio S2-Pro | — | fishaudio/s2-pro | ❌ discarded | Needs ≥24GB VRAM, T4 has 16GB |

**Check status:** `kaggle kernels status yegorby/<slug>`
**Fetch results:** `timeout 90 kaggle kernels output yegorby/<slug> -p <dir> -q --file-pattern '<report>.json'`

---

## 3. E2E streaming test infrastructure (new)

Two new scripts for proper end-to-end streaming measurement:

### `scripts/live/e2e_streaming_test.py`
Full STT→MT→TTS pipeline via WebSocket. Feeds audio in real-time chunks, measures per-stage latency:
- `stt_s` — audio chunk end → transcription
- `mt_s` — transcription → translation
- `tts_s` — translation → first audio
- `total_s` — end-to-end chunk end → first audio
- `rtf` — TTS real-time factor
- `first_word_lag_s` — what the listener perceives

```bash
# Start server first:
BP_NO_SSL=1 BP_PORT=8000 .venv/bin/python app.py &

# Run E2E test:
.venv/bin/python scripts/live/e2e_streaming_test.py \
    --audio processed/testaudio/en_paced.wav --src en --tgt sk \
    --tts piper --clone --ref <ref_wav> \
    --out processed/e2e_results.json
```

### `scripts/live/e2e_compare_tts.py`
Runs the same E2E test across multiple TTS engines and compares against baselines:
```bash
.venv/bin/python scripts/live/e2e_compare_tts.py \
    --audio processed/testaudio/en_paced.wav --src en --tgt sk \
    --tts piper,omnivoice,xtts,xvoice,pocket \
    --clone --ref <ref_wav> \
    --out processed/e2e_compare
```

**Baselines for comparison:**
- Piper: RTF 0.03–0.05 (CPU, live path)
- OmniVoice: RTF 0.40–0.65 (T4, batch/QC only)

---

## 4. Ordered actions (next session)

| # | Action | Where | Acceptance | Needs |
|---|--------|-------|------------|-------|
| 1 | Wait for XTTS v13 Kaggle result; read `xtts_report.json` | Kaggle `yegorby/bp-xtts-voice-clone-t4` | RTF, first-audio latency per language | Kaggle slot free |
| 2 | Fix X-Voice (ctranslate2 + fastlid SSL) and re-push | Kaggle `yegorby/bp-x-voice-t4` | Successful SK synth with RTF | Kaggle slot free |
| 3 | Fix Higgs Audio v3 (flashinfer compile timeout) and re-push | Kaggle `yegorby/bp-higgs-audio-v3-t4` | Server ready + TTS per language | Kaggle slot free |
| 4 | Run E2E streaming tests locally with Piper baseline | Mac, `scripts/live/e2e_streaming_test.py` | per-segment lag under 2 s goal | Server running, load < 10 |
| 5 | Run E2E compare across all TTS engines | Mac + Kaggle results | comparison table vs baselines | All kernels complete |
| 6 | Pick voice route: Piper fallback vs. cloned voice | Owner decision | documented in this file | Ear check / Kaggle data |
| 7 | Commit working tree (31 commits ahead, uncommitted) | `git commit` + `git push` | tree clean, pushed to origin | Owner yes |
| 8 | Room mode real-model test | Kaggle `bp-room-test-t4` | speaker lag ≤ 2× baseline | After voice route decided |
| 9 | Real-speech E2E test (non-generated audio) | Mac, `paced_replay.py` | lag ≤ 2 s median | Owner go, load < 10 |

---

## 5. Handler message — postponed

**Do not message Ivan Minárik until Kaggle TTS tests are complete and voice route is decided.** The DM drafted earlier (Slovak, asking for uni server / better laptop with GPU) is on hold. Once we have conclusive data (XTTS/X-Voice/Higgs results + E2E comparison), send the minimized DM asking for compute access.

---

## 6. How to run the project

```bash
# Clone
git clone https://github.com/brusnyak/bp.git && cd bp

# Setup (Mac, Python 3.11 required)
python3.11 scripts/setup.py --profile mac --langs de,sk --dev

# Run server
BP_NO_SSL=1 BP_PORT=8000 make run PRESET=mac
# → http://localhost:8000

# Run tests
make test          # 141 pass, 1 known flake
make check-docs    # 57/57 claims verified
make demo-check    # demo pre-flight

# Run E2E streaming test
.venv/bin/python scripts/live/e2e_streaming_test.py \
    --audio processed/testaudio/en_paced.wav --src en --tgt sk --tts piper

# Kaggle kernel push (one at a time)
kaggle kernels push -p scripts/gpu_bench/kaggle_<name> --accelerator NvidiaTeslaT4 -t 1800
```

---

## 7. Evidence locations

| What | Path |
|------|------|
| STT/MT per-language measurements | `documentation/evidence/2026-11-07/lang_matrix.json` |
| Whole-chain timing (25 clips) | `documentation/evidence/2026-11-07/e2e_chain.json` |
| Streaming strategy simulation | `documentation/evidence/2026-11-07/stream_sim.json` |
| Real server route test | `documentation/evidence/2026-11-07/route_test.json` |
| Slovak model comparison | `documentation/evidence/2026-11-07/sk_compare.json` |
| Strategy matrix (generic audio, 186 cells) | `documentation/evidence/2026-11-10/strategy_matrix_generic_t4.json` |
| TTS probes (OmniVoice, Piper, OpenVoice) | `documentation/evidence/2026-11-10/tts_probe_*.json` |
| Room test (T4, 3 & 12 listeners) | `documentation/evidence/2026-11-10/room_test_t4_v3.json` |
| Voxtral route tests | `documentation/evidence/2026-11-09/voxtral_route_*.json` |
| Test audio (paced/continuous/long, EN/SK/CS) | `processed/testaudio/` |
| E2E streaming results (new) | `processed/e2e_*.json` (create via scripts) |

---

## 8. Files changed this session (uncommitted)

| File | Change |
|------|--------|
| `documentation/TTS_LANDSCAPE_REVIEW_2026-11-12.md` | **NEW** — comprehensive TTS/voice-cloning landscape review |
| `scripts/live/e2e_streaming_test.py` | **NEW** — E2E streaming pipeline test |
| `scripts/live/e2e_compare_tts.py` | **NEW** — TTS engine comparison |
| `scripts/gpu_bench/kaggle_pocket_tts/job.py` | Fixed — catalog voices, no HF auth |
| `scripts/gpu_bench/kaggle_xvoice/job.py` | Fixed — ctranslate2 unconstrained, SSL verify off |
| `scripts/gpu_bench/kaggle_xtts_clone/job.py` | Fixed — COQUI_TOS_AGREED at runtime, stdin redirect |
| `scripts/gpu_bench/kaggle_higgs_audio_v3/job.py` | Fixed — uv+Python 3.12 venv, `hf` CLI |
| 4 kernel-metadata.json files | Added `yegorby/bp-t4-assets-20261105` dataset source |

---

## 9. How to continue next session

1. **Read this file first** — it's the single source of truth.
2. **Check Kaggle status:** `kaggle kernels status yegorby/bp-xtts-voice-clone-t4` (and x-voice, higgs)
3. **Read results:** `timeout 90 kaggle kernels output yegorby/<slug> -p /tmp/<name> -q --file-pattern '<report>.json'`
4. **Run E2E tests locally** with Piper baseline, then compare across engines.
5. **Decide voice route** (Piper fallback vs. cloned voice).
6. **Commit + push** (needs owner yes).
7. **Message handler** with results + compute ask (uni server / GPU laptop).

---

*This document is the single entry point. All other documentation files in `documentation/` are reference material; this file tells you which one to read for what.*
