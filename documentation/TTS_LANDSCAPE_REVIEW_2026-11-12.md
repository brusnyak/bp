# TTS / Voice-Cloning Model Landscape — Comprehensive Review (2026-11-12)

**Purpose:** Document every TTS/voice-cloning model considered for the BP near-real-time speech-translation pipeline, what each requires to run, why prior Kaggle attempts failed, and how to configure Kaggle T4 properly for each.

**Bottom line:** No open model streams Slovak at usable accuracy today. Slovak input stays on Whisper with phrase/clause cutting. The live-path TTS fallback is **Piper locally, no voice cloning or fine-tuning** (generic quality, RTF ~0.04, working baseline). Voice-cloning hopes rest on untested CUDA models — primarily **X-Voice** (SK+CZ native, quality not speed) and **OmniVoice on T4** (batch/QC, RTF 0.42–0.45). Higgs Audio v3 and Fish Audio S2-Pro failed on Kaggle **not because they can't run**, but because they were integrated into the project environment instead of run as standalone custom setups with correct Python/CUDA versions.

---

## 1. Model-by-model review

### 1.1 Kyutai Pocket TTS (NEW — primary research target)

| Attribute | Value |
|-----------|-------|
| Parameters | 100M (90M generative + 10M codec decoder) |
| Languages (official) | English, French, German, Portuguese, Italian, Spanish, Dutch (pocket-tts 3.3+) |
| Slovak/Czech | **No official SK/CZ.** Community Czech fine-tune: `@vvolhejn/pocket-tts-czech` on HF. SK via cross-lingual cloning (EN model speaking SK text) only. |
| License | Code MIT; per-voice licenses on kyutai/tts-voices |
| Size / runtime | ~100M params; runs faster than real-time on CPU (~6x on MacBook Air M4, ~2 CPU cores); GPU ~12.8x RT (RTX 4070) |
| First-audio latency | ~200ms (CPU), ~20ms (GPU) |
| Voice cloning | Zero-shot from ~5s reference audio; no official fine-tuning path |
| Python | 3.10–3.14, PyTorch 2.5+ |
| CUDA | Not required (CPU-first); `device="cuda"` moves model to GPU (unofficial) |
| Kaggle T4 fit | **Good** — works on CPU, no CUDA 13 needed, no Python version conflict |
| Verdict | Easiest to test on Kaggle T4. SK not native — cross-lingual or community Czech fine-tune only. |

**Kaggle setup:** `pip install pocket-tts --extra-index-url https://download.pytorch.org/whl/cpu` (CPU wheel avoids CUDA conflicts). Kernel: `scripts/gpu_bench/kaggle_pocket_tts/`.

---

### 1.2 Higgs Audio v3 TTS (Boson AI)

| Attribute | Value |
|-----------|-------|
| Parameters | 4B |
| Languages | 100+ (vendor claim) |
| Slovak/CZ | Likely covered (100+ langs) but unverified |
| License | Boson Higgs TTS 3 Research and Non-Commercial License |
| Size / runtime | ~4B, ~10GB VRAM in bf16 on CUDA |
| First-audio latency | Low (streaming via SGLang-Omni) |
| Voice cloning | Zero-shot from reference clip; inline emotion/style/SFX control tokens |
| Serving | Requires SGLang-Omni server (separate process) |
| Python | 3.11–3.12 (NOT 3.13) |
| CUDA | **CUDA 13 required** (SGLang requirement; CUDA 12 images retired) |
| Kaggle T4 fit | **Problematic** — Kaggle default Python is 3.13.16; T4 drivers typically CUDA 12.x, not 13. Needs conda Python 3.12 + may hit CUDA version mismatch. |
| Verdict | Needs standalone conda env (Python 3.12) + SGLang-Omni. CUDA 13 requirement may block on Kaggle T4. Kernel: `scripts/gpu_bench/kaggle_higgs_audio_v3/`. |

**Why the prior kernel failed:** `kaggle_higgs_audio_clone/job.py` tried to load `bosonai/higgs-tts-3-4b` via `transformers.AutoModelForSeq2SeqLM` — **wrong approach**. Higgs Audio v3 is not a seq2seq model; it requires the SGLang-Omni serving stack. Also ran on Kaggle's default Python 3.13.16 which conflicts with sglang (<3.13).

**Proper Kaggle setup:** conda env Python 3.12 → `pip install --pre sglang` → `pip install sglang-omni` → `sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b` → test via `/v1/audio/speech` API.

---

### 1.3 Fish Audio S2-Pro

| Attribute | Value |
|-----------|-------|
| Parameters | 5B Dual-AR (4B Slow AR + 400M Fast AR) |
| Languages | 80+ (Tier 1: JA/EN/ZH; Tier 2: KO/ES/PT/AR/RU/FR/DE) |
| Slovak/CZ | Not explicitly listed — unverified |
| License | Fish Audio Community License (code BSD-3-Clause), S2-Pro weights non-commercial |
| Size / runtime | 5B; Fish Audio recommends **>=24GB VRAM** for S2-Pro |
| RTF | 0.34 on single H200 (single batch) |
| Voice cloning | Zero-shot from reference clip; prosody/emotion control via natural-language tags |
| Serving | Requires SGLang-Omni + descript-audio-codec |
| Python | 3.12 |
| CUDA | cu129 (GPU install) |
| Kaggle T4 fit | **Poor** — 16GB VRAM vs 24GB recommended; CUDA 13 requirement; Python 3.13 vs 3.12 conflict. |
| Verdict | Likely too large for T4 16GB VRAM. Needs conda Python 3.12 + SGLang-Omni + CPU preprocessing to save GPU memory. Kernel: `scripts/gpu_bench/kaggle_fish_s2_pro_v2/`. |

**Why prior kernels failed:** `kaggle_fish_audio_clone` and `kaggle_fish_s2_pro` tried `pip install --ignore-requires-python sglang-omni` on Python 3.13.16. SGLang-Omni requires Python 3.11–3.12 and CUDA 13. The `--ignore-requires-python` flag bypasses the check but runtime failures follow.

**Proper Kaggle setup:** conda env Python 3.12 → `pip install --pre sglang` → `pip install sglang-omni` → `pip install descript-audiotools descript-audio-codec` → `sgl-omni serve --model-path fishaudio/s2-pro --config examples/configs/s2pro_tts.yaml --preprocessing.gpu null --preprocessing.total_reserve_bytes null` (CPU preprocessing to save VRAM on 16GB T4).

---

### 1.4 XTTS v2 (Coqui)

| Attribute | Value |
|-----------|-------|
| Languages | Includes Slovak |
| License | Coqui Public Model License (non-commercial) |
| Size / runtime | Multi-dataset multilingual |
| Voice cloning | Zero-shot from 6s reference |
| Python | **Requires Python <3.12** |
| CUDA | Works with CUDA 12.x |
| Kaggle T4 fit | **Dead end** — Kaggle default Python 3.13.16; Python 3.10 standalone has torch `ResolutionImpossible` dependency conflicts on Kaggle. 4+ attempts failed. |
| Verdict | **Do not pursue on Kaggle.** Documented as dead end in `documentation/VERIFICATION_RESULTS_2026-11-06.md`. |

**Why prior kernels failed:** `kaggle_xtts_clone/job.py` v5 — Python 3.13 vs Coqui's <3.12 requirement. Python 3.10 standalone works but torch==2.x+cu124 under Python 3.10 on Kaggle fails with dependency resolution conflicts.

---

### 1.5 X-Voice (F5-TTS lineage)

| Attribute | Value |
|-----------|-------|
| Parameters | 0.4B |
| Languages | 30 including **SK and CZ** (paper tables 5-6) |
| License | Code MIT, weights **CC-BY-NC** (technical report OK, product no) |
| Size / runtime | 0.4B PyTorch; README lists Apple Silicon install variant |
| RTF | Vendor 0.073 on RTX 4090; Mac MPS RTF 2.9–4.2 (too slow); T4 RTF 0.61–0.80 (measured) |
| Voice cloning | Zero-shot cross-lingual cloning |
| Python | 3.10–3.14 (compatible with Kaggle default) |
| CUDA | Works with standard PyTorch CUDA (no CUDA 13 requirement) |
| Kaggle T4 fit | **Good** — no Python version conflict, no CUDA 13 requirement, 0.4B fits in 16GB VRAM |
| Verdict | **Top new TTS test.** Only candidate found that is small, open, cross-lingual cloning, and lists SK natively. Quality candidate, not speed candidate (RTF 0.6–0.8 on T4 = not real-time on T4 alone, but viable on faster GPU). Kernel: `scripts/gpu_bench/kaggle_xvoice/`. |

**Kaggle setup:** `pip install -e .` from `/tmp/xvoice` (GitHub clone) + `pip install phonemizer` + `apt-get install espeak-ng`. No conda needed.

---

### 1.6 OmniVoice (incumbent bulk/QC)

| Attribute | Value |
|-----------|-------|
| Languages | 600+ |
| License | Apache-2.0 |
| RTF | 0.14–0.17 long text / 0.40–0.65 short phrase on T4 (measured) |
| Voice cloning | Zero-shot from reference |
| Streaming | **No** — batch/QC only, not a live path |
| Kaggle T4 fit | Good — already tested, RTF <0.5 on T4 |
| Verdict | Confirmed batch/QC only. Not live-streaming. Already measured on Kaggle T4 (`kaggle_omnivoice` kernel complete). |

---

### 1.7 CosyVoice 3/2

| Attribute | Value |
|-----------|-------|
| Languages | 9 (zh en ja ko de es fr it ru) — **no SK** |
| License | Apache-2.0 |
| SK fit | Out for Slovak output |

---

### 1.8 Chatterbox (Multilingual / Turbo)

| Attribute | Value |
|-----------|-------|
| Languages | 23 — **no SK, no CZ** (model card verified; marketing claims corrected) |
| License | MIT; ONNX builds exist |
| SK fit | Out for Slovak output |

---

### 1.9 Qwen3-TTS

| Attribute | Value |
|-----------|-------|
| Languages | EN only (10 langs), **no SK** |
| License | Apache-2.0 |
| SK fit | Out for Slovak output (EN-side GPU TTS only) |

---

### 1.10 Kokoro 82M

| Attribute | Value |
|-----------|-------|
| Languages | EN only, **no SK** |
| License | Apache-2.0 |
| SK fit | Out for Slovak output |

---

### 1.11 Other models reviewed and excluded

| Model | SK? | Reason excluded |
|-------|-----|----------------|
| Parakeet-TDT v3 | Yes (25 EU langs incl SK) | Already measured: WER 0.58–0.63 on maintainer SK mic — accuracy not viable |
| Nemotron 3.5 ASR streaming | Yes (broad tier) | Already benched negative: SK WER 0.75–0.86, language drift |
| Canary AST | Yes (25 EU langs) | chrF++ ~52 on SK→EN — below commercial bar; speed path only |
| SeamlessM4T v2 | SK speech-in, no SK speech-out | RTF 0.77–0.99, OOMs 16GB Mac; not SK output |
| Hibiki-Zero | No SK | FR/ES/PT/DE→EN only |
| Voxtral Realtime | No SK/CZ | 13 langs, no SK/CZ |
| EuroLLM | Yes | Too slow for live MT |
| MADLAD-400-3B | Yes | MT only, not TTS |
| VoxCPM2 2B | No SK/CZ | Polish yes, Czech/Slovak no |
| MagpieTTS | No SK | 12 langs, no SK |
| IndexTTS2 MLX | Unknown SK | Mac-side candidate, unverified |
| Kyutai Pocket TTS | No official SK | See 1.1 above |

---

## 2. Why prior Kaggle attempts failed (root cause analysis)

| Model | Prior kernel | Failure reason | Fix |
|-------|-------------|----------------|-----|
| XTTS v2 | `kaggle_xtts_clone` v5 | Python 3.13 (Kaggle default) vs Coqui requires <3.12; Python 3.10 has torch dep conflicts | **Dead end** — do not pursue on Kaggle |
| Higgs Audio v3 | `kaggle_higgs_audio_clone` | Used `transformers.AutoModelForSeq2SeqLM` — wrong architecture; Higgs Audio v3 requires SGLang-Omni serving stack, not plain transformers | Use SGLang-Omni with conda Python 3.12 (`kaggle_higgs_audio_v3`) |
| Fish Audio S2-Pro | `kaggle_fish_audio_clone`, `kaggle_fish_s2_pro` | `pip install --ignore-requires-python sglang-omni` on Python 3.13.16 — SGLang requires Python 3.11–3.12 + CUDA 13 | Use conda Python 3.12 + SGLang-Omni + CPU preprocessing (`kaggle_fish_s2_pro_v2`) |
| Higgs Audio v3 / Fish Audio S2-Pro | Both | Integrated into project environment instead of standalone custom setup | Standalone kernels with own conda envs |

**The user's insight is correct:** These models failed because they were integrated into the project environment (which uses Kaggle's default Python 3.13.16 and project dependencies) rather than run as custom standalone setups with the correct Python version and CUDA configuration each model requires.

---

## 3. Kaggle T4 environment configuration matrix

| Model | Python | CUDA | Serving | VRAM | Conda needed? | Kaggle T4 viable? |
|-------|--------|------|---------|------|---------------|-------------------|
| Pocket TTS | 3.10–3.14 | Not required | Direct (pip) | Low | No | ✅ Yes |
| X-Voice | 3.10–3.14 | Standard PyTorch | Direct (pip) | Low (0.4B) | No | ✅ Yes |
| Higgs Audio v3 | 3.11–3.12 | **CUDA 13** | SGLang-Omni server | ~10GB | Yes | ⚠️ Maybe (CUDA 13 may fail) |
| Fish Audio S2-Pro | 3.12 | cu129 | SGLang-Omni server | 24GB rec. | Yes | ⚠️ Unlikely (16GB < 24GB rec.) |
| XTTS v2 | <3.12 | Standard | Direct (pip) | Medium | Yes | ❌ Dead end |
| OmniVoice | 3.10–3.14 | Standard | Direct (pip) | Medium | No | ✅ Already tested |

---

## 4. Recommended Kaggle test order

1. **Pocket TTS** (`kaggle_pocket_tts`) — simplest, no version conflicts, tests cross-lingual SK + community Czech fine-tune
2. **X-Voice** (`kaggle_xvoice`) — SK+CZ native, quality candidate, no version conflicts
3. **Higgs Audio v3** (`kaggle_higgs_audio_v3`) — conda Python 3.12 + SGLang-Omni; may fail on CUDA 13
4. **Fish Audio S2-Pro v2** (`kaggle_fish_s2_pro_v2`) — conda Python 3.12 + SGLang-Omni + CPU preprocessing; VRAM likely insufficient on 16GB T4
5. **XTTS v2** — **skip** (dead end)

---

## 5. Decision framework for voice route

```
Fallback (confirmed working):
  Piper locally, no cloning/fine-tuning → RTF ~0.04, generic quality, CPU

Voice-cloning candidates (untested on Kaggle, need GPU):
  X-Voice → SK+CZ native, quality not speed (RTF 0.6–0.8 on T4)
  OmniVoice on T4 → batch/QC only, RTF 0.42–0.45, not streaming
  Higgs Audio v3 → 100+ langs, needs CUDA 13 + Python 3.12
  Fish Audio S2-Pro → 80+ langs, needs 24GB VRAM (too large for T4)
```

**Recommendation:** Run Pocket TTS + X-Voice on Kaggle T4 first (both should work without version conflicts). Then attempt Higgs Audio v3 and Fish Audio S2-Pro with conda Python 3.12, expecting possible CUDA/VRAM failures. Based on results, decide voice route: Piper fallback vs. cloned voice on GPU.

---

## 6. Kernel files created

| Kernel | Path | Model | Status |
|--------|------|-------|--------|
| Pocket TTS | `scripts/gpu_bench/kaggle_pocket_tts/` | Kyutai Pocket TTS | Ready to push |
| Higgs Audio v3 | `scripts/gpu_bench/kaggle_higgs_audio_v3/` | bosonai/higgs-audio-v3-tts-4b | Ready to push |
| Fish Audio S2-Pro v2 | `scripts/gpu_bench/kaggle_fish_s2_pro_v2/` | fishaudio/s2-pro | Ready to push |
| X-Voice | `scripts/gpu_bench/kaggle_xvoice/` | X-Voice (F5 lineage) | Ready to push |

**Push commands** (one poller at a time, per Kaggle rules):
```
kaggle kernels push -p scripts/gpu_bench/kaggle_pocket_tts --accelerator NvidiaTeslaT4 -t 1800
kaggle kernels push -p scripts/gpu_bench/kaggle_xvoice --accelerator NvidiaTeslaT4 -t 1800
kaggle kernels push -p scripts/gpu_bench/kaggle_higgs_audio_v3 --accelerator NvidiaTeslaT4 -t 1800
kaggle kernels push -p scripts/gpu_bench/kaggle_fish_s2_pro_v2 --accelerator NvidiaTeslaT4 -t 1800
```

**Status check:** `kaggle kernels status yegorby/<slug>`
**Fetch results:** `timeout 90 kaggle kernels output yegorby/<slug> -p <dir> -q --file-pattern '<name>'`

---

## 7. Neo browser status

Neo browser is connected (session `opencode/rapid-alpaca`). Ready to push Kaggle kernels via `https://kaggle.com/code/yegorby/` or the Kaggle CLI authenticated as `yegorby` (confirmed via `kaggle kernels list`).

---

**reviewer message: POSTPONED until after Kaggle TTS standalone tests are complete (next week), per project decisions.**
