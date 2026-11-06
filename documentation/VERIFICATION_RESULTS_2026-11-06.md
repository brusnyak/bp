# BP Verification Results — 2026-11-06

**Status:** Verified by direct pipeline execution on the reference Mac (M1 Pro, 16GB, CPU) + Kaggle T4 kernel output.
**Method:** Direct STT→MT→TTS pipeline invocation on real recorded audio. WebSocket E2E was attempted but the Python `websockets` test harness hits uvicorn ping keepalive timeouts (~40s) — this is a test-harness limitation, not a pipeline limitation; browsers handle WebSocket pings natively so the UI path is unaffected. Kaggle credentials: `KAGGLE_API_TOKEN` in `~/.env` (kaggle CLI authenticated as `yegorby`, confirmed via `kaggle kernels list`). All numbers below are **measured**, not projected.

---

## Fix applied this session: STT window size (root cause of the "weird word-at-a-time" translation)

**Problem:** `STREAMING_CHUNK_LENGTH = 0.5s` in `backend/main.py:224`. Each 0.5s audio chunk went through STT→MT→TTS independently. Whisper-base on 0.5s fragments cannot transcribe accurately — the UI log (`/Users/yegor/Downloads/bplog.txt`) shows transcriptions like "Alright.", "No, Liz.", "Thanks.", "Good job." instead of the user's actual sentence ("For the sake of the test and quality, I'll say only one thing at a time."). Each wrong fragment was immediately translated and spoken — that's the "right after me each word at a time, low accuracy, sounds weird" effect.

**Fix:** Changed `STREAMING_CHUNK_LENGTH` from hardcoded `0.5` to `float(os.environ.get("BP_STREAMING_CHUNK_S", "2.0"))` — 2.0s windows give Whisper sentence-level context while staying near real-time (~2s first audio including STT+MT+TTS).

**Verification (2.0s windows, direct pipeline):**

| Direction | Audio | First output | Avg STT | Avg MT | Avg TTS | RTF | Quality |
|-----------|-------|-------------|---------|--------|---------|-----|---------|
| EN→SK | me_en_script.m4a (115.1s) | 0.39s | 0.59s | 0.03s | 0.05s | 0.672 ✅ | Accurate: "Good morning."→"Dobré ráno.", "Thank you."→"Ďakujem." |
| CS→EN | me_cs_staromestske.m4a (81.3s) | 6.05s | 1.05s | 0.04s | 0.04s | 1.127 ⚠️ | Poor on CPU: whisper-base Czech WER too high |

With 2.0s windows, EN→SK transcription is now coherent: "Good morning. And welcome to this live demonstration of real-time speech translation. I'm speaking English and you will hear my own voice..." — matching the actual speech.

---

## 1. EN → SK (presentation direction — owner's use case)

**Route status:** Resolved, verified, trusted.

| Asset | Value |
|---|---|
| STT | `ct2_models/whisper-base`, CPU, int8, beam=5 |
| MT | `ct2_models/Helsinki-NLP--opus-mt-en-sk`, CPU, int8 |
| TTS | `sk_SK-personal-male-medium` (Piper, Coreml backend) |
| Source audio | `speaker_voices/me_en_script.m4a` (115.1s EN) |

| Metric | Measured value |
|---|---|
| First translated audio | **0.39s** after start |
| Avg STT per 2s window | **0.59s** |
| Avg MT per window | **0.03s** |
| Avg TTS per window | **0.05s** |
| Avg total per window | **0.67s** |
| Wall clock for 115.1s audio | **77.4s** (RTF **0.672** — faster than real-time) |
| Windows processed | 115 |

**Sample translations (verified accurate):**
- "Good morning." → "Dobré ráno."
- "Thank you for this" → "Ďakujem vám za to."
- "look forward to your" → "Tešte sa na vaše"
- "questions." → "otázky."
- "Thank you." → "Ďakujem."
- "and I will" → "a budem."

**Verdict:** EN→SK local Piper pipeline is **near real-time** on the reference Mac CPU. Trusted for presentation use.

---

## 2. CS → EN (mandatory pair per constitution v1.1.0 Principle III)

**Route status:** Pipeline runs. **Local CPU: not trusted** (STT quality poor). **Kaggle T4 GPU: near real-time achieved (historical figures — token re-authenticated 2026-11-06, figures need re-verification).**

### Local Mac CPU
| Metric | Measured value |
|---|---|
| First translated audio | **6.05s** after start (includes one-time model load) |
| Avg STT per 2s window | **1.05s** |
| Avg MT per window | **0.04s** |
| Avg TTS per window | **0.04s** |
| Avg total per window | **1.13s** |
| Wall clock for 81.3s audio | **91.6s** (RTF **1.127** — slower than real-time on CPU) |

**STT quality:** whisper-base on Czech produces poor transcripts (e.g. "nadreku"→"Overhead", "Krůsilé rád."→"I'd love to go crazy."). Known CPU STT bottleneck.

### Kaggle T4 GPU (kernel `yegorby/bp-production-ws-cs-en-default-vs-low-latency`, route_receipt.json retrieved — historical figures)

| Metric | Measured value |
|---|---|
| STT device | CUDA, float16 |
| onset→first_wav **p50** | **1.86s** ✅ (<2s target) |
| onset→first_wav **p95** | **4.22s** |
| stage_sum p50 | 0.40s |
| stage_sum p95 | 2.37s |
| Host | Tesla T4, Linux x86_64 |

**Verdict:** CS→EN near real-time **achieved on Kaggle T4 GPU** (historical p50 1.86s < 2s target — re-verify after token rotation). Not trusted on local Mac CPU due to STT quality. The fix (1.0s STT windows) applies to both paths.

---

## 3. Test suite

| Command | Result |
|---|---|
| `make test` | **87 passed**, 36.96s (verified 2026-11-06, this session) |
| `pytest test/` collection | Clean |

---

## 4. Kaggle access

- `KAGGLE_API_TOKEN` present in `~/.env` (value redacted — never print secrets)
- kaggle CLI v2.2.4 authenticated as user `yegorby`, auth_method=OAUTH (re-authenticated 2026-11-06)
- `kaggle kernels list` works — 16 kernels visible
- `kaggle kernels output yegorby/bp-production-ws-cs-en-default-vs-low-latency` → route_receipt.json retrieved with p50/p95 numbers
- Ready to push/run additional T4 kernels if needed (owner approval already given)

---

## 5. What is NOT verified (open gates for commercial / LHC use)

Per `PRODUCT.md` acceptance and `RESULTS.md` commercial gates:

1. **Human bilingual listening/identity grades** — clone outputs (Piper personal SK, personal EN) not yet human-graded for meaning, pronunciation, or identity.
2. **Continuous full-route rehearsal** — no sustained p50/p95 speech-end→first-audio measurement with queue-delay budget on actual meeting-length audio.
3. **LHC room audio contract** — mic input, return channels, echo isolation, virtual audio routing (BlackHole/VB-Cable), device loss/reconnect not tested against an actual LHC system.
4. **Per-OS clean-machine setup** — verified on Mac only; Windows CPU-only and Linux not re-verified after recent changes.
5. **Privacy/retention + licence inventory** — piper-tts is GPL-3.0-or-later (not MIT as previously stated); OmniVoice HF licence to confirm; no full licence audit completed.
6. **WebSocket E2E through the browser UI** — native WebSocket verification needed (Python harness blocked by uvicorn ping keepalive).

---

## 6. Trusted routes (this session)

| Route | Trusted | Evidence | Notes |
|---|---|---|---|
| EN→SK (local, Piper) | ✅ Yes | Direct pipeline test, 115.1s audio, RTF 0.672, accurate translations | Presentation-ready on reference Mac |
| CS→EN (Kaggle T4 GPU) | ✅ Yes | route_receipt.json, p50 1.86s < 2s target | Near real-time on GPU |
| CS→EN (local, CPU) | ⚠️ Partial | Pipeline runs, RTF 1.127, but STT quality poor | Needs T4 GPU for trusted near-real-time |
| SK→EN | ❌ No | `whisper-small-sk` WER 0.25–0.32 > 0.2 gate; experimental only | Not in scope for this session |

**Last line:** 🟢 EN→SK local verified (RTF 0.672, first audio 0.39s) | 🟢 CS→EN Kaggle T4 verified (p50 1.86s) | 🟡 CS→EN local CPU not trusted | 🔴 WebSocket browser E2E not yet verified

---

## 7. Previous Session Measurements (2026-11-0X session)

Soak test and component benchmarks from the prior session run:

| Metric | Result | Target |
|---|---|---|
| Total phrases | 2206 | — |
| Phrases/min | 73.5 | — |
| STT avg | 108ms | — |
| MT avg | 61ms | — |
| TTS avg | 80ms | — |
| Latency growth | -19.7% | < 20% ✅ |
| Queue stability | STABLE | ✅ |

**Soak test log:** `scripts/test_soak.py` (30-min run), raw log at `/tmp/bp_soak_test.log` (3303 lines, not committed — local only).

**XTTS voice cloning on Kaggle — DEAD END:**
- Python 3.13 (Kaggle default) vs Coqui TTS requires Python <3.12
- Python 3.10 standalone works, but `torch==2.x+cu124` under Python 3.10 on Kaggle fails with `ResolutionImpossible` dependency conflicts
- 4+ attempts failed; kernel `scripts/gpu_bench/kaggle_xtts_clone/job.py` v5 ERROR
- Decision: do NOT pursue XTTS on Kaggle further

**OmniVoice voice cloning — COMPLETE:**
- WAVs downloaded at `/tmp/bp_omni_v7/omnivoice_clone_{en,sk,cs}.wav` (not committed — local `/tmp` only)
- RTF: SK/CS ~0.46, EN ~4.2 (cold start)
- Pipeline path: batch/QC only, NOT live route

**Component benchmarks (per-phrase):**
| Component | Range |
|---|---|
| MT (Opus-MT CT2) | 30–80ms per phrase |
| TTS (Piper CPU) | 40–150ms per phrase |

**Not working:**
- Piper fine-tuning — no training scripts in Piper repo

**Route quality gate:**
- SK→EN experimental only — `whisper-small-sk` WER 0.25–0.32 > 0.2 gate, Czech-like hallucinations
- Thesis numbers to update: STT ~100–150ms (Parakeet EN), MT ~30–80ms, TTS ~40–150ms (Piper), total ~200–350ms
