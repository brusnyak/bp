# Research Findings — S2S / Voice Cloning / Enterprise Landscape (2026-11-03)

**Purpose:** single source of truth for what exists (open + paid) in speech-to-speech
translation and voice cloning, with Slovak relevance and 8 GB VRAM (T4) fit. Feeds the
final project scope.

**Research date:** 2026-11-03. All vendor figures from public pricing/blog pages (Oct 2026).
Model numbers: vendor cards + community benchmarks; nothing here is measured on our
hardware unless marked `MEASURED`.

---

## 1. The core problem: Slovak voice cloning has no open zero-shot solution

"Live translation in your own voice" = STT → MT → TTS with speaker-identity transfer into
the target language. For Slovak output, every major open zero-shot voice-cloning TTS model
**excludes Slovak**:

| TTS model | License | Slovak output? | VRAM | Zero-shot cloning | Notes |
|-----------|---------|----------------|------|-------------------|-------|
| XTTS v2 (Coqui) | CPML (non-commercial) | ❌ (17 langs: no SK) | 4-6 GB | 6 s | 17 langs; CPML blocks commercial use |
| Chatterbox Multilingual (Resemble) | MIT | ❌ (23 langs: no SK) | 4-8 GB | 5-7 s | 23 langs; MIT; best open quality per blind tests |
| Qwen3-TTS 0.6B/1.7B | Apache 2.0 | ❌ (10 langs: no SK) | 4-8 GB | 3 s | Apache 2.0; 97 ms first-packet; streaming |
| Kokoro 82M | Apache 2.0 | ❌ (EN only) | 2-3 GB / CPU | No | RTF 0.03; CPU OK; no cloning |
| F5-TTS | CC-BY-NC (non-commercial) | ❌ (zh/en) | 2-3 GB | ~10 s | Non-commercial |
| Piper (fine-tune) | MIT | ✅ SK voices exist | ~0.3 GB / CPU | Fine-tune (not zero-shot) | RTF 0.04; the only SK cloning path we have |
| SeamlessM4T v2 (Meta) | CC-BY-NC | ✅ S2S (slk: Sp,Tx,Sp,Tx) | ~6 GB | Prosody, not identity | End-to-end speech→speech; SK output supported |

**Implication:** there is no off-the-shelf open model that clones a voice into Slovak.
The project's existing path (Piper fine-tune on the owner's voice, then generic SK output)
is one of the few viable routes. Any "your own voice in Slovak" feature requires either
(a) training a Piper/custom TTS on Slovak data, or (b) accepting prosody-preserving S2S
(SeamlessM4T v2) without true speaker identity.

---

## 2. Enterprise speech-to-speech translation (paid cloud)

Slovak coverage is the differentiator. The big two (Teams, Google Meet) do **not** support
Slovak speech translation.

| Vendor | Speech translation | Slovak? | Pricing (2026) | Deployment |
|--------|-------------------|---------|----------------|------------|
| Microsoft Teams Interpreter agent | ~9 languages (Jul 2026) | ❌ | Teams Premium ~$10/user/mo | Cloud (Azure) |
| Teams live translated captions | 50+ spoken / 28 caption | ❌ | Teams Premium | Cloud (Azure) |
| Google Meet Speech Translation | EN↔6 (FR/DE/HI/IT/PT/ES) | ❌ | Workspace / Google AI plan | Cloud |
| Google Meet Gemini 3.5 Live Translate | 63 langs (Oct 2026 rollout) | ⚠️ verify SK | Workspace / Google AI | Cloud |
| DeepL Voice-to-Voice | 40+, incl. all 24 EU (SK is EU) | ✅ likely | ~$0.07-0.20/min | Teams/Zoom/API early access |
| KUDO AI Speech Translator | 77 langs, up to 32/meeting | ✅ explicit | ~$0.80-1.25/min | Teams/Zoom/Meet widget |
| Interprefy Aivia | 82 languages | ⚠️ verify SK | ~$0.60-1.00/min | Web/iOS/Android |
| OpenAI gpt-realtime-translate | 70+ input / 13 output | ⚠️ verify SK | $0.034/min | Realtime API |
| Alibaba Qwen3.8-LiveTranslate | multilingual | ⚠️ verify SK | ~$0.03/min est. | API |

**Local vs cloud cost (from handoff):** Google ~$1.00/hr, Azure ~$3.50/hr, AWS ~$1.35/hr
vs local free. Local wins on cost; loses on SK→EN quality (until Canary/Kokoro/etc. are
validated).

---

## 3. Open-source end-to-end S2S (single model, audio-in → audio-out)

| Model | License | Slovak S2S? | VRAM | Latency | Notes |
|-------|---------|-------------|------|---------|-------|
| SeamlessM4T v2 (Meta) | CC-BY-NC | ✅ slk Sp,Tx,Sp,Tx | ~6 GB | RTF 0.77-0.99 (MEASURED on Mac) | 100 langs; 36 output langs incl SK; not real-time live |
| SeamlessStreaming (Meta) | open | ~100 langs | — | ~2 s | Simultaneous; SK input yes |
| Qwen3-Omni-30B-A3B | Apache 2.0 | ⚠️ 10 spoken output langs (verify SK) | 30B — too big for T4 | E2E ~234 ms | Too large for 8 GB VRAM; server-only |
| Hibiki-Zero 3B (Kyutai) | open | FR/ES/PT/DE→EN only, no SK | 8-12 GB | — | No SK |

**SeamlessM4T v2** is the only open S2S model with Slovak speech output. It's CC-BY-NC
(non-commercial), ~6 GB VRAM, RTF ~0.8-1.0 on our Mac — not a live path on CPU/T4, but a
quality reference and potential offline-voice option.

---

## 4. STT for Slovak (the SK→EN bottleneck)

| Model | License | Slovak WER (owner mic) | RTF | VRAM | Fit for T4 |
|-------|---------|------------------------|-----|------|------------|
| whisper-small-sk (fine-tune) | MIT | 0.2143 clean / ~0.32 mic | 0.12-0.14 | ~3.6 GB | ✅ baseline |
| whisper-large-v3-turbo | MIT | 0.36-0.52 mic | 0.6 | ~800 MB | ❌ too slow/poor on SK |
| Parakeet-TDT-v3 0.6B | CC-BY-4.0 | 0.58-0.63 mic | — | ~680 MB INT8 | ❌ SK accuracy not proven |
| Nemotron 3.5 ASR 0.6B | OpenMDW | 0.75-0.86 mic | 0.33 | — | ❌ rejected (language drift) |
| Qwen3-ASR 0.6B/1.7B | open | 52 langs (verify SK) | RTF 0.009-0.024 | — | ⚠️ untested on SK |
| NVIDIA Canary-1B AST | CC-BY-4.0 | UNKNOWN on SK | — | — | ⚠️ AST collapses STT+MT; untested |

**Canary-1B AST** (direct speech→text translation, one step, 25 EU langs incl SK) is the
most promising fix for SK→EN: it collapses the STT+MT cascade into a single step. Untested
on SK — needs GPU validation (Kaggle T4). This is the pivot candidate.

---

## 5. MT (translation)

| Model | License | SK pair? | Commercial-safe? | Notes |
|-------|---------|----------|------------------|-------|
| Opus-MT (CT2 int8) | per-model CC-BY-4.0 | ✅ sk-en/en-sk | ✅ | Current default; speed crown (~0.08-0.11 s/sentence) |
| NLLB-200-600M | CC-BY-NC | ✅ | ❌ | Quality challenger; blocked commercially |
| MADLAD-400-3B-CT2 | Apache-2.0 | ✅ 400+ langs | ✅ | Quality challenger (BLEU 36.0); 3B on T4 questionable |
| Qwen3.8-LiveTranslate | proprietary | ⚠️ | ❌ | Cloud API |

---

## 6. TTS within 8 GB VRAM (T4 budget)

| Model | License | SK? | VRAM (T4) | RTF | Cloning |
|-------|---------|-----|-----------|-----|---------|
| Piper (CPU) | MIT | ✅ | ~0.3 GB | 0.04 | Fine-tune (owner voice) |
| Kokoro 82M | Apache 2.0 | ❌ EN only | 1.33 GB | 0.03 | No (54 preset voices) — EN output only |
| Chatterbox Multilingual | MIT | ❌ | 4-8 GB | ~1-2x | 5-7 s zero-shot |
| Qwen3-TTS 0.6B | Apache 2.0 | ❌ | 4 GB | 97 ms first-packet | 3 s zero-shot |
| XTTS v2 | CPML (non-comm) | ❌ | 4-6 GB | 0.24 | 6 s zero-shot |
| OmniVoice | Apache 2.0 | cross-lingual | 5.8 GB @12 steps | 0.14-0.34 | Zero-shot clone |
| SeamlessM4T v2 | CC-BY-NC | ✅ S2S | ~6 GB | 0.77-0.99 | Prosody, not identity |

---

## 7. Where the project sits vs. the market

**Our moat:** Slovak is not covered by Teams or Google Meet speech translation. Local
EN↔SK that "just works" on a T4-class server (or laptop) with no per-minute fee is a real
differentiator — for education, SMEs, and any organization that can't/shouldn't send audio
to the cloud (GDPR, privacy, cost).

**Our gap:** "your own voice" in Slovak output. No open ZS-TTS supports SK. Piper fine-tune
is the only path; it's CPU-fast (RTF 0.04) but takes 4-8 h to train on a Mac and the voice
quality has been kill-graded for trembling (owner 2026-10-01). Zero-shot cloning into SK
doesn't exist yet in open source.

---

## 8. Recommended engine orientation (T4 / ≤8 GB VRAM)

| Stage | Baseline (CPU, any machine) | Optional GPU engine (T4/RTX) | Status |
|-------|-----------------------------|------------------------------|--------|
| STT (EN) | faster-whisper `base` int8 | Parakeet EN / Qwen3-ASR | verified |
| STT (SK) | whisper-small-sk int8 | Canary AST (SK→EN, collapses STT+MT) | experimental — WER 0.2143 > 0.2 gate |
| MT | Opus-MT CT2 int8 | MADLAD-3B (CUDA) | verified |
| TTS (SK) | Piper (owner fine-tune) | SeamlessM4T v2 S2S (prosody) | Piper verified; SK clone = fine-tune only |
| TTS (EN) | Piper | Kokoro 82M (RTF 0.03, 1.33 GB) | Kokoro = strong GPU EN option |
| S2S (end-to-end) | — | SeamlessM4T v2 (CC-BY-NC) | quality reference, not live |

**Do not** make Canary/Kokoro baseline until validated on real mic (WER < 0.2 for SK STT,
ear grade for TTS). Register them as optional GPU engines behind the existing registry
(fail-closed licence gate).
