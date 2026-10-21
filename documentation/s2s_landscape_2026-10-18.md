# Real-time speech-to-speech: what was, what now — landscape 2026-10-18

Purpose: decide what this project should build on the open-source base, and record
honestly how much of the original idea the industry has absorbed since 2023.
Every external claim carries a source and a date. Labels: `MEASURED` = measured in
this repo; `EXTERNAL` = a vendor/researcher claim, not verified here; `APPROX` = order
of magnitude only.

Companion files: `model_landscape_2026-10.md` (per-component STT/MT/TTS verdicts),
`load_tests_2026-10.md` (this project's measured capacity), `specs/003-*/spec.md`
(the build plan this report feeds).

---

## 1. The one-line history

In 2023 this was a research project. In 2026 it is a product feature and an API call.
Speech-to-speech translation moved from "a cascade you assemble yourself, ~3 s latency,
no Slovak TTS" to "one streaming API, sub-second, voice-preserving, 70+ languages
including Slovak". The engineering problem this thesis solves is now a configuration
of someone else's model, unless the value is deliberately elsewhere (local-only,
voice identity, or a language pair nobody else serves well).

---

## 2. Timeline: what changed, when

| Date | Event | Why it matters | Source |
|---|---|---|---|
| Aug 2023 | Meta SeamlessM4T v1 — one model, ASR + S2TT + S2ST + T2ST, ~100 languages | First credible unified speech translation model, open weights | ai.meta.com/blog/seamless-m4t (2023-08-22) |
| Dec 2023 | SeamlessM4T v2 + SeamlessStreaming + SeamlessExpressive | Streams at ~2 s latency, near-offline accuracy, preserves prosody/style | arxiv.org/abs/2312.05187 (2023-12) |
| 2023 | Whisper-Streaming (CUNI/UFAL) — LocalAgreement policy, ~3.3 s latency | Made Whisper usable streaming; the base most OSS live-ASR projects copied | arxiv (Macháček 2023); github.com/ufal/whisper_streaming |
| Sep 2025 | Qwen3-Omni (Apache 2.0) — Thinker-Talker MoE, 234 ms first-packet, S2ST | Open weights reached streaming-quality S2ST | arxiv.org/abs/2509.17765 (2025-09-22) |
| Jul 2025 | Step-Audio 2 | Open S2ST contender on CoVoST2 | arxiv.org/html/2507.16632 |
| Sep 2025 | Azure Live Interpreter (public preview) — 76 input languages, voice preservation, BYO personal voice | Enterprise S2S with your own cloned voice, from an API | techcommunity.microsoft.com (2025-09-12) |
| Nov 2025 | SimulStreaming (CUNI, MIT) — AlignAtt policy + EuroLLM cascade, Docker, server mode | The successor to Whisper-Streaming; explicitly built as a live service | github.com/mmaudet/SimulStreaming (created 2025-11-16) |
| Jun 2026 | Gemini 3.5 Live Translate — S2S, 70+ langs incl. Slovak, preserves intonation/pacing/pitch, rolling into Google Meet | Speech-to-speech becomes a platform feature, SK included | ai.google.dev/gemini-api/docs/live-api/live-translate (2026); forasoft.com/blog (2026-10-11) |
| Jul 2026 | Zoom Voice Translator for Meetings (EN/ZH/ES/FR/JA at launch) | Meeting platforms ship spoken translation directly | zoom.com/en/blog/voice-translation-zoom (2026-07-22) |
| 2026 | Voxtral Realtime (Mistral) — natively streaming ASR, sub-second, end-to-end trained for streaming | Open-weight streaming ASR that beats chunked-offline approaches | arxiv.org/abs/2602.11298 |
| Sep 2026 | WhisperLiveKit: AlignAtt + NLLW (200-language simultaneous MT) + Streaming Sortformer diarization, pip-installable | A self-hostable, near-turnkey live STT+S2TT server exists | github.com/QuentinFuxa/WhisperLiveKit |

---

## 3. Two architectures, and which one won

- **Cascade** (what BP is): VAD → STT → MT → TTS. Each stage swappable, each stage a
  separate model, errors compound, latency adds.
- **Unified / end-to-end** (what the industry moved to): one model ingests audio and
  emits translated audio. Gemini 3.5 Live Translate, Azure Live Interpreter,
  Qwen3-Omni, Voxtral, SeamlessStreaming. Lower latency, prosody preserved, no
  per-pair MT model to source.

The unified path did not make the cascade obsolete for a **thesis**: the cascade is
inspectable, measurable per stage, swappable, and runs CPU-only. That is the academic
contribution a thesis can defend. The unified path is what a product would use, and
using it would delete the thesis.

---

## 4. Tool comparison — managed APIs

| Tool | Type | Languages | Latency (vendor claim) | Voice preservation | Slovak | Pricing | Notes |
|---|---|---|---|---|---|---|---|
| Gemini 3.5 Live Translate | S2S, managed | 70+ | not stated; Live API is sub-second | Yes (intonation/pacing/pitch) | **Yes** (sk listed) | Live API token pricing | `live-translate-preview`; audio-in only; rolling into Google Meet. `EXTERNAL` |
| Azure Live Interpreter | S2S, managed | 76 input | 500–900 ms first chunk | Yes, incl. **BYO personal voice** | not confirmed for voice-out in this doc | ~$2.50/audio-hour (2 targets) + TTS chars | Enterprise SLA, EU data boundary. `EXTERNAL` |
| DeepL Voice | S2S + subtitles | 40+ | not stated | Speaker match for its own voices | Subtitles yes; voice-to-voice via third party, no speaker match | Enterprise | ISO 27001, no audio retention. `EXTERNAL` |
| Palabra.ai | S2S + API | 60+ | <1 s | Instant voice cloning | not confirmed | from free trial / API | Self-host/on-prem option, SOC2/HIPAA claims. `EXTERNAL` |
| OpenAI Realtime / GPT-Live | S2S, managed | n/a here | <500 ms | varies | not confirmed | per-token | Not evaluated for SK. `EXTERNAL` |
| Teams Interpreter | Meeting-native S2S | 9 (EN,ES,PT,JP,ZH,IT,DE,FR,KO) | n/a | No | **No** | M365 Copilot, 20 h/user/mo | Captions same 9 languages. `EXTERNAL` |
| Zoom Voice Translator | Meeting-native S2S | 5 at launch (EN,ZH,ES,FR,JA) | n/a | Planned | **No** | Zoom plan | Captions cover 46 langs incl. SK, captions only. `EXTERNAL` |
| Google Meet translated captions | Meeting-native captions | ~70 → ~100 | n/a | n/a | **Yes (captions only)** | Business Std $14/user/mo | No spoken translation back. `EXTERNAL` |

## 5. Tool comparison — open weights / reusable OSS

| Project / model | License | What it is | Reuse verdict for BP |
|---|---|---|---|
| **SimulStreaming** (CUNI) | MIT | AlignAtt streaming Whisper + EuroLLM MT cascade, server + Docker | **Study / adopt pattern.** Closest peer to a live S2ST server. Its AlignAtt policy is a real upgrade over our chunk-and-wait VAD. |
| **WhisperLiveKit** | see repo | AlignAtt + NLLW simultaneous MT (200 langs) + Sortformer diarization, `pip install whisperlivekit`, browser UI | **Study / benchmark.** Fastest path to a self-hosted, multi-language live server. |
| WhisperLive (Collabora) | MIT | Nearly-live Whisper server, diarization, translation thread, Docker | Reference for the WS server shape. |
| whisper_streaming (UFAL) | MIT | The 2023 original, LocalAgreement | Superseded by SimulStreaming (README says so, 2025). |
| Qwen3-Omni-30B-A3B | Apache 2.0 | Unified omni, S2ST, 211 ms audio-only | Candidate unified engine; **speech generation only in ~10 languages**, verify SK before claiming. |
| Voxtral Realtime (Mistral) | see model card | Natively streaming ASR, sub-second | Candidate fast streaming STT. |
| Seamless / SeamlessM4T v2 | **CC-BY-NC 4.0** | Unified S2ST, ~2 s streaming | **Non-commercial license.** Cannot ship in a product; usable as a quality reference only. |
| Step-Audio 2 | see repo | Open S2ST | Watch list. |
| Piper | MIT | Fast local TTS, per-language voices | **Keep.** Still the only credible CPU SK TTS base. |
| Opus-MT (Helsinki) | MIT-ish (CC-BY / Apache per model) | Per-pair MT, CTranslate2 int8 | **Keep.** Any-pair expansion path (a model per direction). |
| NLLB-200 / NLLW | CC-BY-NC (NLLB) | 200-language MT | License trap; NLLW is the distilled streaming variant used by WhisperLiveKit. |

---

## 6. Slovak specifically

- **Spoken translation with voice**, Slovak, is served by: Gemini 3.5 Live Translate
  (SK listed, `EXTERNAL`), DeepL Voice (SK subtitles + third-party voice), and event
  vendors (Interpretwise lists Slovak in its live catalogue, `EXTERNAL`).
- **Captions only**, Slovak, free-ish: Google Meet translated captions, Zoom AI
  Companion.
- **Not served**: Teams Interpreter and Zoom Voice Translator (no Slovak).
- So the "does anyone do EN↔SK speech-to-speech" question is answered: yes, since
  2026-06 in Google's case. The gap the project can own is not *existence*, it is
  *locality* (no cloud), *identity* (your own voice), and *cost* (no per-minute fee).

---

## 7. Latency and cost anchors

| Path | First-output latency | Cost anchor | Source |
|---|---|---|---|
| Live captions (ASR+MT) | 300–800 ms | — | forasoft.com/blog (2026-10-11), `EXTERNAL` |
| Cascaded S2S (cloud) | 600–1000 ms | Azure speech translation ~$2.50/audio-hour | forasoft; Azure pricing, `EXTERNAL` |
| Gemini 3.5 Live Translate | sub-second (Live API) | token pricing | Google, `EXTERNAL` |
| Azure Live Interpreter | 500–900 ms | per-hour + per-char | Microsoft, `EXTERNAL` |
| **BP cascade, EN→SK, M1 Pro** | ~0.67–0.87 s turn; first audio ~0.5 s in the streaming meeting sim | $0 (own hardware) | `MEASURED`, PLAN.md 2026-09-29 |
| **BP cascade, SK→EN, M1 Pro** | ~1.27–1.64 s turn | $0 | `MEASURED` |
| **BP on one T4, cloned voice** | first audio ~1.1 s; p95 2.0–2.6 s under load | one T4-hour per concurrent ~2 speakers or ~8–12 meeting participants | `MEASURED`, load_tests_2026-10.md |
| On-prem appliance (Isermo, education) | not stated | **$8,999 one-time**, no subscription | einnews.com PR (2026-09-22), `EXTERNAL` |
| On-prem appliance (Ubestream AIspeakin RL) | not stated | from NT$250,000 (~$7.8k) | shop.ubestream.com, `EXTERNAL` |

BP's cachet: its own numbers are in the same latency band as the cloud APIs and its
marginal cost is zero. Its weakness: the cloud APIs are more accurate on spontaneous
speech and support far more languages out of one model.

---

## 8. Licensing traps (check before any commercial claim)

- **Seamless / SeamlessM4T / SeamlessStreaming: CC-BY-NC 4.0** — non-commercial only.
- **NLLB-200: CC-BY-NC 4.0** — non-commercial.
- **Qwen3-Omni: Apache 2.0** — usable.
- **Opus-MT: per-model (mostly CC-BY-4.0 / MIT)** — verify per pair.
- **Piper: MIT** — usable.
- **SimulStreaming: MIT** — usable, attribute.
- The repo itself has **no LICENSE file** — "all rights reserved" by default; resolve
  before any shared/hosted use.

---

## 9. What this changes for the project

1. **Stop positioning as "a product".** Keep the cascade as the thesis contribution:
   per-stage measurement, CPU-only, swappable, reproducible on a laptop.
2. **Make the pipeline engine-agnostic** so a unified model (Qwen3-Omni local,
   Gemini/Azure via an optional adapter) can be dropped in as a stage, not a rewrite.
   That is what the supervisor's "collaborate on our custom system" needs.
3. **Adopt the OSS pattern, not the OSS repo, where it costs little:** AlignAtt-style
   streaming is the one upgrade worth taking from SimulStreaming/WhisperLiveKit.
4. **Guard the licence:** never wire CC-BY-NC models (Seamless, NLLB) into a shipped
   path; keep them as offline quality references.
5. **Answer the multi-language question once, structurally:** Opus-MT gives a model per
   direction; a language is "supported" when its STT + MT + TTS exist and a test passes.
   See `specs/003-*/spec.md`.

---

## 10. Component swap ladder (what to try next, per stage)

| Stage | Current | Next candidates | Gate |
|---|---|---|---|
| VAD | WebRTC VAD + RMS | Silero VAD; VAC iterator; AlignAtt policy | Latency + WER unchanged or better on the SK clips |
| STT | faster-whisper small-sk / base | Voxtral Realtime, Qwen3-ASR-causal, Parakeet-TDT-v3 (EN) | WER on spontaneous SK + RTF |
| MT | Opus-MT int8 per pair | NLLW / EuroLLM / a small LLM translate step | chrF/BLEU + latency + licence |
| TTS | Piper (CPU) | OmniVoice, CosyVoice3, Qwen3-TTS, Voxtral TTS | RTF + similarity + licence |
| Whole | cascade | Qwen3-Omni local, Gemini Live Translate adapter | Only if the thesis framing can absorb it |

---

## 11. Sources

- Meta Seamless / SeamlessM4T / SeamlessStreaming — ai.meta.com (2023–2024); arxiv 2312.05187.
- Whisper-Streaming / SimulStreaming — github.com/ufal/whisper_streaming; github.com/mmaudet/SimulStreaming (MIT, 2025-11-16).
- WhisperLiveKit — github.com/QuentinFuxa/WhisperLiveKit; WhisperLive — github.com/collabora/WhisperLive (MIT).
- Qwen3-Omni — arxiv.org/abs/2509.17765; huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking (Apache 2.0).
- Voxtral Realtime — arxiv.org/abs/2602.11298.
- Gemini Live / Live Translate — ai.google.dev/gemini-api/docs/live-api/live-translate; cloud.google.com/vertex-ai/.../live-api; forasoft.com/blog (2026-10-11).
- Azure Live Interpreter / Voice Live — learn.microsoft.com/azure/ai-services/speech-service/speech-translation, /voice-live; techcommunity.microsoft.com (2025-09-12); Azure pricing page.
- DeepL Voice languages — support.deepl.com/hc/en-us/articles/26625846174364; deepl.com/en/products/voice.
- Palabra.ai — palabra.ai (2026).
- Teams / Zoom / Meet native — support.microsoft.com (multilingual speech recognition, 2026); zoom.com/en/blog/voice-translation-zoom (2026-07-22); support.google.com/meet/answer/10964115; support.google.com/meet/answer/9300310.
- On-prem appliances — einnews.com PR 944306325 (Isermo, 2026-09-22); shop.ubestream.com/en/pages/products.
- This repo's own measurements — PLAN.md, documentation/load_tests_2026-10.md, documentation/model_landscape_2026-10.md.
