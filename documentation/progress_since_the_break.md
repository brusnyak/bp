# Progress since the break — what is demonstrably better than the spring state (2026-09-28)

Why this file exists: the thesis work was picked up again in August after a dormant
stretch, so the handler will ask "what did you actually add?". This is the answer with a
source per claim, plus the honest list of what did **not** improve. Nothing here is
estimated; every number is from a recorded measurement in this repo or from today's runs.

## The timeline, from the commit history (not from memory)

| Month | Commits | State |
|---|---|---|
| 2025-10 → 2025-12 | 1 + 4 + 5 | first working pipeline: Piper + UI + VAD/whisper/MT plumbing |
| 2026-02 | 1 | — |
| 2026-04 | 8 | the "spring state": single-shot pipeline, mock auth, no captions, XTTS/F5 experiments |
| 2026-05 → 2026-06 | **0** | dormant |
| 2026-07 | 1 | restart |
| 2026-08 | 25 | real-time rearchitecture, hybrid cloning, concurrency + accuracy measurement |
| 2026-09 | 53 | voices (personal EN/SK), Voice Lab, PWA, auth, captions, cross-platform |

## Stage by stage: then vs now

| Stage | Spring 2026 | Now (measured) | Source |
|---|---|---|---|
| EN STT WER | 0.286 (base, short phrase) | **0.077** base on a 115 s reading; **0.023** Parakeet-TDT-v3 (adopt path) | `guide.md` Phase 0.1 vs `processed/stt_matrix.json`, `processed/pipeline_latency.json` |
| SK STT WER | not measured | **0.41** turbo (adopted default), 0.49 small, 0.63 plain base, 1.00 with `auto` | `processed/stt_matrix.json` |
| STT latency | 7.9 s for one short clip | 0.5–0.6 s for a 115–134 s clip (batch); live segment 0.51 s | `guide.md` vs `stt_matrix.json`, rehearsal |
| MT latency | 0.93 s per sentence | **0.50 s for 18 sentences** (chunked, 3.5× faster) | `guide.md` vs `PLAN.md` |
| MT quality | BLEU 29.9 / METEOR 0.50 EN→SK | unchanged model (Opus-MT int8) — quality same, latency better | `guide.md` |
| TTS | XTTS only, RTF ~2.8; Piper generic | **personal SK voice RTF ~0.05**; hybrid (Piper+OpenVoice) RTF **0.123** measured today: 6.66 s of compute for 54.2 s of audio | `TTS_COMPARISON_REPORT.md` vs today's `processed/e2e_ensk.json` |
| End-to-end EN→SK | minutes; first version was not usable live | **18.1 s wall for 53 s of speech** offline (STT 0.52 + MT 1.84 + TTS 6.66); **0.78 s** per live sentence (STT 0.51 / MT 0.10 / TTS 0.17) | `processed/e2e_ensk.json` (fresh), today's WS rehearsal |
| Blocking behaviour | capture froze ~4 s during every translation, no interruption handling | capture never stops; a new utterance cancels the in-flight one within ~1 ms | `documentation/realtime_pipeline_rearchitecture_2026-08-22.md` |
| Streaming STT | not in the pipeline | `caption_partial` captions render while speaking | spec 002, `backend/stt/streaming_captions.py` |
| Auth | mock tokens | Google GIS + real HS256 JWT, verified live | `PLAN.md`, `/api/auth/google` |
| Voice identity | generic Piper / slow XTTS | own fine-tuned **SK male voice** shipped as a Piper voice; in-browser enrollment + registered profiles | `backend/tts/piper_models/sk_SK-personal-male-medium.onnx` |
| Voice QC | none | machine-listening panel (WER thirds, F0 jitter, HNR, clipping) + Praat-style clinical metrics + A/B listening lab | `scripts/machine_listen_qc.py`, `ui/voice-lab/` |
| Platform | macOS dev box only | macOS (M1 Pro) + **Windows 11 CPU-only verified end-to-end** (16/16 tests, e2e ≈3.6 s); Linux documented, not yet run | `SETUP_WINDOWS.md`, `documentation/linux_setup_and_test.md` |
| Distribution | run from source | installable PWA (manifest, maskable icons, offline shell) | `ui/manifest.webmanifest`, `ui/sw.js` |

## What did NOT improve — say these out loud first

| Limit | Number | Why it is still open |
|---|---|---|
| **SK speech recognition is the bottleneck** | WER 0.41 (turbo), 0.49 (small), 0.63 (plain), 1.00 (`auto` on SK) | the only local SK-capable STT is whisper; Parakeet-TDT-v3 scores 0.89 on SK (no language conditioning) |
| **SK through XTTS is a Czech proxy, not Slovak** | XTTS-v2's tokenizer language codes contain `cs`, **not `sk`** (checked in the installed package, 2026-09-28); the repo's own verdict: "XTTS (no SK)" | zero-shot XTTS therefore reads Slovak with Czech phonetics — audible accent; earlier measure: F0 86 Hz, accent, RTF 2.7, rejected as the SK default |
| **Personal voice timbre still "kinda off"** | personal SK F0 spread 11–13 Hz vs 22 Hz for generic lili → the "robotic" impression, quantified | under-training (Piper docs want ~1000 epochs; the shipped run was far fewer) + 18 sentence clips of data |
| **XTTS is slow** | RTF ~1.4 live, ~2.7–2.8 in older runs | architectural (CPU zero-shot), not tunable |
| **Hybrid (Piper + OpenVoice) adds a converter step** | RTF 0.123 today (6.66 s compute / 54.2 s audio) vs 0.05 for plain Piper | acceptable trade for cross-lingual cloning, still 8× real time |
| **The dormancy itself** | 0 commits in May–June 2026 | state it plainly; the September push is what closes it |

## The SK→EN issue: where the damage actually enters

Complaint: "Slovak to English translation is broken." Measured today, the failing part is
**not the translator**:

1. **Translator alone, on proofread Slovak** (`Helsinki-NLP/opus-mt-sk-en`, int8, via
   `scripts/live_direction_probe.py` and a ground-truth probe): 6/6 sentences came out
   accurate and natural, e.g. *"Dobré ráno a vitajte v tejto živej ukážke prekladu reči v
   reálnom čase."* → *"Good morning and welcome to this vivid demonstration of real-time
   translation of speech."* 6 sentences in **0.72 s**.
2. **Whole-clip transcript → EN** (1 263 chars): still coherent English, minor word-choice
   slips ("prepísaná" → "translated" instead of "transcribed", "Titulky" → "Titles").
3. **Earlier live artifact that looked like a translator bug** —
   `processed/pipeline_latency.json` `mt_sk_en` sample *"Good morning, and the countryside
   in this slick, all over in real time."* — came from a **short live SK segment**, i.e. an
   STT/VAD-side input, not from clean text.

Conclusion to hand the Windows work: fix the **SK STT / segmenting** side (model rung,
VAD cuts, short-segment context), and separately guard MT against garbage input; the
sk→en translation model itself is sound. `scripts/live_direction_probe.py` makes this
repeatable in one command and now reports **WER against the proofread reference column**.

## Demo mapping — which claim to show where

| Claim | Show | Time |
|---|---|---|
| live EN→SK in my own voice, <1 s of compute per sentence | live page, one sentence, latency chart | 60 s |
| barge-in | speak over the playing translation | 20 s |
| captions + floating PiP | subtitle strip, pop out over a window | 30 s |
| voice candidates with numbers | `lab.html` A/B (personal male SK vs generic lili) | 60 s |
| "it is all local, no cloud" | `make demo-check`, `/api/voice-lab/status` engines + backends | 30 s |
| "it runs on Windows too" | the Windows laptop live, or the 16/16 + 3.6 s e2e evidence | 30 s |
| what is still weak, honestly | SK STT 0.41 table + this file's limitations table | 60 s |
| the fallback ladder if the mic dies | runbook §5 (`processed/e2e_ensk_sk.wav`) | — |
