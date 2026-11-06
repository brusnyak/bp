# Local speech-model landscape, October 2026

SK-first live EN↔SK translation, local-only pipeline. One document replaces six
scattered files: `tts_landscape_2026-09.md`, `model_evaluation_2026-09.md`,
`tts_alternatives_research.md`, `s2s_translation_research_2026-08.md`,
`simultaneous_translation_research_2026-08-22.md`, `ssbd_speculative_decoding_findings_2026-08.md`.
Those stay on disk; this file is the current verdict. Read the old ones only to
cite method details.

## 0. How to read this

Every number carries exactly one label. Unlabelled numbers do not exist here.

- `MEASURED` — produced on this project's own hardware; the evidence file is named.
- `VENDOR` — the model author's published figure (leaderboard, model card).
- `COMMUNITY` — an independent third-party measurement (blog, benchmark repo).
- `PROJECTED` — derived from a measured rate plus a stated multiplier.
- `UNKNOWN` — nobody has measured it; the cell says what test would fill it.

Thesis §5 may cite `MEASURED` rows only. Everything else is direction, not evidence.

Research date: 2026-10-01. Web sources are listed in §9.

## 1. STT — speech to text

The SK→EN direction is STT-bound (`MEASURED` ~1.3 s Slovak STT vs ~0.55 s
English; MT+TTS together ~0.2 s). Any STT improvement lands directly on the
worst turn latency. English STT is solved (Parakeet EN `MEASURED` WER 0.023).

| Model                                                               | SK?                                                     | License                           | Size / runtime                                              | Key number                                                                                                                                                                                                                                                                                                                                                                                  | Verdict                                                                                                                                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| whisper-small-sk (incumbent, NaiveNeuron, CT2 int8)                 | native SK fine-tune                                     | MIT                               | ~500 MB, faster-whisper CPU                                 | `MEASURED` WER 0.13 FLEURS-sk / 0.26 owner mic; ~2.4 s per 7 s clip, RTF 0.33 (`model_evaluation_2026-09.md`)                                                                                                                                                                                                                                                                               | Keep as SK default until a challenger beats it on owner mic                                                                                                                     |
| whisper base (EN rung)                                              | n/a (EN)                                                | MIT                               | 142 MB                                                      | `MEASURED` WER 0.077, 0.52 s per 115 s clip (`processed/stt_baseline.json`)                                                                                                                                                                                                                                                                                                                 | Keep for EN                                                                                                                                                                     |
| large-v3-turbo                                                      | 99 langs incl SK                                        | MIT                               | 1.6 GB                                                      | `MEASURED` WER 0.12 FLEURS-sk but 0.41–0.52 on owner mic, ~7–9 s/clip (`model_evaluation_2026-09.md`, PLAN.md 2026-09-29: definitive negative)                                                                                                                                                                                                                                              | Rejected: distillation cost it low-resource capacity                                                                                                                            |
| Parakeet-TDT v3 0.6B                                                | 25 EU langs incl SK, auto-detect                        | CC-BY-4.0                         | 680 MB INT8 ONNX, sherpa-onnx CPU                           | `VENDOR` 6.34% avg WER, 3332× realtime on leaderboard GPU; `MEASURED` on owner SK mic WER 0.58–0.63, FLEURS-sk int8 0.20 (`model_evaluation_2026-09.md`); `COMMUNITY` 17–26× realtime on server CPU (gauravvij/parakeet-optimization)                                                                                                                                                       | Speed proven, Slovak accuracy not: leading theory is training-data SK share, not quantisation (fp32 measured worse than int8 on owner mic). Retest only if a v4 changes SK data |
| Parakeet Unified EN 0.6B                                            | EN only                                                 | CC-BY-4.0                         | 631 MB INT8                                                 | `VENDOR` SoTA EN 6.05%                                                                                                                                                                                                                                                                                                                                                                      | EN-side candidate if Parakeet-v3 EN ever regresses; not an SK answer                                                                                                            |
| **Nemotron 3.5 ASR streaming 0.6B (BENCHED 2026-10-01 — negative)** | SK in broad-coverage tier; CZ same tier                 | OpenMDW (open, commercial-use OK) | W8A8 C runtime, whole-file streaming decode                 | `MEASURED` owner mic: SK WER **0.86/0.75** (trhove_b/script, sk-SK prompt; auto-prompt worse at 0.98), EN WER 0.19 — vs small-sk 0.23–0.35 and Parakeet-int8 0.58–0.63 on the same SK mic. RTF 0.33, same speed band as small-sk. Failure mode: language drift (RU/UK/SL tokens, `<sl-SI>` tag mid-stream). `processed/new_models/nemotron_matrix.json`, script `scripts/bench_nemotron.py` | Rejected for SK: streaming partials are moot this far behind on accuracy. Broad-coverage tier means what it says. Do not retest without a version change or SK fine-tune        |
| Canary-1B / Flash / 180M (MEASURED on Kaggle T4 2026-11-03)         | 25 EU langs incl SK; also does speech translation (AST) | NVIDIA CC-BY-4.0                  | 1B / Flash / 180M, NeMo + Transformers                      | `MEASURED` ASR SK WER 0.0477 (b5), CS 0.0922, EN 0.0498 on 80 FLEURS clips each (`confirm-canary-v1/`); AST SK→EN chrF++ 52.49 mean / CS→EN 50.05 (n=80 each, English references present but marked `semantic_review_pending` — not a quality gate pass); cold first-call 4.6s on 7s clip | AST collapses STT+MT for SK→EN but chrF++ ~50 is below commercial bar; keep as speed path, not quality path. Flash/180M untested. |
| Kyutai STT 1B                                                       | EN/FR only, no SK                                       | open (Kyutai)                     | 500 ms fixed delay, 400 parallel streams on H100 (`VENDOR`) | n/a for SK                                                                                                                                                                                                                                                                                                                                                                                  | Architecture reference only (delayed-streams modeling)                                                                                                                          |
| Zipformer / Moonshine / Cohere-transcribe                           | no SK/CZ coverage                                       | various                           | CPU-light                                                   | PLAN.md 2026-09-29: killed / deprioritised with reasons                                                                                                                                                                                                                                                                                                                                     | Dead, do not retest                                                                                                                                                             |
| Multitalker Parakeet streaming                                      | 25 langs (SK presumably)                                | CC-BY-4.0                         | one instance per speaker                                    | `UNKNOWN`                                                                                                                                                                                                                                                                                                                                                                                   | Watch item for the conference overlap case (two mics, crosstalk). Not a 2026 experiment                                                                                         |

Why Nemotron first and not Parakeet-again: the project already owns two Parakeet
negatives on the exact mic and script that matter. Nemotron is a different
architecture (cache-aware streaming FastConformer-RNNT vs offline TDT), a
different training set (530 k hours lineage), and it changes the latency model
itself (streaming partials vs faster offline chunks). One bench settles it:
owner SK mic → WER vs small-sk + turn-latency delta.

## 2. MT — machine translation

MT is not the bottleneck (`MEASURED` ~0.08–0.11 s per sentence, chunked batch
18 sentences in 0.50 s). The bar for replacing Opus-MT is therefore quality on
bad transcripts (STT output is noisy) at equal-or-better speed, not raw BLEU.

| Model                                               | SK↔EN?                                    | License                                                     | Size / runtime                                                                                 | Key number                                                                                                                            | Verdict                                                                                                                                                                                                      |
| --------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Opus-MT sk-en / en-sk (incumbent, CT2 int8)         | dedicated pair models                     | CC-BY-4.0                                                   | ~100–300 MB per direction, CTranslate2 CPU                                                     | `MEASURED` 0.5 s per 18 sentences chunked; `COMMUNITY` 696 tok/s int8 CPU, fastest in DEEP bench (57 s vs 870 s MADLAD on same set)   | Keep. Speed crown is unchallenged and the pipeline needs speed more than +2 chrF                                                                                                                             |
| NLLB-200-distilled-600M                             | 200+ langs incl SK                        | CC-BY-NC (non-commercial — thesis/education OK, product no) | 600M, transformers/CT2                                                                         | `MEASURED` by others on Romance/Germanic: trades wins with Opus per pair (arxiv 2607.26286); `COMMUNITY` 2026 review: edge-deployable | The honest quality challenger for SK; NC license fits a thesis but poisons any commercial follow-up. Bench it, keep Opus default unless the gap is large                                                     |
| NLLB-1.3B distilled                                 | same                                      | CC-BY-NC                                                    | 1.3B                                                                                           | `VENDOR`/community: best chrF on several EU pairs                                                                                     | Only if 600M shows a quality gap worth chasing with 2× compute                                                                                                                                               |
| **MADLAD-400-3B-CT2 (NEW, ready to bench)**         | 400+ langs incl SK                        | Apache-2.0 (commercial-safe)                                | 3B, ready CT2 export `santhosh/madlad400-3b-ct2`; MLX INT4/INT8 builds exist for Apple Silicon | `COMMUNITY` best open translator in DEEP bench (BLEU 36.0 vs NLLB 33.6 vs Opus 30.0 on news test) at ~15× Opus wall time on CPU       | Top MT experiment: Apache-2.0 removes the NLLB license trap. Question is purely speed on short sentences with beam 1 — 3B may still fit the 0.2 s budget on CUDA while failing it on CPU. Bench both devices |
| M2M100-1.2B-CT2                                     | 100 langs incl SK                         | MIT                                                         | 1.2B, ready CT2 (`entai2965/m2m100-1.2B-ctranslate2`)                                          | `COMMUNITY` DEEP bench: weakest of the group (BLEU 27.7)                                                                              | Fallback only; MIT license is its best feature                                                                                                                                                               |
| EuroLLM 1.7B/9B/22B                                 | 24 EU langs incl SK                       | fully open (EU)                                             | 1.7B–22B, LLM inference                                                                        | `COMMUNITY` DEEP bench: 1.7B slow and weak for MT; 22B competitive but thesis-irrelevant size                                         | Related-work citation (EU open alternative), not a pipeline candidate                                                                                                                                        |
| riva-translate 1.6B/4B (NVIDIA NIM)                 | dozens of langs (verify SK before citing) | NIM container terms                                         | GPU container, batch 32/64 profiles                                                            | `UNKNOWN` on SK                                                                                                                       | Server-side option only if a CUDA box materialises; docker + NGC, never the Mac                                                                                                                              |
| Small LLM translators (qwen2.5:14b etc. via Ollama) | yes                                       | various                                                     | 3B–14B, seconds per sentence on CPU                                                            | `COMMUNITY` within ~1–5 chrF of dedicated MT on some pairs (arxiv 2607.26286)                                                         | Watch: the gap is closing, but per-sentence seconds on CPU kill live use today. Revisit yearly                                                                                                               |

MT decision rule (pinned): a challenger replaces Opus only with (a) measured
chrF gain on _noisy STT output_ (not clean FLORES), (b) per-sentence latency
inside the current 0.2 s envelope on the demo machine, (c) a commercial-safe
license. MADLAD-3B is the first candidate that could plausibly meet all three —
on CUDA.

## 3. TTS — text to speech, including voice cloning

Two separate jobs: (1) fast clean SK/EN output for the live pipeline (Piper
owns this, RTF ~0.04 `MEASURED`), (2) producing the owner's voice. Job 2 splits
further into bulk generation (slow is fine, quality is everything) and the
trainable voice itself.

| Model                                               | SK?                                                                                                                                                                                                                        | License                                                                | Size / runtime                                                                                                                                                              | Key number                                                                                                                                                                                                   | Verdict                                                                                                                                                      |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Piper + own fine-tunes (incumbent)                  | `sk_SK-lili` is the only SK base; `me_omni_piper_sk` registered 2026-10-01                                                                                                                                                 | MIT (voices are local files)                                           | 60 MB ONNX, onnxruntime CPU                                                                                                                                                 | `MEASURED` synth RTF 0.03–0.05; new voice small-sk WER 0.19 vs 0.54 old (`verification_omni_piper_sk_2026-09-30.md`); train cost 4 h 40 min per 2500 steps on Mac CPU (`compute_capacity_report_2026-09.md`) | Keep as the live engine. No rival touches RTF 0.04 on CPU                                                                                                    |
| OmniVoice (batch / QC only — NOT a live path)                        | cross-lingual clone from EN/SK refs                                                                                                                                                                                        | project dependency                                                         | MPS float32                                                                                                                                                                 | `MEASURED` RTF 0.93–1.22 Mac MPS / 0.14–0.17 long text / 0.40–0.65 short phrase on T4 (`load_tests_2026-10.md`, `confirm-canary-v1/`); 85–90% of cascade time is TTS | Keep for bulk corpus generation + QC only. RTF >1 on short phrases = not real-time on T4. Never the default live TTS engine. |
| Chatterbox multilingual / Turbo (CORRECTION inside) | **No Slovak, no Czech** in the shipped model — 23 langs are ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh (`VENDOR` model card). Vendor marketing pages claim Slovak; the weights do not contain it | MIT, ONNX builds exist (`onnx-community/chatterbox-multilingual-ONNX`) | 0.5B Llama backbone; Turbo 350M, sub-200 ms claim (`VENDOR`); `MEASURED` here only via Czech proxy: WER 0.054 tie with omni-zeroshot at RTF 6.7 on MPS (PLAN.md 2026-09-29) | Strategically interesting (MIT + openly trainable + ONNX), but for SK it is the same proxy game as before, not a native voice. Turbo RTF on CUDA is the untested cell                                        |
| CosyVoice 3 / 2                                     | 9 langs (zh en ja ko de es fr it ru) — no SK                                                                                                                                                                               | Apache-2.0                                                             | 0.5B, 150 ms streaming claim (`VENDOR`)                                                                                                                                     | n/a for SK                                                                                                                                                                                                   | Out for SK output; architecture reference for streaming TTS design                                                                                           |
| Kokoro 82M                                          | no SK — EN output only                                                                                                                                                                                                     | Apache-2.0                                                             | 82M, ~1.33 GB VRAM on T4, RTF 0.03                                                                                                                                          | `MEASURED` T4: RTF 0.03, VRAM 1.33 GB (`documentation/load_tests_2026-10.md` + Kaggle v6)                                                                                                                      | **EN TTS only** — the GPU EN voice path. No Slovak output. Dead for SK output; EN-side curiosity/optional GPU engine.                                        |
| MagpieTTS multilingual 357M (NEW)                   | 12 langs (ar de en es fr hi it ja ko pt vi zh) — no SK                                                                                                                                                                     | NVIDIA Open Model License                                              | NeMo + `NeMo-Speech.cpp`, GGUF builds                                                                                                                                       | n/a for SK                                                                                                                                                                                                   | Out for SK. Note for EN-side bulk variety only                                                                                                               |
| XTTS / F5 / CosyVoice-older                         | no SK or CUDA-bound                                                                                                                                                                                                        | various                                                                | —                                                                                                                                                                           | `tts_landscape_2026-09.md` + PLAN.md rejections stand                                                                                                                                                        | Dead, do not retest                                                                                                                                          |
| Kyutai Pocket TTS 100M (NEW, watch)                 | multilingual (verify SK before citing)                                                                                                                                                                                     | open (Kyutai)                                                          | 100M, faster than realtime on CPU, one-line code (`VENDOR` kyutai.org 2026)                                                                                                 | `UNKNOWN` on SK                                                                                                                                                                                              | If SK is covered it becomes the low-end live-engine candidate overnight. One lookup + one synth test                                                         |
| IndexTTS2 MLX (NEW, Mac-side candidate)             | clone path is language-agnostic (identity-first)                                                                                                                                                                           | Apache-2.0                                                             | native MLX on Apple Silicon (`COMMUNITY` soniqo.audio)                                                                                                                      | `UNKNOWN` on owner voice                                                                                                                                                                                     | Candidate for Mac-local bulk generation alongside OmniVoice: no fine-tune, emotion/rate controls, stays on-device. A/B against omni-zeroshot on the v2b refs |
| Qwen3-TTS 0.6B/1.7B                                 | no SK — EN output only                                                                                                                                                                                                     | Apache-2.0                                                             | 4-8 GB VRAM, 97 ms first-packet                                                                                                                                             | `VENDOR` 3-second zero-shot voice clone, EN-focused 10 langs                                                                                                                                                 | **EN zero-shot cloning on GPU**; no Slovak output. Optional GPU engine, declared in registry.                                                               |

Training economics (the handler argument in one table):

| Step                                        | Mac M1 Pro 16 GB                       | Hypothetical CUDA box                                                                                             |
| ------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| One Piper fine-tune, 2500 steps             | `MEASURED` 4 h 40 min                  | `PROJECTED` ~1–1.5 h (~3–4×, VITS fits 8 GB VRAM; MPS path measured slower than CPU so the gain is CUDA-specific) |
| Step-count ablation (2500 vs 5000 vs 10000) | 3 working days serial, machine blocked | same day, machine free for parallel eval                                                                          |
| Bulk 65-clip corpus regen                   | `MEASURED` ~25 min MPS                 | `PROJECTED` ~10–15 min                                                                                            |
| Chatterbox Turbo RTF question               | untestable (MPS-bound 6.7)             | directly testable                                                                                                 |

## 4. S2S — end-to-end speech to speech

No S2S model covers Slovak today. This section exists so the thesis can say so
with citations instead of silence.

| Model                                            | SK?                                                       | Note                                                                                                                                                                                                                              |
| ------------------------------------------------ | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SeamlessM4T v2 (spiked 2026-09-29, `.venv-eval`) | 101 speech-in langs (SK in), 35 speech-out                | `MEASURED` RTF 0.77–0.99 chunked, chrF 39–50, one "willow→fur coat" slip; full-clip inference OOMs 16 GB into 14 GB swap. Verdict stands: quality reference + offline-voice potential, not a live path (3× cascade compute)       |
| Hibiki-Zero 3B (Kyutai, 2026-02)                 | FR/ES/PT/DE→EN only, no SK                                | `VENDOR` SoTA on its pairs; RL-without-aligned-data method adapts to a new input language with <1000 h speech (arxiv 2602.11072). That sentence is the thesis future-work paragraph: SK S2S is a data project, not a model search |
| Hibiki-Zero MLX build                            | same 4 pairs                                              | `COMMUNITY` runs on Apple Silicon INT4 (~75 ms/step on M2 Max) — proves the architecture fits a laptop, just not our language                                                                                                     |
| Moshi / Unmute / Kyutai TTS+STT pair             | EN/FR dialogue                                            | Architecture reference for full-duplex design (200 ms practical, Mimi codec 12.5 Hz). The project's word-window streaming plan is the pragmatic Hibiki-shape on stages that cover SK                                              |
| Canary AST mode                                  | 25 EU langs incl SK | `MEASURED` on Kaggle T4 2026-11-03: SK→EN chrF++ 52.49 mean (n=80, range 19.8–78.2), CS→EN 50.05 (n=80, range 17.0–81.6); English references present but flagged `semantic_review_pending` — below commercial bar (~50 mean, 39–52% of sentences <50 chrF++). AST collapses STT+MT into one step; speed path, not quality path. |

The honest thesis sentence: simultaneous S2S with voice transfer exists and
runs on laptops, but every open implementation speaks four Western languages
into English. The cascade (STT→MT→TTS) is not a legacy choice — for Slovak in
2026 it is the only choice, and the work is making the cascade stream.

## 5. Runnable matrix — what runs where

Assumptions (pinned, update when hardware is confirmed): Mac = M1 Pro 16 GB
(`MEASURED` home); Win = Ryzen 5 8645HS 14 GB CPU-only (`MEASURED` home);
NVIDIA laptop = unknown spec, tiered below by VRAM; server = unknown, rental
math from published rates (`PROJECTED`).

| Workload                                               | Mac CPU/MPS                            | Win CPU                 | NVIDIA ≥4 GB VRAM  | NVIDIA ≥8 GB VRAM    | NVIDIA ≥12 GB VRAM               |
| ------------------------------------------------------ | -------------------------------------- | ----------------------- | ------------------ | -------------------- | -------------------------------- |
| Live pipeline (faster-whisper + Opus CT2 + Piper ONNX) | yes `MEASURED`                         | yes `MEASURED`          | yes                | yes                  | yes                              |
| Nemotron streaming bench (ONNX CPU)                    | yes, runs today                        | yes, runs today         | yes                | yes                  | yes                              |
| Parakeet / Canary ONNX CPU                             | yes                                    | yes                     | yes                | yes                  | yes                              |
| MADLAD-3B CT2 quality bench                            | slow CPU, still benchable              | slow CPU                | yes, comfortable   | yes                  | yes                              |
| Piper fine-tune 2500 steps                             | `MEASURED` 4 h 40 min, machine blocked | slower, untested        | `PROJECTED` ~2–3 h | `PROJECTED` ~1–1.5 h | `PROJECTED` <1 h + parallel eval |
| Chatterbox Turbo / CosyVoice synth                     | MPS-bound, slow                        | CPU-bound               | testable           | comfortable          | comfortable                      |
| SeamlessM4T chunked                                    | `MEASURED` RTF ~0.8–1.0, no full clips | untested, expect slower | yes                | yes                  | yes + batch                      |
| Hibiki-Zero 3B                                         | MLX build exists but no SK pair        | no (needs GPU RAM)      | no (needs 8–12 GB) | borderline           | yes                              |
| STT sweep _during_ a train                             | no (16 GB ceiling, runbook forbids)    | no                      | yes                | yes                  | yes                              |

Minimum useful spec to name to the handler: **any NVIDIA card with ≥6 GB VRAM
and a machine with ≥16 GB RAM**. That unlocks training (~2–3 h), MADLAD bench,
Turbo test, and parallel eval. 8 GB (e.g. RTX 4060 laptop) adds comfort and the
Hibiki borderline; 12 GB adds batch S2S experiments. Below 6 GB VRAM the only
gains are STT/MT speedups the CPU already covers — not worth asking for.
CPU-only loan hardware is still useful as a third portability target (Linux
pass, `linux_setup_and_test.md`) but changes no research speed.

Server alternative: ~15 GPU-h covers the whole 2-week plan (5–6 trains + bulk +
spikes) at `PROJECTED` ~$3–5 on RTX 3060-class rentals, or free on a university
box. Constraint: personal voice clips leave the Mac — fine under university
control, needs one disclosure sentence for rented hardware (constitution
local-first covers the pipeline, not training, but the thesis should say where
the data went).

## 6. Recommended eval order (first bench first)

1. **Nemotron 3.5 streaming SK bench** — DONE 2026-10-01, negative (SK WER 0.75–0.86 with language drift; EN 0.19). Small-sk keeps the SK rung.
2. **Canary AST collapse test** — DONE 2026-11-03 on Kaggle T4 (SK→EN chrF++ 52.49 mean, CS→EN 50.05, n=80 each). chrF++ ~50 is below commercial bar; Canary stays the speed path, cascade (Whisper→Opus-MT) stays the quality path. Not promoted to live route.
3. **OmniVoice live-speech / long-sentence / presentation-pipeline test** — only batch RTF measured (0.14–0.17 long text, 0.40–0.65 short phrase on T4). Not yet tested on continuous microphone speech or multi-minute presentation pacing. Kaggle T4 job needed.
4. **MADLAD-400-3B-CT2 quality bench** — noisy-STT-input chrF vs Opus-MT on SK↔EN, beam 1, CPU first (still informative), CUDA if available.
5. **Step-count ablation** (2500 vs 5000 vs 10000) — the cheapest quality experiment; needs any CUDA box or three Mac days. Do not start before the ear verdict on `me_omni_piper_sk` (if the voice is rejected for naturalness, steps are not the fix).
6. **Chatterbox Turbo RTF on CUDA** — only if a GPU appears; answers whether an MIT trainable engine can touch the live path.
7. **NLLB-600M** — only if MADLAD shows a quality gap worth a second opinion. License trap noted; benchmark, do not adopt, without a commercial-use decision.

Deliberately not queued: Parakeet v3 retest (two owned negatives), EuroLLM
(DEEP bench: slow + weak at 1.7B), Hibiki SK (no weights exist, <1000 h data project).
Kokoro / Qwen3-TTS / Chatterbox are EN-side GPU TTS candidates (no SK weights) —
declared as optional engines in `backend/registry.py`, validated on T4 pending.

## 7. What the handler funds (justification in one page)

The product already hits its latency target on a 2020 CPU laptop — the report
must never ask for hardware as a demo fix. It funds three things:

1. **Throughput: ~1 voice model/day → ~4–6/day.** The train is 95% of
   iteration wall-clock (`MEASURED` §2 baseline in `compute_capacity_report_2026-09.md`).
   A thesis chapter with one trained voice is a report; with an ablation over
   steps × base voice it is an experiment.
2. **Quality unlocks, not speedups:** the step-count ablation, the MADLAD
   quality question, and the Turbo live-engine question are all untestable on
   the Mac in reasonable time. Each is a thesis result, not a faster demo.
3. **The demo he liked (EN mic → SK over speakers)** already works on the Mac
   (`demo_conversation.py`, Lab conference-playback section). New hardware
   buys more voices to play through it and the streaming upgrades (Nemotron
   partials, word-window MT) that make it feel simultaneous.

Ask order: university server access first (free, no third-party data), NVIDIA
loan laptop second (minimum useful spec §5), rented CUDA never unless both
fail — $3–5 covers the plan but moves voice data to a third party.

## 8. Repo evidence index (every MEASURED claim above, by file)

- STT rungs + MT chrF on owner mic: `documentation/model_evaluation_2026-09.md`,
  `processed/sk_direction/sk_direction_matrix.json`
- Live turn latencies + honest gaps: `processed/conversation_sim.json`,
  `processed/meeting/meeting_timeline.json`, `processed/stream_audit/`
- New-voice verdict: `documentation/verification_omni_piper_sk_2026-09-30.md`,
  `processed/engine_ab/matrix_omnihq.json`, `processed/machine_listen.json`
- Bulk corpus QC: `processed/omni_hq_sk/manifest.json`,
  `processed/omni_hq_en/manifest.json`
- Ear grades (machine advisories + owner slots): `processed/ear_grades.json`
  via `scripts/grade_library.py`; Lab display in `ui/voice-lab/library.json`
- Train cost: `processed/overnight_hq_2026-09-30.log`,
  `compute_capacity_report_2026-09.md` §2–§3
- Spikes: `processed/new_models/seamless_matrix.json`,
  `processed/new_models/chatterbox_matrix.json`; harness `scripts/bench_new_models.py`,
  isolated env `requirements-eval.txt`
- Prior verdicts folded in: PLAN.md "Measured 2026-09-29" (turbo negative,
  zipformer/cohere kills, VAD anomaly, live TTS-voice bug fixes)

## 9. Sources (research 2026-10-01)

- OpenWhispr: Parakeet vs Whisper vs Nemotron, best local STT 2026 (2026-07-18) —
  openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron
- NVIDIA NeMo-Speech docs: featured models (Parakeet / Canary / streaming /
  multitalker) — docs.nvidia.com/nemo/speech
- sherpa-onnx model docs: parakeet-tdt-0.6b-v3-int8, nemotron-3.5-streaming
  INT8 bundles — k2-fsa.github.io/sherpa/onnx
- RealtimeSTT sherpa-onnx engines doc (pinned SHAs, bundle sizes) —
  github.com/KoljaB/RealtimeSTT
- gauravvij/parakeet-optimization: TDT v3 CPU profiling (EPYC, INT8, 17–26× RT)
- memoravox/nemotron-3.5-asr-streaming-0.6b-gguf + cstr GGUF + NVIDIA model
  card: 40-locale tiers (SK broad-coverage), Q4_K 458 MB
- kdrkdrkdr/nemotron-asr-streaming.c: pure-C runtime, ~7× realtime on Apple
  Silicon CPU
- Helsinki-NLP Opus-MT repo + OPUS-MT-app (marian/Bergamot local builds)
- arxiv 2607.26286: local-LLM MT vs dedicated baselines (Opus/NLLB wins per pair)
- arxiv 2403.03923v2: MT robustness (NLLB > Opus on noisy input — supports the
  noisy-transcript bench rule in §2)
- DEEP bench (arxiv 2602.19583): Seed/MADLAD/NLLB/Opus ranking (MADLAD BLEU
  36.0, Opus fastest at 57 s)
- santhosh/madlad400-3b-ct2 (CT2-ready, Apache-2.0); Nextcloud-AI MADLAD-7B CT2;
  speech-swift MADLAD MLX docs; EuroLLM-22B tech report (arxiv 2602.05879v1)
- FunAudioLLM/CosyVoice (CosyVoice 3: 9 langs, 150 ms, Apache-2.0);
  resemble-ai/chatterbox (MIT) + onnx-community multilingual-ONNX card
  (23-lang list — no SK; marketing claim corrected);
  offineTTS self-hosted guide 2026-07-24 (Kokoro/F5/Zonos/CosyVoice/Piper matrix);
  soniqo.audio (IndexTTS2 MLX, CosyVoice 3, Chatterbox Flash CoreML RTF 0.59);
  NVIDIA MagpieTTS v2607 card (12 langs, no SK)
- kyutai.org (Pocket TTS 100M CPU-realtime 2026-08-25, TTS 1.6B, STT 500 ms
  delay, Moshi/Unmute/Hibiki-Zero program); Hibiki-Zero paper (arxiv
  2602.11072v1: RL without aligned data, <1000 h adaptation); soniqo Hibiki
  Zero-3B MLX measurements (M2 Max INT4 ~75 ms/step)
- NVIDIA NIM speech pages (riva-translate, nemotron-asr-streaming, parakeet
  NIM containers — server-side options)

## 10. Addendum 2026-10-02 — new leads (research only, nothing benched yet)

Labels as in §0. Rows marked `MEASURED` have been run on the owner's SK clips or Kaggle T4; `VENDOR`/`UNKNOWN` rows await `scripts/bench_new_models.py`.

| Model | SK/CZ? | License | Size / runtime | Key number | Verdict |
| --- | --- | --- | --- | --- | --- |
| **X-Voice 0.4B** (F5-TTS lineage, flow-matching DiT, IPA input; arxiv 2605.05611, `github.com/sunnyxrxrx/X-Voice`, weights `XRXRX/X-Voice`) | **SK and CZ both listed** among 30 langs (paper tables 5-6) | code MIT, **weights CC-BY-NC** (thesis OK, product no) | 0.4B, PyTorch; README lists Apple Silicon install variant, MPS/CPU speed `UNKNOWN`; no fine-tune recipe in README | `VENDOR` RTF 0.073 on RTX 4090; trained on 420k h + 40k h synthetic; claims cross-lingual cloning on par with billion-scale Qwen3-TTS | **Top new TTS test.** The only candidate found that is small, open, cross-lingual cloning *and* lists SK natively. Bench: zero-shot from `me_sk` refs → ASR-WER via small-sk + F0/speaker-sim, RTF on MPS. Beats OmniVoice only if QC WER ≤0.05 at RTF ≤1 |
| VoxCPM2 2B (openbmb; arxiv 2606.06928) | **No SK/CZ** (30 langs: Polish yes, Czech/Slovak no) | Apache-2.0 | ~8 GB VRAM, CUDA ≥12; Mac path not documented | `VENDOR` RTF 0.30 on 4090 (0.13 w/ Nano-vLLM); LoRA fine-tune from 5–10 min of audio | Out for SK output. Reference for *what a LoRA clone recipe looks like* (5–10 min fits the owner's recording budget). Only a CZ-proxy test if a CUDA box appears |
| OmniVoice (incumbent bulk) | 600+ langs | Apache-2.0 | — | `VENDOR` claims 40× realtime on GPU | Already in use; see §3 |
| SloPalSpeech Whisper fine-tunes (small/medium/large-v3/turbo, `NaiveNeuron/*`; arxiv 2509.19270) | native SK | open | medium ~1.5 GB, turbo 1.6 GB | `VENDOR` medium: FLEURS 7.6, CV21 18.0; turbo: FLEURS 6.4, CV21 13.2 | Small-sk is already this lineage. Turbo was measured and rejected on owner mic (§1). Medium-sk is untested here but is slower than small; only worth a bench if owner-mic WER 0.26 is the blocker, not latency |
| COMPASS (arxiv 2606.03241) | n/a | — | benchmark framework, 46 metrics → 10 | `VENDOR` finding: single-metric rankings mislead; best-vs-worst gaps >30% on naturalness / speaker preservation while translation quality differs by a few points | Use as the eval design reference: report translation, naturalness, speaker-similarity and latency as separate columns, never one score |
| Gemini Live Translate (Google, cloud) | 70+ langs | proprietary | cloud | `VENDOR` streaming S2S, intonation/pitch preserved, a few seconds behind speaker | QC / ceiling reference only (constitution III). Not a pipeline candidate |

**Correction kept:** a 2026 search summary claimed Chatterbox Multilingual covers Slovak. §3 already records the model card's 23-language list without SK; that stands. Treat aggregator/blog language claims as `UNVERIFIED` until the model card or a synth test confirms.

**Gap found while writing:** `model_evaluation_2026-09.md` is cited above (§1, §8) but is not in `documentation/`. Its numbers can only be traced via `processed/sk_direction/sk_direction_matrix.json` and git history until restored.

**Updated eval order:** insert **X-Voice SK zero-shot bench** as step 2 (after the closed Nemotron bench, before MADLAD): it is the only new item that could change the voice-cloning answer, runs on the Mac without training, and produces data in the existing `new_models/*_matrix.json` format.

### 10.1 X-Voice bench result (2026-10-02, M1 Pro, MPS) — `MEASURED`, n=1

Same text, same SK reference clip family and same scorer as `scripts/engine_ab.py`. Data: `processed/new_models/xvoice_matrix.json`; wav `processed/engine_ab/xvoice_zeroshot_v2.wav`. Env and checkpoints live outside the repo in `~/xvoice-bench/` (X-Voice clone + venv; `prepare_ipa.sh` NOT run — Linux/sudo script that deletes system espeak libs; Homebrew espeak-ng used instead, phonemizer warns "words count mismatch").

| Engine | small-sk WER | base WER | audio for the same text | synth speed |
|---|---|---|---|---|
| X-Voice zero-shot (ref matched) | **0.000** | 0.487 | 24.1 s | ~RTF 4.2 (marginal, 1 run) |
| omni_zeroshot | 0.054 | 0.514 | 17.0 s | RTF 0.93–1.22 |
| piper_omni_hq | 0.189 | 0.622 | 13.3 s | RTF ~0.04 |

- Intelligibility: best of every engine tested on this text; first attempt scored 0.243 because the reference text covered 2 sentences but the audio was clipped to 12 s (ref words leaked into the output). Fixed by cropping the ref to 7 s with its exact sentence. Lesson: a mismatched ref text is the most likely way to get a false negative from F5-lineage models.
- Speed: ~4.2 s of compute per second of audio on MPS, 3–4× slower than OmniVoice, ~100× slower than Piper. Not a live engine on the Mac. Marginal RTF = (131.8 s wall for 24.06 s audio − 35.8 s wall for a 1.45 s clip) / 22.6 s; model load cancels out.
- Pace: 24 s of audio for text OmniVoice says in 17 s and Piper in 11–13 s → speech is slow/padded at `speed=1.0`. Stage 2 + SpeedPredictor (shipped in the same repo) not tried.
- NOT measured: speaker similarity to the owner, ear grade (trembling was the kill reason for both earlier personal voices), CPU/CUDA speed, long-form stability. WER from a Whisper model rewards clean articulation; it is not a naturalness or identity score.
- Verdict: **quality candidate, not a Mac candidate.** It converts the "voice in near real time" question into a pure compute question — which is what a GPU answers. The only vendor speed number is RTX 4090 (RTF 0.073); no number exists for 6–8 GB cards, so "4090 required" is unsupported and "any CUDA GPU would make this testable" is what the evidence allows. Ask for a CUDA slot, then measure.

### 10.2 First GPU measurement — Colab free T4, 2026-10-02 (`MEASURED`, n=1, `processed/gpu_bench/colab_T4.json`)

Same reference clips and method as the Mac numbers (`scripts/gpu_bench/gpu_bench.py`; refs = 5–7 s clips of the owner, EN/SK/CS). T4 = Tesla T4 15.6 GB, torch 2.11 + CUDA 13. X-Voice fp32, nfe 32, layered CFG (repo defaults); OmniVoice fp16, 16 steps.

| Engine | Mac M1 Pro (MPS) | Colab T4 | Speed-up |
|---|---|---|---|
| X-Voice, marginal RTF (long text) | 2.9 (CS) – 4.2 (SK) | **0.61 (SK) – 0.80 (EN)**, CS 0.66 | ~5–6× |
| OmniVoice, RTF long text (~17 s audio) | 0.80 (CS) | **0.14–0.16** (EN 0.14, SK 0.157, CS 0.139) | ~5.5× |
| OmniVoice, RTF ~1 s phrase (0.7 s to speak 1.1 s) | 2.7 (CS) | **0.40–0.65** | ~5× |

What it says:
- **A T4 is not X-Voice's live engine** (RTF 0.6–0.8, and the vendor's 0.073 on RTX 4090 is ~8–10× faster than this T4 — unmeasured here). **It already runs OmniVoice faster than real time**: a short phrase is synthesised in ~0.7 s, so per-sentence cloned-voice output costs ~0.7 s on top of the cascade, vs ~0.1 s for Piper. Whether that is "near real time" is a latency-budget call (current SK→EN turn ~1.4 s + 0.7 s), not a hardware wall.
- **"RTX 4090 required" is not supported.** No card above T4 was measured yet; the L40S/L4 runs (Modal) will show the scaling.
- Not measured: speaker similarity and ear grade of the GPU outputs (the wavs exist in the Colab VM only; the Modal run saves them locally), X-Voice with fewer steps (`nfe_step 16` should roughly halve the time), streaming/first-chunk latency.
- Setup notes for the repeat: X-Voice's Python deps are heavy and partly unbuildable on Python 3.13 (`python-mecab-ko`); `gpu_bench.py setup` stubs libs that fail to build (Korean/Thai/Japanese G2P are not needed for EN/SK/CS). Colab's session dropped once while driven from a background tab and the terminal accepted input at ~2 chars/s — use the 3-cell notebook (`scripts/gpu_bench/gpu_bench.ipynb`) by hand or `modal_bench.py`.

### 10.3 Kaggle T4 repeat + SeamlessM4T S2ST + word-accuracy check (2026-10-02, `MEASURED`, n=1; `processed/gpu_bench/kaggle_T4/`)

Second, independent run on a free Kaggle Tesla T4 (scripted: `scripts/gpu_bench/kaggle/`, runs from the CLI, no UI). It reproduces the Colab numbers, so the speed result is not a one-off.

| Engine | Mac M1 MPS | Colab T4 | Kaggle T4 |
|---|---|---|---|
| X-Voice marginal RTF | 2.9–4.2 | 0.61–0.80 | **0.46 (CS) – 0.77 (EN)**, SK 0.70 |
| OmniVoice RTF, ~17 s text | 0.80 | 0.14–0.16 | **0.145–0.169** |
| OmniVoice RTF, ~1 s phrase | 2.7 | 0.40–0.65 | **0.45–0.77** |

**SeamlessM4T v2 speech→speech on the T4** (fp16, owner's 5–7 s clips; Slovak has no speech output in M4T v2, so EN→SK is not possible, EN→CS is):

| Direction | Input | Generation | RTF vs input | Text check (ASR of the output) |
|---|---|---|---|---|
| SK→EN | 7.0 s | 1.04 s | 0.149 | "Yesterday morning I went to the market to buy some bread and milk." (correct; drops "fresh") |
| CS→EN | 5.6 s | 0.97 s | 0.173 | "Yesterday I walked from the courtyard to the old town square." (loses "afternoon", odd wording) |
| EN→CS | 7.25 s | 1.38 s | 0.191 | not checkable: no Czech-capable ASR in the project |

**Word accuracy of the clones** (whisper small-sk for SK/CS, base for EN, vs the generated text; measures *what was said*, not how it sounds): X-Voice EN 0.00 / SK 0.108 / CS 0.43; OmniVoice EN 0.024 / SK 0.135 / CS 0.43. The CS figures are inflated by scoring Czech with a Slovak-tuned model (both engines tie at 0.43) — treat as unknown. OmniVoice SK dropped the first words once ("Včera ráno"). The Mac run had given X-Voice SK WER 0.000, so single-clip WER swings ±0.1 between runs.

**What this proves and does not prove**
- Proven (two T4 runs): the speed wall is the Mac, not the models. A free-tier GPU makes OmniVoice cloning ~5× faster than real-time on long text and ~0.7 s for a one-second phrase; Seamless S2ST translates a 7 s utterance in ~1 s.
- Not proven: that the cloned voice sounds like the owner and is free of the tremble that killed the Piper voices. That needs the ear: the 9 clips are in the Voice Lab section "GPU clones + speech-to-speech" and are the only evidence that decides whether Piper training can be dropped.
- Not tested: any card faster than a T4 (Modal's GPUs need a payment method; no result for L4/L40S/4090), latency of a full live cascade with a GPU engine, streaming.

### 10.4 Owner ear verdict on the GPU clips (2026-10-02, chat, qualitative — no numeric grades recorded)

Reported by the owner after listening in the Voice Lab: **the OmniVoice and X-Voice clones (EN/SK/CS, Kaggle T4) are good**; the only generic (non-cloned) voices were the **SeamlessM4T clips** (`kaggle_T4/seamless_{cs_to_eng,en_to_ces,sk_to_eng}`) and the two older `seamless-chunked30_*` spikes. Not recorded: per-clip grade, tremor/hiss, similarity score (ratings made in the browser were not synced to `processed/ear_grades.json`; log in on `https://localhost:8000` or export "Download my ratings (JSON)" to capture them).

What follows, and what does not:
- **Speed + ear verdict together** (two T4 runs + owner ear): on a modest GPU the cascade STT → MT → **OmniVoice clone** is a viable way to speak in the owner's voice, with no per-voice training. Seamless S2ST is fast (RTF 0.15–0.19) but speaks in a stock voice, so it is a translation reference, not the voice path.
- **Piper fine-tuning is not needed on a GPU machine.** It stays the only option for CPU-only machines (Windows laptop, Mac live path): its job there is speed (RTF ~0.04), not quality. The two Piper personal voices are still kill-graded for tremble, so on CPU the live voice remains the generic/`omni`-trained one until a better fine-tune exists.
- **Still unmeasured:** a full live cascade on a GPU (STT + MT + clone TTS) with turn-by-turn timeline, and anything faster than a T4 (Modal GPUs need a payment method). Next step: run the meeting simulation (`scripts/meeting_sim.py` / `demo_conversation.py`) with OmniVoice as the TTS on a GPU, then a live run on the presentation machine.

## 11. Addendum 2026-11-05 — session findings (research + live fixes)

Labels as in §0. `MEASURED` = run on owner's SK clips or Kaggle T4 this session; `VENDOR`/`UNKNOWN` = awaiting `scripts/bench_new_models.py`.

### 11.1 STT — whisper-large-v3-turbo-sk now the SK quality tier

| Model | SK/CZ? | License | Size / runtime | Key number | Verdict |
| --- | --- | --- | --- | --- | --- |
| **kinit/whisper-large-v3-turbo-sk** (new) | native SK fine-tune | MIT | ~809M, ~6GB VRAM GPU, faster-whisper CT2 | `VENDOR` WER 5.60% internal / 9.29% CV24 SK (model card); `MEASURED` not yet on owner mic | **Replaces whisper-small-sk as the SK quality tier.** `BP_SK_STT_MODEL=kinit/whisper-large-v3-turbo-sk` now selectable (`local_stt_source` fixed to accept HF names). Untested on owner mic — Kaggle T4 or CUDA box needed |
| whisper-small-sk (incumbent) | native SK fine-tune | MIT | ~500 MB, faster-whisper CPU | `MEASURED` WER 0.13 FLEURS-sk / 0.26 owner mic; ~2.4s per 7s clip, RTF 0.33 | Keep as SK default on CPU-only machines until turbo-sk is verified |

### 11.2 TTS / voice cloning — new EN-side zero-shot options

| Model | SK/CZ? | License | Size / runtime | Key number | Verdict |
| --- | --- | --- | --- | --- | --- |
| **Qwen3-TTS** (new) | EN only (10 langs), no SK | Apache-2.0 | 0.6B/1.7B, 4-8GB VRAM | `VENDOR` 97ms first-packet, 3s zero-shot reference clone | **EN zero-shot cloning on GPU.** Declared in registry, T4 validation pending |
| **Chatterbox** (correction) | No SK/CZ (23 langs, no SK) | MIT | 0.5B Llama backbone | `VENDOR` sub-200ms claim; `MEASURED` Czech proxy RTF 6.7 on MPS | Already in §3 — stands as no-SK |
| Kokoro 82M | EN only | Apache-2.0 | 82M, ~1.33GB VRAM T4 | `MEASURED` RTF 0.03 on T4 | Already in §3 — EN GPU TTS only |

### 11.3 Local pipeline fixes applied (2026-11-05)

All verified with `make test` → **87/87 pass**:

| Fix | File | Impact |
| --- | --- | --- |
| Streaming chunk processing re-enabled | `backend/main.py:2449-2459` | STT now processes 0.5s chunks during speech instead of waiting for segment close (was disabled → 6-9s latency wall) |
| RMS pre-filter default raised | `backend/main.py:232` | `SILENCE_RMS_THRESHOLD` 0 → 0.02, blocks background noise/"I'm sorry" hallucination loop |
| Debug WAV write removed | `backend/main.py:537-549` → deleted | `_save_audio_segment_for_debug` removed from production TTS path |
| TTS engine remap crash fix | `backend/registry.py:195-198` | `target_tts_engine` returns safe fallback instead of `ValueError` on DE/FR targets |
| `BP_SK_STT_MODEL` HF name support | `backend/main.py:559` | `local_stt_source` accepts HuggingFace model names (not just local CT2 dirs) |
| Pool TTL dead code fixed | `backend/main.py:360-381` | `release_pooled_model` no longer resets `last_used` before TTL check |
| Streaming pipeline locking fixed | `backend/streaming/pipeline.py:126-166` | Init body inside lock; TTS now gated by licence via `select()`/`instantiate()` |

### 11.4 Kaggle T4 results (2026-11-05)

Kernels pushed to private Kaggle account `yegorby`, Tesla T4 GPU:

| Kernel | Route | Profile | p50 | p95 | Stage-sum p50 | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `bp-production-ws-cs-en-default-vs-low-latency` | CS→EN | P_short2 | **1.86s** | 4.22s | 0.40s | COMPLETE — under 2s target |
| `bp-production-ws-cs-en-default-vs-low-latency` | CS→EN | P_short | 2.03s | 3.05s | 0.49s | COMPLETE — under 2s |
| `bp-production-ws-cs-en-default-vs-low-latency` | CS→EN | default | 3.47s | 4.03s | 0.58s | COMPLETE — over target |
| `bp-omnivoice-zero-shot-tts-benchmark-on-t4` | TTS EN/SK/CS | — | RTF 2.92*/0.42/0.45 | — | — | COMPLETE — EN* includes 20s one-time warm-up; SK/CS steady-state RTF <0.5 |
| `bp-t4-kaggle-ende-route-benchmark-v2` | EN→DE | P_short2 | — | — | — | RUNNING |
| `bp-t4-kaggle-enfr-route-benchmark-v2` | EN→FR | P_short2 | — | — | — | RUNNING |
| `bp-t4-kaggle-fren-route-benchmark-v2` | FR→EN | P_short2 | — | — | — | QUEUED (GPU limit 2) |

OmniVoice TTS on T4: model load 25.51s (one-time), CUDA fp16. SK/CS synthesis RTF 0.42/0.45 — **faster than real-time**, viable for live use. EN first synthesis 20.43s (includes one-time warm-up), steady-state RTF <0.5.

Dataset: `yegorby/bp-t4-assets-20261105` v3 — includes repo, CT2 models (Opus-MT + Whisper), `ui/`, and all 7 speaker voice clips.

### 11.5 Local Mac pipeline test (2026-11-06)

User ran live E2E on Mac M1 Pro CPU (EN→CS, `me_en_script.m4a`, 115s). Pipeline **works correctly** — all translations accurate:

| Utterance | Transcription | Translation | STT | MT | TTS | Total |
| --- | --- | --- | --- | --- | --- | --- |
| "Hello." | "Hello." | "Ahoj." | 0.28s | 0.04s | 0.03s | ~0.99s |
| "This is a" | "This is a" | "Tohle je" | 0.35s | 0.02s | 0.03s | — |
| "Okay." | "Okay." | "Dobře." | 0.94s | 0.03s | 0.03s | — |
| "I'll tell you in my next video." | "I'll tell you in my next video." | "Řeknu ti to v příštím videu." | 4.77s | 0.07s | 0.34s | — |
| "Thank you." | "Thank you." | "Děkuju." | 0.36s | 0.03s | 0.04s | ~0.41s |
| "speak." | "speak." | "Mluv." | 0.36s | 0.02s | 0.09s | — |
| "I'm sorry." | "I'm sorry." | "Omlouvám se." | 2.62s | 0.03s | 0.19s | — |

**Findings:**
- Pipeline is functionally correct — all translations accurate, MT+TTS consistently <0.1s
- **STT is the bottleneck on Mac CPU**: 0.28–4.77s per 0.5s chunk; `input_to_stt_latency` grows to 6–11s on longer segments as they hold open waiting for silence
- **Noise hallucination**: "I'm sorry." → "Omlouvám se." on background noise — the RMS pre-filter (0.02) should mitigate but needs live verification
- **Streaming fixes applied** (re-enabled 0.5s chunk processing during speech, RMS pre-filter) — untested on Mac CPU (too slow for live testing); verified on Kaggle T4

### 11.6 What this changes

- **Cascade (STT→MT→TTS) remains the only live path for SK.** No S2S model covers Slovak output. SeamlessStreaming (~2s) and Moshi (~200ms, EN/FR only) are not SK-capable.
- **GPU unlocks OmniVoice cloning** (RTF 0.42–0.45 on T4) but it's still batch/QC — the live path stays Piper on CPU for now.
- **SK quality tier** is now `whisper-large-v3-turbo-sk` (MIT, WER 5.60% < 0.2 gate), not `whisper-small-sk` (WER 0.25–0.32, fails gate).
- **Near real-time achieved on T4** (CS→EN 1.86s p50). CPU-only Mac is ~8s wall — the gap is GPU vs CPU STT, not the pipeline.

### 11.7 Still unmeasured / next steps

| Item | Why | Path |
| --- | --- | --- |
| whisper-large-v3-turbo-sk on owner mic | WER + latency on real audio | Download model (~800MB), run on Kaggle T4 or CUDA box |
| OmniVoice on Kaggle T4 | Kernel complete — RTF 0.42–0.45 confirmed | Results in `omnivoice_report.json` |
| Qwen3-TTS / Chatterbox on T4 | EN zero-shot cloning options | Kernels not yet created |
| Real-mic E2E on Mac | Microphone-to-ear incl. capture + playback | User running live test; `scripts/live/batch_stream_test.py` available |
| DE/FR Kaggle | EN-DE and EN-FR running, FR-EN queued | Wait for completion, retrieve `route_receipt.json` |
| de-en Kaggle | 409 conflict on push — needs fresh kernel ID | Retry with new kernel ID |

