# Handler update (draft, to send before the meeting)

## Správa (Slovak, prototype — upraviť pred odoslaním)

Dobrý deň, pán Minárik,

posielam stručný update k BP (real-time preklad reči, teraz už EN↔{SK, CS} a viac, lokálne, bez cloudu v potrubí):

- Funkčný lokálny pipeline STT→MT→TTS, overený meraniami (38/38 testov zelených, `bp selftest` prejde): STT EN WER ~0,08 / SK 0,21+ (skúšame Canary AST na GPU, aby sme klesli pod 0,2 bránu), preklad Opus-MT ~0,08–0,1 s/vetu, syntéza Piper RTF ~0,04.
- Vlastný SK hlas dotrénovaný z mojich nahrávok (beží lokálne, bez cloudu); čeština je pridaná ako druhý jazyk s existujúcim Piper hlasom `cs_CZ-jirka-medium` a silným Opus-MT párom (chrF ~0,72).
- **Úzka miestnosť:** SK→EN STT je stále experimentálna (najlepší otvorený slovenský model je nad 0,2 WER bránou); pre český smer je to miernejšie, ale nie vyriešené.
- "Preklad vo vlastnom hlase" (vlastná interprétácia): pre SK výstup neexistuje otvorený zero-shot model na klonovanie hlasu (XTTS, Chatterbox, Qwen3-TTS, Kokoro všetky vynechávajú SK). Jediná cesta je Piper fine-tune (4–8 h na Macu, RTF ~0,04 pri inferencii). Pre EN výstup klonovanie funguje (Chatterbox/Qwen3-TTS/XTTS na GPU).

Vlastný SK hlas dotrénovaný z mojich nahrávok (beží lokálne, bez cloudu); _EN strana a streamovanie (súvislý vstup namiesto chunkov) ostávajú otvorené.

Rád by som Vám to ukázal naživo (1 veta + ukážky v prehliadači) a prekonzultoval štruktúru meraní do kapitoly 5.

Vyhovoval by Vám krátky call/demonštrácia tento týždeň?

S pozdravom,
Yegor Brusnyak

## Talking numbers (do not read out all — pick two)

| Metric | Value | Label | Evidence |
|--------|-------|-------|----------|
| STT EN WER | ~0.08 | MEASURED | `processed/stt_baseline.json` |
| STT SK WER (clean) | 0.2143 | MEASURED | Kaggle v6 T4, `documentation/research_findings_2026-11-03.md` |
| STT SK WER (mic) | ~0.32 | MEASURED | `documentation/project_status_2026-10-20.md` |
| STT CZ WER (FLEURS/CV) | ~12% | VENDOR/community | BuzzASR/czech, `documentation/research_findings_2026-11-03.md` |
| MT EN↔SK chrF | ~0.66 | VENDOR | `Helsinki-NLP/opus-mt-tc-big-ces_slk-en` flores |
| MT EN↔CZ chrF | ~0.72 | VENDOR | `Helsinki-NLP/opus-mt-tc-big-ces_slk-en` tatoeba |
| MT | 0.50 s / 18 sentences chunked | MEASURED | chunked batch |
| TTS own SK voice | RTF ~0.04 | MEASURED | Piper ONNX CPU |
| TTS EN (Kokoro, GPU) | RTF 0.03, 1.33 GB VRAM | MEASURED | T4, `documentation/research_findings_2026-11-03.md` |
| E2E EN→SK turn | ~0.78–1.3 s | MEASURED | live smoke test 2026-11-03 |
| E2E SK→EN turn | ~1.27–1.64 s | MEASURED | STT-bound |
| Tests | 38/38 pass | MEASURED | `make test` |
| Selftest | PASSED | MEASURED | `bp selftest` |

## Hardware ask (numbers prepared — see compute report)

Full analysis: `documentation/compute_capacity_report_2026-09.md`. Every figure there is labelled `MEASURED` or `PROJECTED`; only the measured ones may be quoted as results.

- **The product does not need a GPU.** The shipped pipeline hits ~1 s EN→SK and ~3 s SK→EN on a 2020 CPU-only laptop (`MEASURED`, `demo_report_2026-09.md`).
- **The research loop does.** One Piper fine-tune = ~4 h 40 min on the M1 Pro (`MEASURED`, 2026-09-30) — ~95 % of a working day, and the 16 GB unified-memory ceiling blocks running any evaluation alongside it.
- **Projected on an NVIDIA laptop (32 GB + RTX 4060-class):** same fine-tune ~1–1.5 h (~3–4×), bulk corpus ~2× faster, Chatterbox synthesis ~4× faster, training + eval concurrent. Voice iteration cadence ~1 model/day → ~4–6 models/day. `PROJECTED`.
- **Projected on a university server or rented CUDA:** same per-hour gains; one fine-tune ≈ $0.25–0.40 compute, whole 2-week plan ≈ $3–5. University server preferred (no cost, voice recordings stay inside university). `PROJECTED`.
- **What it would buy concretely:** step-count ablation (2500 vs 5000 vs 10 000) — currently one day per data point; CZ STT fine-tune validation; Canary AST SK→EN validation on T4.

## Demo fallback ladder (if live audio fails)

1. Live sentence EN→SK in own voice
2. Voice Lab A/B (male_last_base vs generic)
3. `processed/e2e_ensk_sk.wav` (53 s proof, plays anywhere)

## New: Czech (CZ) — second verified direction

- CZ is mutually intelligible with SK (West Slavic); adding CZ is low-effort because the pipeline already resolves engines by config.
- STT: Whisper supports CZ natively; BuzzASR/czech fine-tune WER ~12% (much better than SK's ~21–32%).
- MT: Opus-MT CZ↔EN chrF ~0.72 (stronger than SK↔EN ~0.66); CZ↔SK pair exists.
- TTS: Piper `cs_CZ-jirka-medium` already in `backend/tts/piper_models/` — ready to ship.
- S2S: SeamlessM4T v2 supports Czech S2S (ces: Sp,Tx,Sp,Tx) same as Slovak.

**Recommendation:** ship CZ as a verified second direction alongside SK; it de-risks the "any language" claim and gives the handler a working multi-language demo.
