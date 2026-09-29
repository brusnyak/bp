# Handler update (draft, to send before the meeting)

## Správa (Slovak, prototype — upraviť pred odoslaním)

Dobrý deň, pán Minárik,

posielam stručný update k BP (real-time preklad reči EN↔SK):

- Funkčný lokálny pipeline STT→MT→TTS, boilerplate overený meraniami
  (nie odhadmi): STT EN WER 0,02 / SK 0,41, preklad 18 viet za 0,5 s,
  syntéza vlastným hlasom RTF 0,05.
- Vlastný SK hlas dotrénovaný z mojich nahrávok (beží lokálne, bez cloudu);
 _EN strana a streamovanie (súvislý vstup namiesto chunkov) ostávajú otvorené.
- Rád by som Vám to ukázal naživo (1 veta + ukážky v prehliadači) a
  prekonzultoval štruktúru meraní do kapitoly 5.

Vyhovoval by Vám krátky call/demonštrácia tento týždeň?

S pozdravom,
Yegor Brusnyak

## Talking numbers (do not read out all — pick two)

- STT: EN 0.023 (Parakeet) / SK 0.41 (turbo, adopted default)
- MT: 0.50s / 18 sentences chunked; word-streaming 3x faster to first audio
- TTS: own male SK voice, 0.28s per 6s audio; F0 111Hz vs speaker 101Hz
- E2E wall 18.7s incl. loads; live sentence path ~2–4s (STT-bound)
- Honest gaps: SK STT is the binding constraint; streaming STT is next build

## Hardware ask (numbers prepared — see compute report)

Full analysis: `documentation/compute_capacity_report_2026-09.md`.
Every figure there is labelled `MEASURED` or `PROJECTED`; only the measured
ones may be quoted as results.

- **The product does not need a GPU.** The shipped pipeline hits ~1 s EN→SK and
  ~3 s SK→EN on a 2020 CPU-only laptop (`MEASURED`, `demo_report_2026-09.md`).
- **The research loop does.** One Piper fine-tune = **~8–9 h** on the M1 Pro
  (0.07–0.08 it/s CPU, `MEASURED` 2026-09-29) — ~95 % of a working day, and the
  16 GB unified-memory ceiling blocks running any evaluation alongside it.
- **Projected on an NVIDIA laptop (32 GB + RTX 4060-class):** same fine-tune
  **~1–1.5 h** (~6×), bulk corpus ~2× faster, Chatterbox synthesis ~4× faster,
  and training + eval can run concurrently. Voice iteration cadence goes from
  ~1 model/day to ~4–6 models/day. `PROJECTED`.
- **Projected on a university server or rented CUDA:** same per-hour gains;
  one fine-tune ≈ $0.25–0.40 of compute, the whole 2-week plan ≈ $3–5.
  A university server is preferred (no cost, and personal voice recordings stay
  inside the university rather than going to a third party). `PROJECTED`.
- **What it would buy, concretely:** the step-count ablation (2500 vs 5000 vs
  10 000 steps) — currently one day per data point, which is why the thesis has
  a single setting instead of a comparison.

## Demo fallback ladder (if live audio fails)

1. Live sentence EN→SK in own voice
2. Voice Lab A/B (male_last_base vs generic)
3. `processed/e2e_ensk_sk.wav` (53s proof, plays anywhere)
