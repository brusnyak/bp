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

## Demo fallback ladder (if live audio fails)

1. Live sentence EN→SK in own voice
2. Voice Lab A/B (male_last_base vs generic)
3. `processed/e2e_ensk_sk.wav` (53s proof, plays anywhere)
