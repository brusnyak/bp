# Recording session — new voices for Piper (2026-09-28)

Why: the personal Piper voices are trained on **your** recordings — the clone's quality
ceiling is the audio you give it. This is everything to record, in priority order, with
the exact commands. Everything is local; nothing is uploaded anywhere.

## Before you start (once)

- `.venv` must exist: `python3.11 scripts/setup.py --dev` (already done on this Mac).
- The 75-sentence script is generated: `documentation/recording_script_sk_v2.md`
  (regenerate anytime with `.venv/bin/python scripts/build_recording_set.py`).
- Find your microphone number: `.venv/bin/python scripts/record_reading.py --list-devices`
- Room rules: quiet room, mic 15–20 cm from your mouth, same position for the whole
  session, **normal conversational pace** (not announcer style), ~0.5 s of silence before
  and after each sentence. The recorder warns live on clipping/too-quiet and lets you
  redo (`r`), skip (`s`), or quit and resume later (`q`).

## Block A — Slovak, 75 sentences · CRITICAL PATH (~12–15 min)

Trains the SK personal voice **and** forms the SK evaluation set (WER/speed numbers).

```bash
.venv/bin/python scripts/record_reading.py            # prints each sentence; resumes
.venv/bin/python scripts/record_reading.py --redo 12  # re-record only sentence 12
```

Files: `eval_data/recording_set_v2/sk_00.wav … sk_74.wav`.

You do **not** need native pronunciation — see the pronunciation policy below. If a
phrase feels unnatural to you, say so and it gets replaced *before* you record.

## Block B — Slovak colloquial passage, 3 paragraphs (~5 min)

The ElevenLabs instant-voice-cloning sample text you supplied. Record the three paragraphs
below as three separate files (Voice Memos/QuickTime is fine, any wav/m4a — I convert to
16 kHz). Why this text: (1) it is the standard-class enrollment sample for instant
cloning, so it is the direct comparison reference against our fine-tune; (2) colloquial
storytelling is exactly what the formal news script is not — the hardest case for Slovak
STT/MT, so it doubles as an accuracy probe.

**Paragraph 1**
> Dobre, takže neuveríte, čo sa včera stalo v obchode s potravinami. Práve schmatnem avokádo, správne, starám sa o svoje veci, a tento chlapík sa zvalí vedľa mňa s obrovským papagájom na pleci. Obrovský jasne červený papagáj, ktorý tam sedí, akoby to bolo úplne normálne. A nikto okolo nás nereaguje vôbec. Stojím tam ako, som jediný, kto to teraz vidí? Vážne, musel som to urobiť dvakrát.

**Paragraph 2**
> Tak sa snažím nezízať, no potom sa papagáj pozrie priamo na mňa a ide „pekné tričko“. A mám na sebe to najhoršie staré tričko, aké mám, ako to, v ktorom občas spím. Chlap sa ani nepohne. Len na mňa prikývne, akoby sme sa rozumne porozprávali, a odíde k pultu s lahôdkami. Skoro mi vypadlo avokádo. Ten papagáj si zo mňa práve vystrelil a majiteľ sa tváril, akoby o nič nešlo. Stále mi to neprešlo.

**Paragraph 3**
> A najdivokejšia časť? Spýtal som sa na to pokladníka, keď som odchádzal. Tento chlap zrejme prichádza každý utorok. Každý utorok! S papagájom. A papagáj má rôzne veci, ktoré hovorí rôznym ľuďom. Pokladník mi minulý týždeň povedal, že niekomu povedal, že ich vozík vyzerá smutne. Myslím, čo? Ako papagáj posudzuje vaše potraviny? Mám toľko otázok a nulové odpovedí. Napríklad, kto vycvičil tohto vtáka?

## Block C — English, same 75 sentences · recommended (~15 min)

Your English pronunciation is the strong side: this trains the **EN** personal voice
(the voice used when SK→EN output speaks English) and gives a clean EN STT eval set
with zero accent risk.

```bash
.venv/bin/python scripts/record_reading.py --lang en
```

Files: `eval_data/recording_set_v2/en_00.wav … en_74.wav`.

## Block D — Czech passage, 3 paragraphs · OPTIONAL (~5 min)

Only worth it if you want a personal Czech voice (nothing in the EN↔SK thesis path
needs it). We QC the takes first and drop them if the accent metric is bad.

**Paragraph 1**
> Dobře, takže nebudeš věřit, co se včera stalo v obchodě s potravinami. Právě si vybírám avokáda, jo, dělám si svoje, a přivalí se ke mně chlap s obrovským papouškem na rameni. Jako fakt obrovský, jasně červený papoušek, sedí mu tam, jako by to bylo úplně normální. A nikdo kolem vůbec nereaguje. Stojím tam a říkám si, jsem tu jediná, kdo to vidí? Vážně, musela jsem se podívat dvakrát.

**Paragraph 2**
> Tak se snažím nezírat, ale pak se papoušek podívá přímo na mě a řekne „pěkné triko.“ A mám na sobě to nejšpinavější staré tričko, co vlastním, to, ve kterém občas spím. Ten chlap ani necukne. Jen na mě kývne, jako by proběhla naprosto normální výměna, a odejde k pultu s lahůdkami. Málem jsem upustila avokáda. Ten papoušek mě právě sejmul a majitel se choval, jako by o nic nešlo. Pořád mi to nedá.

**Paragraph 3**
> A ta nejdivočejší část? Cestou ven jsem se na to zeptala pokladní. Ten chlap zjevně přichází každé úterý. Každé úterý! S papouškem. A papoušek má různé průpovídky pro různé lidi. Pokladní mi řekla, že minulý týden někomu řekl, že jeho vozík vypadá smutně. Chci říct, co? Jak může papoušek posuzovat vaše nákupy? Mám tolik otázek a nulové odpovědí. Jako, kdo toho ptáka vycvičil?

## After you share the files (my side)

1. **Machine-listen QC on every take**: clipping/headroom, duration, STT WER per sentence,
   F0/HNR sanity → you get a short re-record list (`--redo N`), only the failures.
2. **QC pass → Piper fine-tune** (a longer run than the 18-clip one), then Voice Lab A/B +
   Praat panel before/after: WER thirds, F0 spread, HNR, RTF (speed).
3. Decisions are made on the lab numbers, not impressions.

## Pronunciation policy (non-native speaker caveat)

- **Training (your voice):** your own accent is part of *your* voice — the thesis claim is
  "a clone of me", so accented Slovak is honest and correct. The QC loop (record → WER →
  redo flagged) catches real mistakes; it does not require native perfection.
- **Evaluation (system accuracy):** native speech stays covered by the public FLEURS-sk
  set already used in `documentation/model_evaluation_2026-09.md`. If you want the
  project's own set read natively too, one native speaker + 30 minutes solves it —
  otherwise set-B numbers are labelled "non-native reader", a legitimate deploy condition
  for an international-conference tool anyway.
- **No synthetic Slovak from your English audio** (voice-cloning TTS speaking Slovak):
  XTTS-v2 has no Slovak in its tokenizer (Czech-phonetics proxy — measured, rejected as
  the SK default 2026-09-28), and cloud-TTS-generated audio in "your voice" would break
  both the local-first rule and the trained-on-my-own-audio claim.

## Time budget

| Block | What | Time |
|---|---|---|
| A | SK 75 sentences (critical) | 12–15 min |
| B | SK colloquial passage ×3 | ~5 min |
| C | EN 75 sentences (recommended) | ~15 min |
| D | CZ passage ×3 (optional) | ~5 min |
