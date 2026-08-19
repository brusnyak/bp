# SK/DE personal-voice bootstrap attempt — 2026-08-19

Same recipe as the English personal voice (RTF 0.0562, commit d41c124), attempted for Slovak and German using XTTS v2 as the offline synthetic-data generator instead of GPT-SoVITS.

## Slovak: blocked, hard wall

XTTS v2 does not support Slovak at all. Its `config.languages` list: `en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi`. No `sk`. This isn't a quality problem — the assertion fails outright. Closest supported relative is Czech (`cs`), already used elsewhere in this project's TTS docs. Not attempted as a substitute here — a Czech-voiced XTTS bootstrap producing a "Slovak" Piper voice would be a real quality/authenticity claim that needs the user's explicit sign-off, not a default fallback.

## German: real audio bootstrapped, Piper fine-tune blocked

- Generated 15 real, diverse German sentences via XTTS v2 cross-lingual cloning from the existing English reference audio (`bootstrap_de/de_000.wav` .. `de_014.wav`, `speaker_voices.json`). This part worked and is committed.
- Piper fine-tuning requires a `.ckpt` warm-start checkpoint. Checked three sources, none have one:
  - `rhasspy/piper-checkpoints` — 401/repository not found (this exact repo name is what the English bootstrap docstring cites; it may have been renamed/removed since)
  - `OHF-Voice/piper-checkpoints` — same 401
  - `rhasspy/piper-voices` (the repo that now shows up in HF search for "piper") — exists, but contains only inference `.onnx` files, zero `.ckpt` training checkpoints, for any language including German
  - A third-party mirror (`csukuangfj/vits-piper-de_DE-thorsten-medium`) — same, `.onnx` only
- Training from scratch (no warm-start) was not attempted — needs far more data and time than fits this session, and wasn't the agreed plan.

**Flag for the parent/user**: the English bootstrap's own checkpoint source (`rhasspy/piper-checkpoints`) may no longer be reachable the way its docstring assumes — worth a quick re-check next time that pipeline is touched, in case it was working only from a locally cached file rather than a live fetch.

## What's real and usable right now

- `bootstrap_de/*.wav` (15 files) + `speaker_voices.json`: genuine XTTS-cloned German audio in the user's voice, RTF-class 1.72 (XTTS speed, not Piper). One sample copied to `/Users/yegor/Desktop/personal_voice_test_de_xtts_not_piper.wav` for listening — explicitly labeled as XTTS-quality, not the fast Piper tier.
- No fast (Piper-speed) German or Slovak personal voice exists yet. Slovak is blocked at the XTTS layer (no language support, no workaround attempted without user sign-off on a Czech substitute). German is blocked only at the "no downloadable warm-start checkpoint" layer — the synthetic data itself is ready and committed, so this is resumable the moment a checkpoint source is found, not a dead end requiring redoing.
