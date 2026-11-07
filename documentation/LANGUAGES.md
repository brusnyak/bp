# Languages: best configuration and measured quality

Every number here is checked against its saved receipt by `make check-docs` (`<!-- claim:... -->` markers; evidence in `documentation/evidence/2026-11-07/`).
Test data: speech recognition on FLEURS read speech (60 clips per language), translation on 200 opus-100 test sentences (chrF, relative scores),
chain timing on 25 FLEURS clips of about 10 s. Hardware: Kaggle Tesla T4 (GPU numbers) and an M1 Pro (Mac numbers). Read `documentation/BENCHMARKS.md` for method and limits.

## Pick a profile

| Machine | Install | Run | Speech model | Why |
|---|---|---|---|---|
| Apple Silicon | `python scripts/setup.py --profile mac --langs de,sk` | `make run PRESET=mac` | Whisper large-v3-turbo on the GPU (MLX) | German 4.4% word error on the Mac; one call costs about 0.7 s |
| NVIDIA GPU | `python scripts/setup.py --profile nvidia --langs de,fr,pl` | `make run PRESET=nvidia` | faster-whisper large-v3-turbo, float16 | German 4.8% on a T4 at 0.4-0.5 s per 10 s clip |
| CPU only | `python scripts/setup.py --profile cpu` | `make run PRESET=cpu` | Whisper small, int8 | German about 11% word error; slower, same model |

`--langs` takes `de,sk,cs,fr,pl` (default `de,sk,cs`). It fetches that language's voice and translation models. English is always included.

## Per language (English paired with the language)

Speech recognition, mean word error % on T4 (lower is better), translation chrF (higher is better), voice, and the measured chain on a 10 s utterance.

| Language | Best recognition | Small / base | Translation to / from English (chrF) | Piper voice | Chain to English / from English |
|---|---|---|---|---|---|
| **German** | large-v3-turbo <!-- claim:stt:whisper-large-v3-turbo:de:wer -->4.8<!-- /claim -->% (median 1.1%); Parakeet 6.3% | 10.9% / 18.1% | <!-- claim:mt:opus-mt:de:to_en:chrf -->57.9<!-- /claim --> / 52.9 | `de_DE-thorsten-medium` | <!-- claim:e2e:de->en:chain -->1.34<!-- /claim --> s / <!-- claim:e2e:en->de:chain -->1.21<!-- /claim --> s |
| English | large-v3-turbo <!-- claim:stt:whisper-large-v3-turbo:en:wer -->4.5<!-- /claim -->%; Parakeet 5.1% | 6.1% / 8.6% | n/a | `en_US-john-medium` | n/a |
| French | Parakeet 5.0%; turbo <!-- claim:stt:whisper-large-v3-turbo:fr:wer -->6.9<!-- /claim -->% | 14.8% / 25.6% | 60.4 / 61.0 | `fr_FR-siwis-medium` | 1.57 s / 1.24 s |
| Polish | turbo <!-- claim:stt:whisper-large-v3-turbo:pl:wer -->7.7<!-- /claim -->%; Parakeet 9.9% | 19.6% / 37.9% | 50.6 / no model | `pl_PL-gosia-medium` | 1.15 s / n/a |
| Czech | turbo <!-- claim:stt:whisper-large-v3-turbo:cs:wer -->11.6<!-- /claim -->%; Parakeet 14.0% | 37.8% / 67.3% | <!-- claim:mt:opus-mt:cs:to_en:chrf -->59.2<!-- /claim --> / 51.2 | `cs_CZ-jirka-medium` | <!-- claim:e2e:cs->en:chain -->1.31<!-- /claim --> s / 1.41 s |
| Slovak | **kinit Slovak turbo** <!-- claim:sk:kinit:fleurs -->6.5<!-- /claim -->%; Parakeet 9.9% | plain turbo 10.7%, small-sk 11.8% | 62.1 / 55.6 | `sk_SK-lili-medium` (or your own fine-tune) | <!-- claim:e2e:sk->en:chain -->1.34<!-- /claim --> s / <!-- claim:e2e:en->sk:chain -->1.14<!-- /claim --> s |

Notes that decide the configuration:

- **Never use Whisper base for anything but English.** It scores 38-67% word error on Czech, Slovak and Polish.
- **German** is the best-covered pair: turbo is accurate, Opus-MT is solid, and the chain is fast. The default setup installs both translation directions and the `thorsten` voice. German runs end to end through the real server on a T4 (English reading in, German speech out: delay <!-- claim:route:en-de|agree10:lag -->1.56<!-- /claim --> s with `BP_INCREMENTAL=1 BP_INCR_STEP_S=1.0`, set by the `nvidia` preset). It has not been run through the browser UI or on a Mac yet.
- **Slovak** needs the kinit fine-tune (`BP_SK_STT_MODEL`, set by the `mac` and `nvidia` presets; `setup.py --langs sk` converts it). On a non-native speaker's own reading it scores <!-- claim:sk:kinit:owner -->19.8<!-- /claim -->% (plain turbo <!-- claim:sk:plain:owner -->30.2<!-- /claim -->%, the old small model 20.8%): the speaker's pronunciation, not the model, is the limit. Parakeet auto-detects the language and picked the wrong one on that reading (80.7%): do not use it for Slovak from non-native speakers.
- **Parakeet** (Parakeet TDT v3, CC-BY-4.0, `BP_STT_ENGINE=parakeet_onnx`) wins French and Slovak by under 2 points, loses everywhere else, and is 3x faster per clip on a GPU. Keep it optional.
- **Translation:** Opus-MT beat NLLB-600M and NLLB-1.3B in every language and direction measured, and is 4-8x faster. There is no English-to-Polish Opus-MT model; use another route for that direction.
- **Voices:** Piper voices are generic. A voice that sounds like you needs a fine-tune (the Slovak personal voice) or cloning (GPU, research licences): see `documentation/STATUS.md`.

## Adding a language

1. Add a pack in `languages/packs/<src>-<tgt>.json` (copy `en-de.json`).
2. Add the language to `LANGS` in `scripts/setup.py` (voice and translation pairs) and run `python scripts/setup.py --langs <code>`.
3. Measure it: run `scripts/gpu_bench/kaggle_matrix` and `kaggle_e2e` on a Kaggle T4 with the new code, then add the receipt to `documentation/evidence/` and mark the numbers you quote.
