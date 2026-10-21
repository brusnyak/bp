# Language packs

A language is "supported" when a pack here resolves its components (STT + MT + TTS)
and passes `bp language test <pair>`. Packs are data, not code — adding a language
means adding a JSON file, not editing the pipeline (constitution III as of 1.1.0;
spec `specs/003-end-to-end-completion/`).

## File name

`languages/packs/<source>-<target>.json`, e.g. `en-sk.json` (English spoken → Slovak out).

## Fields

| Field | Meaning |
|---|---|
| `pair`, `source`, `target` | ISO 639-1 codes. `source` is what the STT hears, `target` is what TTS speaks. |
| `stt.model` | faster-whisper size. Resolves to `ct2_models/whisper-<model>/` if present, else a HF id. |
| `stt.language` | Whisper decoder language for the source (`en`, `sk`, `cs`, ...). |
| `stt.initial_prompt` | Optional orthography hint (helps diacritics / script). |
| `mt.model` | Opus-MT dir under `ct2_models/` (`Helsinki-NLP--opus-mt-<src>-<tgt>`), or an `org/name` HF id to convert. |
| `tts.voice` | Piper voice id (file in `backend/tts/piper_models/<id>.onnx`). |
| `tts.fallback` | Optional second Piper voice tried when `voice` is absent (e.g. a personal voice that is local-only). |
| `clip` | Default test clip, relative to repo root. |
| `license` | `commercial-ok` \| `non-commercial` \| `unknown`. Non-commercial packs are blocked in the default profile. |
| `notes` | Free text. |

## Commands

```
bp language list                 # every pack, resolved component status
bp language show en-sk           # the raw pack
bp language add pl-en            # scaffold a pack from auto-detected components
bp language remove pl-en --yes   # delete a pack
bp language test en-sk           # run STT->MT->TTS on the clip, write an eval record
```

Eval records land in `processed/language_eval/<pair>.json` and carry the engine
versions + per-stage latency, so "verified" always points at a measurement.
