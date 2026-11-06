# Licence inventory (commercial use)

Read 2026-11-04 from each voice's MODEL_CARD at `huggingface.co/rhasspy/piper-voices` and the dataset licence pages. "Verified" = the licence text or card was read in this session. Not legal advice; the owner should confirm before selling anything.

## Piper voices
| Voice | Role | Dataset licence | Commercial use | Notes |
|---|---|---|---|---|
| `sk_SK-lili-medium` | SK default | CC0 (OHF-Voice/voice-datasets) | Yes | Verified. Quality: earlier vault note measured WER 0.25 on 8 sentences; re-check |
| `cs_CZ-jirka-medium` | CS default | CC0 (OHF-Voice/voice-datasets) | Yes | Verified |
| `en_US-john-medium` | EN default (new) | public domain (LibriVox) | Yes | Verified; finetuned from `kristin` (also public domain). Local test: WER 0.044, RTF 0.03 (6 sentences, Whisper base ASR) |
| `en_US-kristin-medium` | EN alternative (female) | public domain (LibriVox) | Yes | WER 0.075 in the same test |
| `en_US-norman-medium` | EN alternative (male) | public domain (LibriVox) | Yes | WER 0.080 |
| `en_US-ljspeech-medium` | EN alternative | public domain (LJ Speech) | Yes | Card read, not tested |
| `en_US-libritts_r-medium` | EN multi-speaker | CC BY 4.0 | Yes, with attribution | Card read, not tested |
| `en_US-ryan-medium` | old EN default | CC BY-NC-SA 4.0 | **No** | Removed from defaults. Do not ship |
| `en_US-lessac-medium` | - | Blizzard 2013 licence (research use, licence granted to the named recipient) | **No** (treat as non-commercial) | Voices fine-tuned from lessac (amy, joe, arctic, kusal, sam, jenny_dioco) inherit this risk |
| `en_US-personal-*`, `sk_SK-personal-male-medium` | owner's own fine-tunes | owner's recordings, but `personal-v2` was warm-started from `ryan` (CC BY-NC-SA) | **No until retrained** | Retrain from a clean base (`john`/`kristin`, `lili`, `jirka`) before shipping. `personal-v2` base confirmed in `documentation/personal_voice_bootstrap_2026-08-19.md`; SK personal base not checked |

Piper voices as published are listed under MIT at the repository level; the dataset licence above is the one that matters.

## Other components
| Component | Licence | Note |
|---|---|---|
| faster-whisper / Whisper weights | MIT | |
| `whisper-small-sk` (fine-tune used for SK) | not checked | Check the model card before shipping |
| Opus-MT (Helsinki-NLP) | CC-BY-4.0 | Attribution required |
| Piper runtime | GPL-3.0-or-later | Fine for open source; check implications for a closed product |
| OmniVoice weights | CC-BY-NC (vault note, not re-verified here) | Research only |
| VoiceStudio (debpalash) | AGPL-3.0 app; model licences differ | Use as an offline tool only; never link into BP |
| Kokoro-82M | Apache-2.0 (English only) | Candidate for better EN quality; not integrated |

## Rules for this repo
- Defaults must come from the "Yes" rows. Pack `license` fields in `languages/packs/` now say what is verified.
- Do not commit `.onnx` voices (they are git-ignored); `scripts/setup.py` downloads the defaults.
- For custom voices: record the speaker's written consent, train from a clean base, publish the result under CC0 or CC-BY.
