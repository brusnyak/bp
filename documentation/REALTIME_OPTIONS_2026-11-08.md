# Near real-time options: what exists, checked 2026-11-08

Question: is there something that already produces a transcript, a translation or a generic voice in near real time, so that this project does not have to invent it? Only the lines marked "read" were verified this session from the page named in the last column; the rest is labelled.

## Hosted services (cloud: reference points only; the project's own pipeline stays local)

| Service | Slovak | What it offers | Checked | Source read |
|---|---|---|---|---|
| Palabra | Slovak listed among 60+ languages | Two-way speech-to-speech translation API; vendor claims under 1 s latency and $50 sign-up credit (vendor claims, not measured) | read | palabra.ai/voice-translation-api |
| Azure AI Speech | `sk-SK` listed in the language-support tables | Speech translation, including a "Live Interpreter" speech-to-speech mode that detects the input language | read | learn.microsoft.com speech-translation, language-support |
| Deepgram Nova-3 | `sk` listed | Streaming recognition (no translation or voice read) | read | developers.deepgram.com models-languages-overview |
| Soniox | `sk` listed | Recognition languages (translation feature not read) | partial | soniox.com/docs supported-languages |
| Speechmatics | not read | Real-time translation product page exists | not read | speechmatics.com/product/translation |
| ElevenLabs Scribe realtime, Google streaming | not read | nothing read | unverified | none |

## Open models you can run yourself

| Model | Languages | Fit | Source read |
|---|---|---|---|
| Voxtral Realtime (Mistral, Apache 2.0) | 13: ar, de, en, es, fr, hi, it, nl, pt, zh, ja, ko, ru. No Slovak, no Czech | Streaming recognition at 80-2400 ms configurable delay, 480 ms advised, accuracy on par with Whisper; MLX on a 16 GB Mac at INT4 | arXiv 2602.11298, model card |
| Hibiki-Zero (Kyutai, 3B) | French, Spanish, Portuguese, German into English only | Real-time speech-to-speech translation with voice transfer; NVIDIA GPU, 8-12 GB VRAM | github.com/kyutai-labs/hibiki-zero |
| SimulStreaming (UFAL) | Whisper's 99 source languages, 35 target languages via EuroLLM | Whisper plus a simultaneous policy (AlignAtt); the project's own growing-buffer route is the simpler cousin; GPU of 10 GB advised | github.com/ufal/SimulStreaming |
| Nemotron streaming | Slovak benched negative here (word error 0.75-0.86) | not usable for Slovak | `model_landscape_2026-10.md` |

## What this means

1. Nothing read shows an open model that streams Slovak speech at usable accuracy. Slovak input stays on Whisper with phrase or clause cutting.
2. English and German input have a streaming path worth testing (Voxtral Realtime), and the demo case (an English speaker, Slovak or German listeners) is exactly the accurate case.
3. A hosted service can show what "near real time for Slovak" looks like with someone else's engineering, which settles whether the target is physically reachable. Sign-up needs the owner's decision (account creation is gated).
4. Streaming recognition into a voice without translation is what voice-agent frameworks (LiveKit Agents, Pipecat) do; that is from general knowledge, not read this session.
