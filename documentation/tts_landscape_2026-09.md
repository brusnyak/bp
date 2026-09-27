# TTS + S2S landscape, September 2026 — what exists, what we can use

Question: is there anything better than our STT→MT→TTS Piper pipeline,
especially after the Gemini 3.8 and XTTS results? Answer per stage.

## Cloud TTS with cloning (reference only — never the pipeline)

- **Gemini 3.8 Flash TTS** (2026-09-25): 30s voice replica, 100+ languages,
  emotion/tempo control, API available. BLOCKED: voice replica geo-fenced out
  of the EEA (Slovakia included). Cloud latency + cost + thesis local-first rule
  it out regardless. Status: possible single-call QC reference IF accessible.
- **ElevenLabs IVC/PVC**: best-reported clone quality; IVC struggles with
  uncommon accents (docs); PVC needs 30–180 min + subscription. Cloud-only.

## End-to-end speech-to-speech (the architecture horizon)

- **Kyutai Hibiki / Hibiki-Zero** (2025-02 / 2026-02): THE open reference for
  simultaneous S2S with voice transfer — decoder-only, 12.5Hz joint
  source/target streams, temperature sampling (batchable), on-device 1B build.
  Wall: FR/ES/PT/DE→EN only, no Slovak; Zero needs 8–12GB NVIDIA GPU.
  Thesis value: related-work + architecture target, not usable stages today.
- **Moshi** (Kyutai, full-duplex dialogue, 200ms practical, MLX Mac builds):
  EN/FR only. Same wall.
- Our streaming plan (resident STT + word-window MT + Piper) is the pragmatic
  Hibiki-shape on stages that cover SK today.

## Local stages that cover Slovak (the actual pipeline)

| Stage | Choice | Rejected (why) |
|---|---|---|
| STT SK | whisper-small, turbo adopted (0.41) | Parakeet-v3-transformers (0.89, no lang conditioning); NeMo-lang-prompt untested |
| STT EN | Parakeet-v3 (0.023) / whisper-base | — |
| MT | Opus-MT CT2 int8, chunked (0.5s/18 sents) | NLLB-600M (600MB, NC license, unneeded) |
| TTS SK | Piper, own fine-tune (male base) | XTTS (no SK), CosyVoice/F5 (no SK/CUDA), Gemini (cloud+EEA) |
| TTS EN | Piper personal v2 (fast) / XTTS zero-shot (reference) | XTTS live (RTF 1.4) |
| TTS CS | Piper jirka generic / XTTS-cs zero-shot | — |

No local Slovak voice base appeared in 2026; `sk_SK-lili` remains the only one,
which is why the male-Czech warmstart was the structural fix, not a hack.

## Decision log

- XTTS fine-tune for speed: rejected (speed is architectural; no SK).
- Synthetic (XTTS-generated) training data: rejected (copy-of-copy ceiling;
  August v1 already tried bootstrap, ears prefer real-only v2).
- Czech-model-on-Slovak-text: rejected (F0 86Hz, accent, RTF 2.7).
- Gemini replica as reference: parked behind EEA block.
