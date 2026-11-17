# BP — LHC technical handoff

## One-paragraph summary

BP is a local, self-hosted speech translation and live captions pipeline: microphone → VAD → speech recognition → translation → voice → speakers/captions, all processing on your own machine after model installation. The demonstrated route is English → Slovak (also Czech, German available), with measured median lag 0.79 s / p90 1.39 s on a Mac M1 Pro using generated English audio. Slovak input and cloned voices remain experimental; room mode (one speaker, many listeners) is built but not yet run with real models.

## Precise infrastructure requests

| Item | Minimum | Recommended for demo |
|---|---|---|
| Machine | Apple Silicon Mac or NVIDIA GPU box | 32 GB+ Apple Silicon Mac dedicated to the demo, or NVIDIA L4/A10/A100 |
| VRAM | — | ≥ 8 GB (Voxtral fp16 on T4 uses 8.25 GB); 24 GB not an evidence-based requirement |
| RAM | 16 GB | 32 GB+ (avoid swap pressure that killed Mac Voxtral runs) |
| Network | Loopback (localhost) only | LAN for room listeners: `BP_HOST=0.0.0.0 BP_HTTPS=1`; HTTPS cert warning on phone |
| Audio | Headphones (no feedback) | Virtual cable (BlackHole macOS / VB-Cable Windows / PipeWire Linux) to route into meeting |
| Browser | Chrome or Edge | Chrome/Edge for browser output selection |

**24 GB NVIDIA is a preliminary target, not an evidence-based requirement.** The T4 (16 GB) runs the full chain; the binding constraint is STT real-time factor, not VRAM size. Whisper turbo pads every call to a 30 s window (~0.7 s/call regardless of audio length), which is the Mac floor; a streaming recogniser (Voxtral/Nemotron) removes that floor but needs RTF well under 1.0 to hold lag.

## Audio integration questions (for the LHC coordinator)

1. **Room audio routing:** Will the speaker's microphone feed come through a virtual cable into BP, or is a separate capture device acceptable? Listeners need headphones only; the speaker's mic must not hear the translated audio (feedback).
2. **Language pairs per room:** Room mode supports one language per listener. How many simultaneous language pairs are needed at the event?
3. **Listener count:** Room mode is unit-tested for 3 and 12 listeners but not yet run with real models — what's the expected maximum concurrent listener count?
4. **Network:** Will listeners join on the same Wi-Fi (plain HTTP OK for loopback speaker)? Is HTTPS required for captive-portal networks?
5. **Recording/retention:** Does the event need session audio/text logged (`BP_LOG_SESSIONS=1`), or must nothing be retained?

## Corrections needed in the proposed email

None identified from this review — the pipeline claims (median 0.79 s / p90 1.39 s / WER 5.2%) are receipt-verified. Do not present batch TTS or model-loading success as proof of real-time cloned-voice performance.

## Where to find things

- Public site (to be republished): `https://brusnyak.github.io/bp/`
- Demo script: `documentation/demo/demo_script.md`
- Readiness report: `documentation/demo/READINESS_REPORT.md`
- Reading script (EN/SK/DE sentences): `documentation/reading_script_multilingual.md`
- Evidence receipts: `documentation/evidence/2026-11-15/`
- Source: `https://github.com/brusnyak/bp`
