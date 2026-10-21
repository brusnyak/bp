# BP Pipeline SDK — REST/WS Contract & Python Facade

**Version**: 0.1.0 (2026-10-21)  
**Status**: Experimental — API may change before 1.0

---

## Overview

The BP Pipeline SDK provides a clean, typed interface for embedding real-time speech translation (EN↔SK, EN↔CS) into applications. It wraps the full STT→MT→TTS cascade with:

- **Local-first**: No cloud dependencies; models run on-device
- **Licence-aware**: Fail-closed gate blocks non-commercial engines (XTTS, NLLB) by default
- **Model pooling**: Shared instances across requests, bounded memory
- **Config-driven**: Engine selection via `PipelineConfig`, not code changes
- **Async-native**: Built for FastAPI/WebSocket servers and async applications

---

## Installation

```bash
# From the repo root
pip install -e .  # installs backend/ as package
# or
pip install -r requirements.txt
```

Requires: Python 3.11+, `faster-whisper`, `ctranslate2`, `piper-tts`, and converted models in `ct2_models/` + Piper voices in `backend/tts/piper_models/`.

Run `python scripts/setup.py --dev` for one-command setup (venv + deps + models + certs).

---

## Quick Start (Python)

```python
import asyncio
from backend.pipeline import create_pipeline

async def main():
    # Create pipeline for EN→SK with Piper TTS (fast, generic voice)
    pipe = await create_pipeline("en", "sk", tts_engine="piper", stt_model_size="base")
    
    # Text-only translation (MT → TTS)
    result = await pipe.translate("Good morning, welcome to the demo.")
    print(f"Translation: {result.translation}")
    print(f"Latency: {result.latency}")  # {'mt': 0.08, 'tts': 0.12, 'total': 0.21}
    
    # Save audio
    import soundfile as sf
    sf.write("output.wav", result.audio, result.sample_rate)
    
    await pipe.close()

asyncio.run(main())
```

### Audio Input (Full STT→MT→TTS)

```python
import librosa
from backend.pipeline import create_pipeline

async def translate_file():
    pipe = await create_pipeline("sk", "en", tts_engine="piper", stt_model_size="small-sk")
    
    audio, sr = librosa.load("input.m4a", sr=16000, mono=True)
    result = await pipe.translate(audio)
    
    print(f"Transcript: {result.transcript}")
    print(f"Translation: {result.translation}")
    print(f"Detected lang: {result.detected_lang}")
    print(f"STT latency: {result.latency['stt']:.2f}s")
    
    await pipe.close()
```

### Real-time Streaming

```python
async def streaming_example():
    pipe = await create_pipeline("en", "sk")
    
    # Simulate chunks from WebSocket/WebRTC
    chunks = [audio_chunk_1, audio_chunk_2, ...]  # np.ndarray @ 16kHz
    
    for chunk in chunks:
        result = await pipe.translate(chunk)
        # result.audio is ready to play immediately
        play_audio(result.audio, result.sample_rate)
    
    await pipe.close()
```

---

## Configuration (`PipelineConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_lang` | str | **required** | Source language code (e.g., `en`, `sk`, `cs`) |
| `target_lang` | str | **required** | Target language code |
| `stt_engine` | str | `faster_whisper` | STT engine name (registry key) |
| `stt_model_size` | str | `base` | Whisper model size (`base`, `small`, `small-sk`, `medium`, `large-v3-turbo`) |
| `mt_engine` | str | `opus-mt` | MT engine name (`opus-mt`, `nllb`*, `hybrid-mt`*) |
| `tts_engine` | str | `piper` | TTS engine name (`piper`, `piper_sk_personal`, `omnivoice`*, `xtts`*, `hybrid`*) |
| `beam_size` | int | `5` | Faster-Whisper beam width (1=greedy/fast, 5=quality) |
| `allow_noncommercial` | bool | `False` | Allow CC-BY-NC engines (NLLB, XTTS) |
| `allow_unknown_license` | bool | `False` | Allow unlabelled engines (OmniVoice, Hybrid) |
| `speaker_wav_path` | str? | `None` | Path to speaker reference for cloning engines |
| `speaker_text` | str? | `None` | Transcript of speaker reference |
| `speaker_lang` | str? | `None` | Language of speaker reference |
| `pool_stt` | int | `3` | Max pooled STT instances |
| `pool_mt` | int | `5` | Max pooled MT instances |
| `pool_tts` | int | `5` | Max pooled TTS instances |

* = Requires explicit licence override

---

## Engine Registry

Engines are registered in `backend/registry.py` with capability manifests:

```python
from backend.registry import engines, select, instantiate

# List all engines per stage
for cap in engines("stt"):
    print(cap.name, cap.license, cap.available())

# Licence-gated selection (raises PermissionError if blocked)
cap = select("tts", "xtts", allow_noncommercial=True)

# Instantiate (gate + availability + factory)
tts = instantiate("tts", "piper_sk_personal")
```

### Current Engines (2026-10-21)

| Stage | Engine | Licence | Status | Notes |
|-------|--------|---------|--------|-------|
| VAD | `webrtcvad` | commercial-ok | ✅ | Default, streaming |
| VAD | `silero` | commercial-ok | declared | Used via streaming STT path |
| STT | `faster_whisper` | commercial-ok | ✅ | `base`, `small`, `small-sk`, `large-v3-turbo` |
| MT | `opus-mt` | commercial-ok | ✅ | Helsinki-NLP Opus-MT, per-pair CT2 |
| MT | `nllb` | non-commercial | 🔒 | NLLB-200 (CC-BY-NC-4.0) |
| MT | `hybrid-mt` | unknown | 🔒 | Component licences must be verified |
| TTS | `piper` | commercial-ok | ✅ | CPU baseline, ~60MB/voice |
| TTS | `piper_sk_personal` | commercial-ok | ✅ | Owner's fine-tuned SK voice |
| TTS | `piper_personal` | commercial-ok | ✅ | Owner's fine-tuned EN voice |
| TTS | `omnivoice` | unknown | 🔒 | Cloning, needs GPU (~5.8GB VRAM) |
| TTS | `xtts` | non-commercial | 🔒 | Coqui XTTS v2 (CPML) |
| TTS | `hybrid` | unknown | 🔒 | Inherits component licences |
| TTS | `rvc-mlx` | commercial-ok | declared | MLX voice conversion (Mac only) |
| TTS | `rvc-onnx` | commercial-ok | declared | Cross-platform RVC via ONNX |
| Unified | `qwen3-omni` | commercial-ok | declared | Planned, not implemented |

**Licence policy**: `commercial-ok` = ships by default; `non-commercial` = blocked unless `allow_noncommercial=True`; `unknown` = blocked unless `allow_unknown_license=True` (confirm model card first).

---

## REST API (FastAPI)

Base URL: `https://localhost:8000` (self-signed cert in `certs/`)

### Public Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — per-stage readiness, active sessions, pool stats. No auth. |
| GET | `/api/capabilities` | Engine availability, hardware backends, Piper models, QC scores. No auth. |
| GET | `/auth/config` | Google OAuth client ID (or null). No auth. |
| POST | `/register` | User registration (email, username, password). Rate limited. |
| POST | `/token` | Password login → JWT. Rate limited. |
| POST | `/login` | JSON login → JWT + user info. Rate limited. |
| POST | `/auth/google` | Google ID token → JWT. Rate limited. |

### Authenticated Endpoints (Bearer JWT)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/initialize` | Initialize session models (STT, MT, TTS, VAD). Returns session config. |
| POST | `/voices/upload` | Upload speaker voice (webm/wav/mp3) → converted to 16kHz WAV. |
| POST | `/voices/process` | Segment enrollment session into training corpus (approval-gated). |
| GET | `/voices/list` | List user's uploaded voices. |
| GET | `/voices/file/{filename}` | Download voice file (auth-aware). |
| GET | `/api/ratings` | Export ear-grading data (for Voice Lab). |
| GET | `/metrics` | Detailed metrics — pool stats, per-session detail, capacity. |
| WS | `/ws` | **WebSocket** — real-time audio streaming (see below). |

---

## WebSocket Protocol (`/ws`)

### Connection

```
wss://localhost:8000/ws?token=<JWT>
```

### Client → Server Messages

| Type | Payload | Description |
|------|---------|-------------|
| `binary` | Raw audio bytes (16-bit PCM, 16kHz, mono) | Audio chunk for processing |
| `config_update` | `{source_lang, target_lang, tts_model_choice, stt_model_size, beam_size, vad_enabled}` | Change session config mid-stream |

### Server → Client Messages

| Type | Payload | Description |
|------|---------|-------------|
| `transcription_result` | `{transcribed, detected_lang, latency_s}` | Final STT result |
| `translation_result` | `{translated, latency_s}` | MT result |
| `caption_partial` | `{text, is_final}` | Streaming caption (if available) |
| `tts_audio` | `binary` | Synthesized audio (16-bit PCM, 22050Hz or 16kHz) |
| `mt_conversion_status` | `{status, model_name, error?}` | MT model conversion progress |
| `error` | `{message}` | Pipeline error |

### Example Flow

```javascript
const ws = new WebSocket("wss://localhost:8000/ws?token=...");
ws.binaryType = "arraybuffer";

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: "config_update",
        source_lang: "en",
        target_lang: "sk",
        tts_model_choice: "piper",
        stt_model_size: "base"
    }));
};

// Send audio chunks (16kHz mono PCM)
mediaRecorder.ondataavailable = (e) => ws.send(e.data);

ws.onmessage = (event) => {
    if (typeof event.data === "string") {
        const msg = JSON.parse(event.data);
        if (msg.type === "transcription_result") updateTranscript(msg.transcribed);
        else if (msg.type === "translation_result") updateTranslation(msg.translated);
    } else {
        // Binary TTS audio — play via Web Audio API
        playAudio(event.data);
    }
};
```

---

## Language Packs

Language pairs are declarative JSON files in `languages/packs/`:

```json
{
  "pair": "en-sk",
  "source": "en",
  "target": "sk",
  "stt": {"engine": "faster_whisper", "model": "base", "language": "en"},
  "mt": {"engine": "opus-mt", "model": "Helsinki-NLP--opus-mt-en-sk"},
  "tts": {"engine": "piper", "voice": "sk_SK-personal-male-medium"},
  "clip": "speaker_voices/me_en_script.m4a",
  "reference": "Good morning, and welcome to this live demonstration...",
  "license": "commercial-ok",
  "notes": "Verified pair"
}
```

### CLI Management

```bash
# List all packs
bp language list

# Show one pack
bp language show en-sk

# Add a new pair (scaffolds from Helsinki-NLP)
bp language add pl-en

# Test a pair (runs STT→MT→TTS on clip, writes eval record)
bp language test en-sk

# Remove a pack
bp language remove pl-en --yes
```

### Status Levels

- **verified**: All components resolve + passing eval record (WER < 0.2)
- **experimental**: Missing component, no eval, or WER ≥ 0.2

---

## SK→EN Direction — Known Limitation

The Slovak→English direction uses `whisper-small-sk` (NaiveNeuron/whisper-small-sk, MIT), which has a **known quality issue**: it produces Czech-like hallucinations.

| Config | WER | STT Time | RTF | Status |
|--------|-----|----------|-----|--------|
| small-sk, beam=5 | 0.23–0.35 | 15–18s | 0.14–0.16 | Default (quality) |
| small-sk, beam=2 | 0.29–0.35 | 11–15s | 0.10–0.12 | **Recommended balance** |
| small-sk, beam=1 | 0.34–0.49 | 9–12s | 0.09 | Fast mode (demo only) |
| base, beam=5 | 0.71 | 8.9s | 0.07 | Unusable for SK |

**Root cause**: The fine-tune itself is flawed (Czech contamination), not a config issue.  
**Workaround**: Use `BP_STT_BEAM_SIZE=1` env var for "fast mode" in demos.  
**Fix needed**: GPU model (Voxtral, Parakeet) or RVC-MLX voice layer.

---

## Model Conversion

Opus-MT models must be converted to CTranslate2 format:

```bash
# Convert a single pair
python backend/mt/convert_opus_mt_to_ct2.py --model_name Helsinki-NLP/opus-mt-en-sk

# Or use the language pack resolver (auto-converts on first use)
bp language test en-sk
```

Models are stored in `ct2_models/Helsinki-NLP--opus-mt-<src>-<tgt>/` with `model.bin` + tokenizer files.

---

## Voice Cloning

### Piper Personal Voices (CPU, fast)

Fine-tuned Piper voices in `backend/tts/piper_models/`:
- `en_US-personal-medium.onnx` — Owner's EN voice
- `en_US-personal-v2.onnx` — v2 EN voice
- `sk_SK-personal-male-medium.onnx` — Owner's SK voice (demo default)

Training: `scripts/finetune_personal_voice.py` (separate venv, ~hours on GPU).

### OmniVoice (GPU, high quality)

```python
# Requires GPU, ~5.8GB VRAM
pipe = await create_pipeline("en", "sk", tts_engine="omnivoice", 
                             allow_unknown_license=True)
```

### RVC-MLX / RVC-ONNX (Voice conversion layer)

Engine-agnostic: train once, apply to any TTS output.
- `rvc-mlx`: Mac only (MLX), `~500MB` model
- `rvc-onnx`: Cross-platform (ONNX Runtime), pack `.pth` + `.index` → `.onyx`

---

## Error Handling

| Exception | Cause | Resolution |
|-----------|-------|------------|
| `PermissionError` | Licence gate blocked engine | Add `allow_noncommercial=True` or `allow_unknown_license=True` (internal only) |
| `KeyError` | Unknown engine name | Check `bp engine list` or `engines(stage)` |
| `RuntimeError` | Engine not installed here | Install deps (e.g., `pip install coqui-tts` for XTTS) |
| `HTTPException 403` | MT licence gate on WS | Session config `allow_noncommercial: true` |
| `FileNotFoundError` | Model/voice not found | Run `bp language test <pair>` to auto-convert |

---

## Monitoring

```bash
# Health check (public)
curl -k https://localhost:8000/health

# Metrics (requires auth)
curl -k -H "Authorization: Bearer <token>" https://localhost:8000/metrics
```

### `/health` Response

```json
{
  "status": "healthy",
  "active_sessions": 2,
  "stages": {
    "stt": {"ready": true, "engines": ["faster_whisper"]},
    "mt": {"ready": true, "engines": ["opus-mt"]},
    "tts": {"ready": true, "engines": ["piper_sk_personal"]},
    "vad": {"ready": true, "engines": ["webrtcvad"]}
  },
  "model_pool": {"total_instances": 4, "by_stage": {"stt": 1, "mt": 1, "tts": 1, "vad": 1}},
  "timestamp": 1729500000.0
}
```

---

## Testing

```bash
# Unit tests (38 tests)
make test

# Engine quality gates
bp engine gate stt faster_whisper
bp engine gate mt opus-mt
bp engine gate tts piper
bp engine gate vad webrtcvad

# Full pipeline smoke test (needs running server)
make run
.venv/bin/python scripts/live/interrupt_smoke_test.py

# Language pack eval
bp language test en-sk
bp language test sk-en
```

---

## Roadmap to 1.0

- [ ] Stable REST/WS contract (OpenAPI spec)
- [ ] Python wheel on PyPI
- [ ] TypeScript client (`npm i @bp/pipeline`)
- [ ] Docker image with models baked in
- [ ] Kubernetes deployment manifests
- [ ] Multi-speaker diarization support
- [ ] Streaming STT (WhisperLiveKit/SimulStreaming integration)

---

## License

MIT — see `LICENSE`. Third-party models carry their own licences (Opus-MT: CC-BY-4.0 per pair; Whisper: MIT; Piper: MIT; XTTS: CPML non-commercial; NLLB: CC-BY-NC-4.0). The SDK enforces licence gates at runtime.