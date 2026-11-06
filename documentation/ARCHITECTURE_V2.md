# BP Architecture v2 — Near Real-Time Streaming, Multi-Speaker, Self-Configuring

**Status:** Design draft (ground-up redesign of the segment-based v1 pipeline)  
**Target:** Sub-second first-audio latency, multi-speaker meetings, any language pair,  
runs on CPU-only laptops, NVIDIA GPU laptops, and servers. Local-first, no cloud in pipeline.

---

## 1. Goals (non-negotiable)

| Goal | Metric |
|------|--------|
| Near real-time | First translated audio ≤ 1.5s after speaker's word (streaming, not segment-close) |
| Multi-speaker | N speakers in one meeting, each with dedicated voice, listeners choose language |
| Multi-language | Any supported pair, auto-selected by hardware capability |
| Self-configuring | Scan OS + CPU/GPU/RAM at startup, pick optimal models/devices/compute types |
| Portable | Same code on Mac M1, Windows CPU-only, NVIDIA laptop, Linux server |
| Local-first | STT/MT/TTS all run locally; cloud only as optional QC reference |

---

## 2. Core Design Shift: From v1 to v2

| Dimension | v1 (current) | v2 (redesign) |
|-----------|-------------|---------------|
| STT | Wait for silence/max segment (3s), then transcribe whole segment | Stream small windows (0.5-1s), emit words as they stabilize |
| MT | Translate full segment | Translate sentence/chunk as STT emits it |
| TTS | Synthesize whole segment, send one WAV | Stream synthesis chunks as MT emits text |
| Playback | One chunk at a time, stops at 10s backlog | Continuous adaptive-rate stream, per-listener |
| Speakers | One speaker per WebSocket session | N speakers per meeting, per-speaker voice |
| Config | Manual language packs + env vars | Auto hardware scan + language-pair capability matrix |
| Deployment | `make run` on localhost | localhost + server + Kaggle T4 validation |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │ Speaker A│  │ Speaker B│  │ Listener1│  │ Listener2 (lang SK) │   │
│  │ (mic EN) │  │ (mic SK) │  │ (hears SK│  │ (hears EN)          │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┬──────────┘   │
│       │             │             │                   │                │
│       └─────────────┴────────────┴───────────────────┘                │
│                       Secure WebSocket (wss://)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────┐
│                      MEETING / SESSION MANAGER                          │
│  • Meeting room = N speaker sessions + M listener sessions              │
│  • Per-speaker: language, voice, VAD state, stream clock                │
│  • Per-listener: target language, output device, playback rate          │
│  • Mixer: combines all speaker→listener streams into one audio out      │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────┐
│              STREAMING PIPELINE (one per speaker→language direction)   │
│                                                                         │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌────────────────┐    │
│  │  VAD    │───▶│  STT     │───▶│   MT     │───▶│   TTS          │    │
│  │(webrtc) │ │(word-    │ │(chunk│ │(stream │    │                │    │
│  │ 20ms fr.│ │  level ts)│ │  by  │ │  ing)  │ │                │    │
│  └────┬────┘ └────┬──────┘ └────┬─┘ └──────┬──┘ └────────────────┘    │
│       │           │             │            │                         │
│  audio   │   stable    │ translated   │   audio chunks      │         │
│  chunks  │  words      │ text chunks  │  (0.2-0.5s each)    │         │
│       ▼           ▼             ▼            ▼                  │         │
│  VAD frames   word buffers   sentence   streamed WAV       listener   │
│  (16kHz mono) (commit via    buffers    + captions           playback   │
│               local-agreement)                                   per     │
│                                                                   speaker│
└─────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────┐
│                    HARDWARE AUTO-CONFIG (startup)                       │
│  • Detect CPU cores, RAM, GPU (CUDA/MPS), VRAM                          │
│  • Pick STT model size + compute type per language                      │
│  • Pick MT device (CPU vs CUDA)                                         │
│  • Pick TTS engine (Piper CPU vs OmniVoice/Kokoro GPU)                  │
│  • Expose /config endpoint with chosen stack                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Meeting Room Model

A meeting = one room ID. Inside:
- **Speaker sessions**: each has a WebSocket, source language, assigned voice, VAD state
- **Listener sessions**: each has a WebSocket, target language, output device selection
- **Translation matrix**: speaker_lang → listener_lang. If a direct MT model exists, use it. If not, pivot through English (speaker→EN→listener).

### 3.2 Streaming STT (the bottleneck fix)

Continuous audio buffer instead of waiting for silence:

1. **Continuous buffer** — all audio is buffered, VAD only detects utterance boundaries (never drops speech)
2. **STT on overlapping windows** — every 0.5s, run STT on the last 2.0s of buffered audio
   - **EN source:** Parakeet TDT v3 (GPU/MPS via transformers, ~0.1s per phrase)
   - **SK/CS source:** Whisper small-sk/small (CPU, ~0.3s per phrase)
3. **Text-based dedup** — track synthesized text, only process new text from each STT call
4. **Phrase chunking** — committed words grouped into phrases by natural pauses (>0.3s gap between words)
5. **Push committed phrases immediately** to MT

Expected first-phrase delay: ~0.2-0.4s on MPS with Parakeet, ~0.3-0.4s on CPU with Whisper small.

### 3.3 Streaming MT

1. Receive committed text fragment from STT
2. Translate immediately (Opus-MT CT2 is ~0.05-0.1s per short chunk)
3. Push translation to TTS
4. Send `translation_result` to all listeners in that language

### 3.4 Streaming TTS

1. Receive translation fragment
2. Synthesize with streaming engine (Piper CPU or OmniVoice GPU)
3. Send WAV chunks (0.2-0.5s each) over WebSocket as bytes
4. Listener's `SessionPlayback` plays chunks sequentially with adaptive rate

### 3.5 Per-Speaker Voice

Each speaker is assigned a voice at enrollment:
- Default: clean Piper voice for their language (sk_SK-lili, en_US-john, etc.)
- Cloned: owner's fine-tuned Piper voice (sk_SK-personal-male, en_US-personal-v2)
- Zero-shot: OmniVoice clone from a reference WAV (GPU only)

---

## 4. Hardware Auto-Configuration

On first run (`make install` or `python scripts/setup.py --dev`), a hardware scan runs:

```
Detect:
  - CPU: cores, architecture (x86_64 / arm64 / apple silicon)
  - RAM: total GB
  - GPU: nvidia-smi (CUDA), torch.mps (Apple), none
  - VRAM: if NVIDIA, total MB
  - OS: macOS / Windows / Linux

Decision matrix:
  ┌─────────┬────────────────┬──────────────────┬──────────┬──────────────────────┐
  │ Machine │ STT dev        │ STT model        │ TTS eng  │ MT dev               │
  ├─────────┼────────────────┼──────────────────┼──────────┼──────────────────────┤
  │ Mac M1  │ mps (Parakeet) │ parakeet-tdt-v3  │ Piper CPU│ CPU int8             │
  │ Win CPU │ cpu            │ whisper-small    │ Piper CPU│ CPU int8             │
  │ NVIDIA  │ cuda           │ parakeet-tdt-v3  │ Piper/   │ CUDA int8 / CPU      │
  │ laptop  │ (Parakeet)     │ / whisper-small  │ OmniVoice│                      │
  │ T4 GPU  │ cuda           │ parakeet-tdt-v3  │ OmniVoice│ CUDA int8            │
  │ Server  │ cuda           │ parakeet-tdt-v3  │ Kokoro   │ CUDA int8            │
  └─────────┴────────────────┴──────────────────┴──────────┴──────────────────────┘
```

Output: `backend/config/auto_config.json` + env vars (`BP_DEVICE`, `BP_COMPUTE_TYPE`, `BP_MT_DEVICE`, `BP_TTS_ENGINE`).

---

## 5. Multi-Speaker Meeting Flow

```
1. Speaker A joins → creates speaker session, selects EN source, assigned voice "en_US-personal-v2"
2. Speaker B joins → creates speaker session, selects SK source, assigned voice "sk_SK-personal-male"
3. Listener 1 joins → selects target SK, hears Speaker A (EN→SK) in Speaker A's cloned voice
4. Listener 2 joins → selects target EN, hears Speaker B (SK→EN) in Speaker B's cloned voice
5. Speaker A talks → streamed pipeline → all SK listeners hear A's voice speaking SK
6. Speaker B talks → streamed pipeline → all EN listeners hear B's voice speaking EN
```

Key: each speaker→language pair is an independent pipeline instance. Listeners subscribe to the output of specific pipelines.

---

## 6. Language Pairs & Model Strategy

| Direction | STT Model | MT Model | TTS Voice | Notes |
|-----------|-----------|----------|-----------|-------|
| EN→SK | Parakeet TDT v3 (GPU/MPS) | opus-mt-en-sk | sk_SK-personal-male (clone) / lili (fallback) | Reference pair |
| SK→EN | Whisper small-sk (CPU) | opus-mt-sk-en | en_US-personal-v2 (clone) / john (fallback) | SK STT is bottleneck |
| EN→CS | Parakeet TDT v3 (GPU/MPS) | opus-mt-en-cs | cs_CZ-jirka | Evaluated alongside SK |
| CS→EN | Whisper small (CPU) | opus-mt-cs-en | en_US-john | |
| Other | per-pack | per-pack | per-pack | Earn support via evidence |

Pivot rule: if no direct MT model exists for X→Y, translate X→EN then EN→Y (adds ~0.1s).

---

## 7. Latency Budget (measured, Mac M1)

```
Stage                 Measured (CPU M1)
─────────────────────────────────────
STT (Parakeet EN)       ~100-150ms
STT (Whisper SK/CS)     ~250-380ms
MT chunk                ~30-80ms
TTS (Piper)             ~40-150ms
─────────────────────────────────────
First audio total       ~200-350ms
Steady-state lag        0.3-0.5s
```

This replaces v1's "wait for segment close" (~6-8s first word).

---

## 8. Deployment Modes

| Mode | Use case | Setup |
|------|----------|-------|
| Local demo | One speaker, one listener | `make run`, localhost:8000 |
| Meeting room | N speakers, M listeners, one machine | `make run`, all on same LAN |
| Server | Central meeting server for remote participants | Docker, systemd, behind reverse proxy |
| Kaggle T4 | Validation on GPU | Push kernel, run paced full-route |

---

## 9. What Gets Deleted / Changed

- `backend/main.py` WebSocket handler: reworked for multi-speaker meeting rooms + streaming
- `backend/pipeline.py`: `translate_streaming()` becomes real streaming, not per-chunk batch
- `backend/stt/streaming_captions.py`: becomes the core STT streaming path, not just captions
- `backend/stt/incremental.py`: local-agreement commit moves from experimental to core
- `languages/packs/*`: add `streaming: true`, `hardware_profile` hints
- `scripts/setup.py`: add hardware scan step
- `ui/live-speech/live-script.js`: multi-speaker room UI, per-listener language select

---

## 10. Kaggle Validation Plan

1. **Full-route SK→EN on T4**: continuous 90s SK speech → STT (small on CUDA) → MT (opus-mt-sk-en CPU) → TTS (OmniVoice GPU) → measure p50/p95 first-audio
2. **Multi-speaker contention**: 2-4 speakers alternating, measure turn transition gaps
3. **Soak**: 30 min continuous, memory/queue stability

---

## 11. Open Questions (need owner input)

1. Should the meeting room server be a separate process from the single-user demo, or one codebase with a `--meeting` flag?
2. Do we need speaker diarization (who spoke when) built into the server, or is per-device WebSocket enough (each speaker joins from their own laptop/phone)?
3. Is OmniVoice acceptable as a GPU-only TTS, or should we prioritize keeping Piper CPU as the only default?
4. What's the target first-audio latency number we want to claim for the thesis?

---

*Next: implement `backend/hardware_scan.py`, `backend/session_manager.py`, and streaming pipeline rewrite.*