# BP v2 Multi-Speaker Meeting Design

**Status:** Design  
**Goal:** Support N speakers + M listeners in one meeting room, each speaker with a dedicated voice, each listener choosing their target language.

---

## 1. Use Cases

| Scenario | Speakers | Listeners | Languages |
|----------|----------|-----------|-----------|
| Lecture (Mode A) | 1 | Many | 1→N (one source, many targets) |
| Bilingual meeting | 2 | 2-10 | 2↔2 (each speaks their own language) |
| Multilingual conference | 5+ | 20+ | N→M (matrix) |
| Remote hybrid room | 1+ local, 1+ remote | Many | configurable |

---

## 2. Architecture

### 2.1 Meeting Room

```
MeetingRoom (room_id)
  ├── SpeakerSession A  → WebSocket, mic EN, voice "en_US-personal-v2"
  ├── SpeakerSession B  → WebSocket, mic SK, voice "sk_SK-personal-male"
  ├── ListenerSession 1 → WebSocket, target SK, hears A→SK
  └── ListenerSession 2 → WebSocket, target EN, hears B→EN
```

Each speaker→language pair is an independent `StreamingPipeline` instance. Listeners subscribe to pipeline outputs matching their target language.

### 2.2 Translation Matrix

| Speaker Lang | Listener Lang | Path |
|-------------|--------------|------|
| EN | SK | Direct MT: opus-mt-en-sk |
| SK | EN | Direct MT: opus-mt-sk-en |
| EN | EN | Passthrough (no MT) |
| SK | SK | Passthrough (no MT) |
| CS | EN | Direct MT: opus-mt-cs-en |
| X | Y (no direct model) | Pivot: X→EN→Y (two MT hops, ~0.1s extra) |

### 2.3 Per-Speaker Voice

Each speaker is assigned a voice at join time:

| Voice Type | Source | Use case |
|------------|--------|----------|
| Generic Piper | sk_SK-lili, en_US-john, cs_CZ-jirka | Default, no enrollment needed |
| Personal Piper | sk_SK-personal-male, en_US-personal-v2 | Owner's cloned voice |
| Zero-shot OmniVoice | reference WAV | GPU only, any speaker |

The voice is chosen from `speaker_voices/` (user-uploaded) or defaults. Each speaker's TTS output uses their assigned voice, so listeners hear each speaker in a distinct voice.

### 2.4 Listener Experience

- Listener joins a room, selects target language
- Receives `translation_result` JSON (source text, translated text, timing)
- Receives audio chunks (WAV bytes) played sequentially
- Can adjust playback rate (1.0, 1.25, 1.5, adaptive)
- Can select output device (speakers, headphones, virtual cable)

---

## 3. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/meeting/create` | POST | Create a room (returns room_id) |
| `/api/meeting/list` | GET | List active rooms |
| `/api/meeting/{room_id}` | DELETE | Close a room |
| `/ws/meeting` | WebSocket | Join as speaker or listener |
| `/api/hardware` | GET | Detected hardware + auto-config |

### 3.1 WebSocket Messages

**Client → Server:**
```json
{"type":"join","role":"speaker","name":"Alice","source_lang":"en","target_lang":"sk","voice":"en_US-personal-v2"}
{"type":"join","role":"listener","name":"Bob","target_lang":"sk","playback_rate":"adaptive"}
{"type":"start"}                                    // speaker: begin capture
{"type":"stop"}                                     // speaker: end capture
{"type":"audio","data":"<base64 PCM16k mono>"}      // speaker: audio frame
{"type":"leave"}
```

**Server → Client:**
```json
{"type":"joined","room_id":"room-abc","session_id":"sess-123","role":"speaker"}
{"type":"translation_result","speaker_id":"...","source_text":"Hello","translated_text":"Ahoj","target_lang":"sk","sequence":1,"stt_ms":550,"mt_ms":80,"tts_ms":140}
{"type":"transcription_result","speaker_id":"...","text":"Hello","is_final":false}
{"type":"error","message":"..."}
```

Audio is sent as binary WebSocket messages (WAV bytes) interleaved with JSON.

---

## 4. Capacity Limits

| Machine | Max speakers | Max listeners | Notes |
|---------|-------------|---------------|-------|
| Mac M1 8GB RAM | 2 | 4 | CPU path |
| Windows CPU 8GB | 2 | 4 | CPU path |
| NVIDIA T4 16GB | 4 | 8 | GPU path |
| NVIDIA laptop 8GB VRAM | 4 | 8 | GPU path |
| Server 32GB + GPU | 8+ | 16+ | Configurable |

Enforced by `max_concurrent_sessions` from hardware auto-config.

---

## 5. What Is NOT Built Yet

| Feature | Status | Notes |
|---------|--------|-------|
| Speaker diarization (shared mic) | Not started | Each speaker needs their own WebSocket/device |
| Server-side audio mixing | Not started | Each listener gets their own stream; mixing is future |
| Per-language broadcast (one STT → many MT/TTS) | Not started | Currently one pipeline per speaker→language |
| Room persistence/rejoin | Not started | Rooms are in-memory, lost on restart |

---

## 6. LHC Integration Fit

| LHC scenario | BP v2 fit |
|-------------|-----------|
| One remote person, EN↔SK | ✅ Single speaker, one listener, direct route |
| Per-seat earbuds (each own language) | ✅ N listeners, each picks target language |
| Remote room join | ⚠️ Needs room audio capture (virtual cable) |
| Accenture platform integration | ❌ Unknown API — not promised |

The LHC room audio contract (mic input, return channels, DSP/mixer) must be confirmed before deployment.