# BP Multi-User Server Architecture

**Status:** Design  
**Goal:** A central server runs the full STT→MT→TTS pipeline. Multiple users connect from their own devices (laptops/phones), each with their own virtual mic. Every speaker is translated in near real-time; every listener chooses their target language and hears it in the speaker's dedicated voice.

---

## 1. Deployment Model

```
┌──────────────────────────────────────────────────────────────────────┐
│                     CENTRAL SERVER (BP backend)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ Speaker A   │  │ Speaker B   │  │ Speaker C   │                  │
│  │ (WebSocket) │  │ (WebSocket) │  │ (WebSocket) │                  │
│  │ mic → EN    │  │ mic → SK    │  │ mic → CS    │                  │
│  │ voice: EN   │  │ voice: SK   │  │ voice: CS   │                  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │
│         │                │                │                          │
│  ┌──────▼────────────────▼────────────────▼──────┐                  │
│  │     Streaming Pipeline Pool (per speaker→lang) │                  │
│  │  STT→MT→TTS × N speaker-language pairs         │                  │
│  └──────┬────────────────┬────────────────┬──────┘                  │
│         │                │                │                          │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐                  │
│  │ Listener 1  │  │ Listener 2  │  │ Listener 3  │                  │
│  │ (hears SK)  │  │ (hears EN)  │  │ (hears EN)  │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
                          ▲
                  HTTPS / WSS (LAN or internet)
```

**Key principle:** The server does all the heavy computation (STT/MT/TTS). Client devices only send microphone audio and receive translated audio + captions. This means even a phone or thin laptop can participate.

---

## 2. User Modes

### 2.1 Speaker Mode (presenter in auditorium)
- User opens BP on their laptop, clicks "Speaker"
- Selects their source language (e.g., English)
- Selects their voice (generic or cloned)
- Runs a sound check (mic + output)
- Clicks Start — mic audio streams to server via WebSocket
- Hears nothing translated (they're the source), but sees captions of what others said

### 2.2 Listener Mode (audience member)
- User opens BP on their phone/laptop, clicks "Listener"
- Selects their target language (e.g., Slovak)
- Selects output device (earbuds, speakers, virtual cable to meeting room)
- Clicks Start — receives translated audio chunks + captions from server
- Can adjust playback rate (1.0, 1.25, 1.5, adaptive)

### 2.3 Hybrid Mode (speaker + listener)
- A participant who both speaks and listens
- Has both a speaker session (their mic → server) and listener session (server → their ears)
- Server routes their speaker output to other listeners, not back to them

---

## 3. Presentation Scenario (Auditorium)

**The setup you described:** You're presenting in an auditorium, speaking English, and the audience hears Slovak in real-time.

```
┌─────────────────────────────────────────────────────────────┐
│                    AUDITORIUM                               │
│  ┌─────────────┐                                          │
│  │  PRESENTER  │  (you)  →  laptop mic → BP server → SK   │
│  │   (EN)      │                                        audio│
│  └──────┬──────┘                                          ↓   │
│         │                                          ┌───────────┐│
│         │  WSS (LAN)                               │  AUDIENCE ││
│         │                                          │ phones/   ││
│         └─────────────────────────────────────────→│ laptops   ││
│                                                    │ hear SK   ││
│                                                    └───────────┘│
└─────────────────────────────────────────────────────────────┘
```

**Difference from initial single-user design:**
| Aspect | Initial (single-user) | Multi-user server |
|--------|----------------------|-------------------|
| Where STT/MT/TTS runs | Presenter's laptop | Central server |
| Who connects | One browser tab | N devices (speaker + listeners) |
| Audio source | Local mic | Each speaker's device mic |
| Audio output | Local speakers | Each listener's chosen device |
| Languages | One direction | N source → M target matrix |
| Scalability | 1 user | Limited by server (4-8 concurrent on T4) |

---

## 4. Server-Side Processing Flow

```
1. Speaker connects via WSS, joins room, selects source_lang + voice
2. Server creates SpeakerSession + StreamingPipeline (STT→MT→TTS)
3. Speaker sends 20ms PCM frames over WebSocket (base64)
4. Server VAD-gates → STT window → local-agreement → fragment
5. Fragment → MT (target languages of all listeners in room)
6. Each MT output → TTS (speaker's assigned voice)
7. Audio chunks broadcast to listeners matching each target_lang
8. Listener's browser plays chunks sequentially with adaptive rate
```

**Per-speaker pipeline instances:** If 3 speakers each have 2 listener languages, that's 6 pipeline instances running concurrently on the server.

**Capacity:** Limited by hardware auto-config (see `backend/hardware_scan.py`):
- Mac M1 8GB: 2 speakers, 4 listeners
- NVIDIA T4 16GB: 4 speakers, 8 listeners
- NVIDIA laptop 8GB VRAM: 4 speakers, 8 listeners
- Server 32GB + GPU: 8+ speakers, 16+ listeners

---

## 5. API Endpoints (Multi-User)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/hardware` | GET | Server auto-detected hardware + selected config |
| `/api/meeting/create` | POST | Create a meeting room (returns room_id) |
| `/api/meeting/list` | GET | List active rooms |
| `/api/meeting/{room_id}` | DELETE | Close a room |
| `/ws/meeting` | WebSocket | Join as speaker or listener |
| `/ws` | WebSocket | Legacy single-user (backward compatible) |

### 5.1 WebSocket Protocol

**Client → Server:**
```json
{"type":"join","role":"speaker","name":"Alice","source_lang":"en","target_lang":"sk","voice":"en_US-personal-v2"}
{"type":"join","role":"listener","name":"Bob","target_lang":"sk","playback_rate":"adaptive"}
{"type":"start"}                                    // speaker only
{"type":"stop"}                                     // speaker only
{"type":"audio","data":"<base64 PCM16k mono>"}      // speaker only
{"type":"leave"}
```

**Server → Client:**
```json
{"type":"joined","room_id":"room-abc","session_id":"sess-123","role":"speaker"}
{"type":"translation_result","speaker_id":"...","source_text":"Hello","translated_text":"Ahoj","target_lang":"sk","sequence":1,"stt_ms":550,"mt_ms":80,"tts_ms":140}
{"type":"transcription_result","speaker_id":"...","text":"Hello","is_final":false}
{"type":"error","message":"..."}
```

Audio is sent as binary WebSocket messages (WAV bytes).

---

## 6. Virtual Mic / Output Routing

Each listener selects their output device:
- **Speakers/headphones** — default, hear translation locally
- **Virtual cable (BlackHole / VB-Cable)** — route translation into a meeting app (Teams/Zoom) or room audio system
- **Adaptive playback rate** — speeds up when behind, pitch preserved

Each speaker selects their input device:
- **Physical mic** — laptop microphone
- **Virtual mic** — audio from another app routed into BP

---

## 7. Hardware Requirements for Server

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU only) | NVIDIA CUDA 8+ GB VRAM |
| OS | macOS / Windows / Linux | Linux server |
| Network | LAN | LAN + WSS for remote |

**Auto-config:** `backend/hardware_scan.py` runs at startup, detects all of the above, selects optimal models/devices. Exposed via `/api/hardware`.

---

## 8. What's Built vs What's Next

| Component | Status |
|-----------|--------|
| Hardware auto-config | ✅ `backend/hardware_scan.py` |
| Meeting session manager | ✅ `backend/session_manager.py` |
| Streaming pipeline | ✅ `backend/streaming/pipeline.py` |
| `/ws/meeting` endpoint | ✅ In `app.py` |
| `/api/hardware` endpoint | ✅ In `app.py` |
| `/api/meeting/*` endpoints | ✅ In `app.py` |
| Multi-user UI (speaker/listener join) | ❌ Needs frontend work |
| Server deployment (Docker/systemd) | ❌ Not started |
| Virtual cable documentation | ❌ Not written |
| Kaggle T4 validation | 🔄 Running (kernel pushed) |

---

## 9. Open Questions

1. **Server deployment:** Docker container vs systemd service vs cloud VM?
2. **Room persistence:** In-memory (lost on restart) vs database-backed rooms?
3. **Authentication:** JWT per user (already built) or anonymous room access?
4. **Max concurrent speakers:** Enforced by hardware auto-config, but what's the default for a university server?
5. **Presenter flow:** Does the presenter also need to hear translated audience questions (reverse direction)?

---

## 10. Next Steps

1. Wait for Kaggle T4 results (p50/p95 first-audio for SK→EN streaming)
2. Build the multi-user UI (speaker/listener join screens on `ui/live-speech/`)
3. Add Dockerfile + docker-compose for server deployment
4. Document virtual cable setup for Teams/Zoom/room audio
5. Define session capacity limits per hardware tier