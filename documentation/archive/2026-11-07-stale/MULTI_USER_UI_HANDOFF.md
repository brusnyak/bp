# Multi-User UI Handoff — Jumpstart Document

**Status:** Ready for implementation in a separate session  
**Purpose:** Give the next session everything needed to build the speaker/listener multi-user UI without re-researching.

---

## 1. What Exists Already (Reuse, Don't Rebuild)

| File | What it provides |
|------|-----------------|
| `ui/live-speech/live.html` | Main translation page — already has mic input, captions, playback, PiP, device selection |
| `ui/live-speech/live-script.js` | WebSocket client, `SessionPlayback` (sequential audio), language selects, sound check, start/stop |
| `ui/live-speech/live-style.css` | Styling for the live page |
| `ui/live-speech/session-playback.js` | One-at-a-time audio player with adaptive rate, queue budget, terminal Stop |
| `ui/live-speech/audio-processor.js` | AudioWorklet for mic capture at 16kHz |
| `ui/home/home.html` | Landing page with direction selection |
| `app.py` `/ws/meeting` | Multi-speaker WebSocket endpoint (speaker + listener roles) |
| `app.py` `/api/hardware` | Auto-detected hardware config |
| `app.py` `/api/meeting/create|list|{room_id}` | Room management |
| `backend/session_manager.py` | MeetingRoom, SpeakerSession, ListenerIdentity classes |
| `documentation/MULTI_USER_SERVER.md` | Server architecture, API protocol, WebSocket messages |

---

## 2. What Needs to Be Built (Multi-User UI)

### 2.1 New Pages

| Page | Purpose | Route |
|------|---------|-------|
| `ui/meeting/join.html` | Speaker or listener chooses role, enters name, selects room | `/ui/meeting/join` |
| `ui/meeting/speaker.html` | Speaker interface: mic, sound check, start/stop, captions of own speech | `/ui/meeting/speaker` |
| `ui/meeting/listener.html` | Listener interface: target language select, output device, playback rate, translated captions + audio | `/ui/meeting/listener` |

### 2.2 WebSocket Protocol (from `app.py` `/ws/meeting`)

**Client → Server (JSON):**
```json
{"type":"join","role":"speaker","name":"Alice","source_lang":"en","target_lang":"sk","voice":"en_US-personal-v2"}
{"type":"join","role":"listener","name":"Bob","target_lang":"sk","playback_rate":"adaptive"}
{"type":"start"}                                    // speaker: begin capture
{"type":"stop"}                                     // speaker: end capture
{"type":"audio","data":"<base64 PCM16k mono>"}      // speaker: audio frame (20ms)
{"type":"leave"}
```

**Server → Client (JSON + binary WAV):**
```json
{"type":"joined","room_id":"room-abc","session_id":"sess-123","role":"speaker"}
{"type":"translation_result","speaker_id":"...","source_text":"Hello","translated_text":"Ahoj","target_lang":"sk","sequence":1,"stt_ms":550,"mt_ms":80,"tts_ms":140}
{"type":"transcription_result","speaker_id":"...","text":"Hello","is_final":false}
{"type":"error","message":"..."}
```

Audio is sent as binary WebSocket messages (WAV bytes) interleaved with JSON.

---

## 3. Concrete File Edits

### 3.1 `app.py` — Add meeting page routes

Add after the existing static mounts (around line 270):

```python
@_app.get("/ui/meeting/join", response_class=HTMLResponse)
async def meeting_join(request: Request):
    return templates.TemplateResponse(request, "meeting/join.html")

@_app.get("/ui/meeting/speaker", response_class=HTMLResponse)
async def meeting_speaker(request: Request):
    return templates.TemplateResponse(request, "meeting/speaker.html")

@_app.get("/ui/meeting/listener", response_class=HTMLResponse)
async def meeting_listener(request: Request):
    return templates.TemplateResponse(request, "meeting/listener.html")
```

Mount the meeting directory:
```python
os.makedirs("ui/meeting", exist_ok=True)
_app.mount("/ui/meeting", StaticFiles(directory="ui/meeting"), name="meeting")
```

### 3.2 `ui/meeting/join.html` — Role selection page

Minimal structure:
- Room ID input (optional, auto-generate if empty)
- Name input
- Role buttons: "Speaker" / "Listener"
- Source language select (speaker only)
- Target language select (both)
- Voice select (speaker only)
- "Join meeting" button → navigates to speaker.html or listener.html with query params

### 3.3 `ui/meeting/speaker.html` — Speaker interface

Reuses `ui/live-speech/live-script.js` patterns:
- Mic input with VAD level display
- Sound check button (2s record + playback)
- Start/Stop button
- Source language + target language + voice (from join page)
- Captions: own transcript + translation
- PiP button (reuse from live-speech)
- WebSocket: `/ws/meeting?token=<jwt>` with role=speaker

### 3.4 `ui/meeting/listener.html` — Listener interface

- Target language select
- Output device select (speakers/headphones/virtual cable)
- Playback rate: 1.0 / 1.25 / 1.5 / adaptive
- Captions: source text + translated text (from all speakers in room)
- Audio player: `SessionPlayback` (sequential, adaptive rate)
- WebSocket: `/ws/meeting?token=<jwt>` with role=listener

---

## 4. JavaScript Architecture (Reuse Patterns)

### 4.1 Reuse from `ui/live-speech/live-script.js`

Copy/adapt these patterns into the meeting pages:

| Pattern | File | Use in |
|---------|------|--------|
| WebSocket connection + auth token | `setupWebSocket()` | Both speaker & listener |
| Sound check (2s record + playback) | `soundCheckBtn` handler | Speaker |
| Mic capture via AudioWorklet | `startRealtimeAudioStreaming()` | Speaker |
| Send audio frames as base64 | `audio-processor.js` + `websocket.send()` | Speaker |
| Sequential audio playback | `SessionPlayback` from `session-playback.js` | Listener |
| Adaptive playback rate | `playbackRateSelect` + `adaptiveRate()` | Listener |
| Output device selection | `enumerateAudioOutputDevices()` + `setSinkId` | Listener |
| Captions display | `transcriptionOutput` / `translationOutput` | Both |
| PiP subtitles | `pipBtn` + `drawPip()` | Both (optional) |
| Start/Stop | `startStopBtn` | Speaker |

### 4.2 New Code Needed

| Component | Description |
|-----------|-------------|
| `meeting-sync.js` | Shared: room join, query param parsing, WebSocket factory for `/ws/meeting` |
| `speaker-ui.js` | Speaker-specific: mic control, start/stop, own captions |
| `listener-ui.js` | Listener-specific: language select, output device, playback rate, multi-speaker caption mixing |

---

## 5. Multi-Speaker Caption Mixing (Listener)

When multiple speakers talk, the listener needs captions from all of them:

```
Speaker A (EN) → translated to SK → listener hears A's voice speaking SK
Speaker B (SK) → translated to EN → listener hears B's voice speaking EN
```

The listener's WebSocket receives `translation_result` from multiple `speaker_id`s. The UI should:
- Show each speaker's transcript + translation in separate caption lines (color-coded by speaker)
- Play audio sequentially via `SessionPlayback` (one chunk at a time)
- Label each line with speaker name + language

---

## 6. Room Flow

```
1. User opens /ui/meeting/join
2. Enters name, chooses role, selects languages
3. Clicks "Join"
4. If speaker: → /ui/meeting/speaker?room=xxx&name=xxx&source_lang=en&target_lang=sk&voice=...
5. If listener: → /ui/meeting/listener?room=xxx&name=xxx&target_lang=sk
6. Page loads, reads query params, connects to /ws/meeting?token=<jwt>
7. Sends {"type":"join",...}
8. Server responds {"type":"joined",...}
9. Speaker clicks Start → sends audio frames → gets translations back
10. Listener receives translation_result + audio chunks → plays them
```

---

## 7. Auth

Meeting WebSocket uses the same JWT as `/ws`:
- `localStorage.getItem('userToken')` from login
- Pass as `?token=<jwt>` in WebSocket URL
- Guest mode: `BP_GUEST=1` allows token-less demo (see `app.py` guest logic)

---

## 8. Testing Checklist (Next Session)

- [ ] Join page: room creation, role selection, language selects
- [ ] Speaker page: mic capture, sound check, start/stop, WebSocket connection
- [ ] Listener page: language select, output device, playback rate, audio playback
- [ ] Multi-speaker: 2 speakers, 1 listener, captions from both, audio mixed sequentially
- [ ] Room persistence: join same room from 2 browsers, both see each other's translations
- [ ] Stop: speaker stops, audio stops, captions freeze
- [ ] Device loss: unplug headphones, output falls back to default with notice

---

## 9. Files to Create

```
ui/meeting/
├── join.html        # Role selection
├── speaker.html     # Speaker interface
├── listener.html    # Listener interface
├── meeting-sync.js  # Shared WebSocket + room logic
├── speaker-ui.js    # Speaker-specific UI
├── listener-ui.js   # Listener-specific UI
├── meeting-style.css# Shared styling
└── audio-processor.js # (copy from ui/live-speech/)
```

---

## 10. Notes

- The `/ws/meeting` endpoint in `app.py` is already implemented — no backend changes needed for the UI handoff
- `backend/session_manager.py` handles room/speaker/listener state on the server
- `documentation/MULTI_USER_SERVER.md` has the full server architecture
- `documentation/MULTI_SPEAKER.md` has the translation matrix and per-speaker voice design
- Keep UI plain HTML/CSS/JS (match existing project style — no React/Vue)

---

**Ready for implementation.** Copy the patterns from `ui/live-speech/`, create the 3 HTML pages + 3 JS files, add 3 routes to `app.py`.