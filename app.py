import uvicorn
from dotenv import load_dotenv

from backend.cuda_libs import preload_pip_cuda_libs
preload_pip_cuda_libs()  # pip-installed CUDA libraries on Linux (no-op elsewhere); must run before any engine import

load_dotenv()  # optional local configuration — untracked, never committed

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from urllib.parse import urlsplit
import posixpath

import asyncio
import json
import os
import sys # Import sys
import uuid
import numpy as np # type: ignore

# Slovak/Czech text in log lines raises UnicodeEncodeError on consoles that default to cp1252 (stock Windows),
# which used to kill the pipeline task. Make the streams UTF-8 so no PYTHONIOENCODING is needed.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import time # Import time for timing logs
from typing import Optional, List # Ensure List is imported
import threading

# Add the absolute path of the backend directory to sys.path to ensure backend modules are found
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import logging # Import logging
from backend.main import initialize_all_models, handle_audio_stream, get_initialized_models, AUDIO_SAMPLE_RATE, router as backend_router
import backend.main as backend_main # Import backend.main as an alias
from backend.utils.auth import GuestSession, session_from_token
from fastapi import UploadFile, File, Depends, Query # Import Depends, Query
from backend.hardware_scan import autoconfigure, get_config, AutoConfig # type: ignore
from backend.session_manager import ( # type: ignore
    get_or_create_room, remove_room, list_rooms, get_auto_config,
    MeetingRoom, SpeakerIdentity, ListenerIdentity, TranslationResult,
)
from backend.streaming.pipeline import ContinuousStreamingPipeline # type: ignore
import backend.session_manager as _session_manager

class _AudioOnlyStatic(StaticFiles):
    """Serve voice clips but never speaker_voices.json (it lists private local filenames)."""

    async def get_response(self, path, scope):
        if not path.lower().endswith((".wav", ".m4a", ".mp3", ".webm", ".ogg")):
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start periodic cleanup and drain sessions on shutdown."""
    from backend.utils.auth import _jwt_secret
    _jwt_secret()  # persist a random signing secret before serving requests
    # Periodic model pool cleanup task (P0)
    cleanup_task = None
    
    async def periodic_model_cleanup():
        """Periodic cleanup of idle models from the pool."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                from backend.main import _cleanup_idle_models
                _cleanup_idle_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.warning(f"Periodic model cleanup error: {e}")
    
    cleanup_task = asyncio.create_task(periodic_model_cleanup())
    
    try:
        yield
    finally:
        # Graceful shutdown (P0): drain active sessions
        print(f"APP: Initiating graceful shutdown...")
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Wait for active pipeline tasks to complete (with timeout)
        from backend.main import active_sessions
        pending_tasks = []
        for session_data in active_sessions.values():
            task = session_data.get("active_pipeline_task")
            if task and not task.done():
                pending_tasks.append(task)
        
        if pending_tasks:
            print(f"APP: Waiting for {len(pending_tasks)} active pipeline tasks to complete...")
            try:
                await asyncio.wait_for(asyncio.gather(*pending_tasks, return_exceptions=True), timeout=30.0)
            except asyncio.TimeoutError:
                print(f"APP: Shutdown timeout, cancelling remaining tasks")
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()
        
        print(f"APP: Application shutdown completed at {time.strftime('%H:%M:%S', time.localtime(time.time()))}")

def create_app() -> FastAPI:
    _app = FastAPI(lifespan=lifespan)
    _app.include_router(backend_router, prefix="/api")

    class ConnectionManager:
        def __init__(self):
            self.active_connections: List[WebSocket] = []

        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket):
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

        async def send_personal_message(self, message: str, websocket: WebSocket):
            await websocket.send_text(message)

        async def broadcast(self, message: str):
            inactive_connections = []
            for connection in list(self.active_connections):
                try:
                    await connection.send_text(message)
                except WebSocketDisconnect:
                    inactive_connections.append(connection)
                except RuntimeError as e:
                    print(f"Error broadcasting to a connection: {e}")
                    inactive_connections.append(connection)
            
            for connection in inactive_connections:
                self.active_connections.remove(connection)

    manager = ConnectionManager()

    def authenticated_session(token):
        if not token:
            return None
        try:
            return session_from_token(token)
        except HTTPException:
            return None

    @_app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
        return response

    guest_sockets: set = set()  # live guest WebSockets; capped so an open demo cannot exhaust the machine

    @_app.middleware("http")
    async def protect_research_media(request: Request, call_next):
        path = posixpath.normpath(request.url.path.replace("\\", "/"))
        # The Voice Lab is a private research room: development only (BP_LAB=1), never on a shared server.
        if path.startswith("/ui/voice-lab") and os.environ.get("BP_LAB") != "1":
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        # A generated research manifest is not a public-demo consent receipt.
        if path == "/ui/voice-lab/library.json" or path.startswith(("/voice_qc/", "/processed/")):
            return JSONResponse(status_code=403, content={"detail": "Research media is private; use an approved public demo export."})
        if path.startswith("/speaker_voices/"):
            filename = path.removeprefix("/speaker_voices/")
            if not filename.lower().endswith((".wav", ".m4a", ".mp3", ".webm", ".ogg")):
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            auth = request.headers.get("authorization", "")
            token = auth[7:] if auth.lower().startswith("bearer ") else request.query_params.get("token")
            if authenticated_session(token) is None:
                return JSONResponse(status_code=403, content={"detail": "Private voice media."})
            try:
                backend_main.authorize_speaker_wav(filename)
            except HTTPException:
                return JSONResponse(status_code=404, content={"detail": "Not found"})
        response = await call_next(request)
        if path.startswith("/speaker_voices/"):
            response.headers["Cache-Control"] = "private, no-store"
        return response

    # Mount static files for the UI
    # makedirs here (not only in __main__): StaticFiles raises at mount time on a
    # clean checkout where processed/voice_qc doesn't exist yet.
    os.makedirs("processed/voice_qc", exist_ok=True)
    _app.mount("/ui", StaticFiles(directory="ui"), name="ui")
    _app.mount("/ui/images", StaticFiles(directory="ui/images"), name="images") # Explicitly mount images
    _app.mount("/speaker_voices", _AudioOnlyStatic(directory=backend_main.SPEAKER_VOICES_DIR), name="speaker_voices")
    # Voice-lab eval page (ui/voice-lab/): serves QC candidate audio + manifests.
    # Read-only static mount, same pattern as /speaker_voices above.
    _app.mount("/voice_qc", StaticFiles(directory="processed/voice_qc"), name="voice_qc")
    # The Lab's library.json points at ../../processed/<dir>/*.wav. Mount only the audio
    # dirs it references, audio extensions only: processed/ also holds session logs and
    # private recordings that must not be served.
    for _d in ("conversation", "demo_audio", "meeting", "new_models", "gpu_bench", "omni_hq_en",
               "omni_hq_sk", "omnivoice", "stt_input_test"):
        if os.path.isdir(f"processed/{_d}"):
            _app.mount(f"/processed/{_d}", _AudioOnlyStatic(directory=f"processed/{_d}"), name=f"processed_{_d}")
    _app.mount("/processed/voice_qc", _AudioOnlyStatic(directory="processed/voice_qc"), name="processed_voice_qc")

    # Templates for serving HTML
    templates = Jinja2Templates(directory="ui")


    @_app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        return templates.TemplateResponse(request, "home/home.html")  # (request, name) signature: starlette >= 0.29



    @_app.post("/initialize")
    async def initialize_pipeline(
        source_lang: str,
        target_lang: str,
        tts_model_choice: str,
        stt_model_size: str = "base", # Add stt_model_size with default
        vad_enabled_param: bool = True, # Add vad_enabled_param with default
        speaker_wav_path: Optional[str] = None,
        speaker_text: Optional[str] = None,
        speaker_lang: Optional[str] = None,
        current_session: GuestSession = Depends(backend_main.get_current_session),
    ):
        init_start_time = time.time()
        print(f"APP: Received initialization request at {time.strftime('%H:%M:%S', time.localtime(init_start_time))}: source_lang={source_lang}, target_lang={target_lang}, tts_model_choice={tts_model_choice}, stt_model_size={stt_model_size}, vad_enabled_param={vad_enabled_param}, speaker_wav_path={speaker_wav_path}, speaker_text={speaker_text}, speaker_lang={speaker_lang}")
        
        speaker_wav_path = backend_main.authorize_speaker_wav(speaker_wav_path)

        # HTTP preparation is scoped to the anonymous session.
        dummy_client_info = f"http_init_session_{current_session.id}"
        
        # This will trigger the model loading in the background
        result = await initialize_all_models(
            client_info=dummy_client_info, # Pass dummy client_info
            source_lang=source_lang,
            target_lang=target_lang,
            tts_model_choice=tts_model_choice,
            stt_model_size=stt_model_size,
            speaker_wav_path=speaker_wav_path,
            speaker_text=speaker_text,
            speaker_lang=speaker_lang,
            vad_enabled_param=vad_enabled_param
        )
        
        init_models_end_time = time.time()
        print(f"APP: initialize_all_models completed at {time.strftime('%H:%M:%S', time.localtime(init_models_end_time))}. Duration: {init_models_end_time - init_start_time:.2f}s")

        # HTTP initialization belongs to this anonymous session, never broadcast it.
        stored = backend_main.active_sessions.get(dummy_client_info, {})
        result["fully_loaded"] = all(stored.get(k) is not None for k in
                                    ("stt_model", "tts_engine", "vad_instance"))
        return result


    # ------------------------------------------------------------------ hardware
    @_app.get("/api/hardware", summary="Detected hardware + auto-selected pipeline config")
    async def hardware_config():
        cfg = await get_auto_config()
        return {
            "hardware": {
                "os": cfg.hardware.os,
                "arch": cfg.hardware.arch,
                "cpu_cores": cfg.hardware.cpu_cores,
                "ram_gb": cfg.hardware.ram_gb,
                "gpu": cfg.hardware.gpu,
                "gpu_name": cfg.hardware.gpu_name,
                "vram_gb": cfg.hardware.vram_gb,
            },
            "stt": {"device": cfg.stt.device, "compute_type": cfg.stt.compute_type,
                    "model_size": cfg.stt.model_size, "engine": cfg.stt.engine,
                    "notes": cfg.stt.notes},
            "mt": {"device": cfg.mt.device, "compute_type": cfg.mt.compute_type,
                   "model_size": cfg.mt.model_size, "engine": cfg.mt.engine,
                   "notes": cfg.mt.notes},
            "tts": {"device": cfg.tts.device, "compute_type": cfg.tts.compute_type,
                    "model_size": cfg.tts.model_size, "engine": cfg.tts.engine,
                    "notes": cfg.tts.notes},
            "vad": {"device": cfg.vad.device, "engine": cfg.vad.engine,
                    "notes": cfg.vad.notes},
            "streaming": cfg.streaming,
            "max_concurrent_sessions": cfg.max_concurrent_sessions,
            "notes": cfg.notes,
        }

    # ----------------------------------------------------------------------- room mode
    @_app.post("/api/room", summary="Create a room (speaker side); listeners join /ws/listen?room=ID&lang=xx")
    async def room_create(request: Request, current_session: GuestSession = Depends(backend_main.get_current_session)):
        from backend.room import get_room, new_room_id, listen_url, lan_address
        rid = new_room_id(); get_room(rid).owner = current_session.id   # only this session’s speaker session may publish into it
        host = os.environ.get("BP_HOST", "127.0.0.1")
        return {"room_id": rid, "listen_path": f"/ui/listen/listen.html?room={rid}",
                "listen_url": listen_url(str(request.base_url), rid, lan_address()),
                "lan_ready": host not in ("127.0.0.1", "localhost", "::1")}   # False: the server only answers on this computer, phones cannot join (start with BP_HOST=0.0.0.0)

    @_app.get("/api/room/{room_id}/qr", summary="QR code (SVG) of the listener link, speaker side")
    async def room_qr(room_id: str, request: Request, current_session: GuestSession = Depends(backend_main.get_current_session)):
        from backend.room import get_room, listen_url, lan_address, qr_svg
        room = get_room(room_id, create=False)
        if room is None or (room.owner is not None and room.owner != current_session.id):
            raise HTTPException(status_code=404, detail="no such room")
        return Response(qr_svg(listen_url(str(request.base_url), room_id, lan_address())), media_type="image/svg+xml")

    @_app.get("/api/room/{room_id}", summary="Room status: listeners per language")
    async def room_status(room_id: str, current_session: GuestSession = Depends(backend_main.get_current_session)):
        from backend.room import get_room
        room = get_room(room_id, create=False)
        if room is None:
            raise HTTPException(status_code=404, detail="no such room")
        langs: dict = {}
        for l in room.listeners.values():
            langs[l.lang] = langs.get(l.lang, 0) + 1
        return {"room_id": room_id, "listeners": len(room.listeners), "languages": langs, "phrases_published": room.published}

    @_app.websocket("/ws/listen")
    async def listen_websocket(websocket: WebSocket):
        """Room listener: ?room=ID&lang=de&token=TOKEN. Receives JSON {type: room_caption, lang, seq, source, translated} and binary WAV chunks.
        Client may send {"type": "set_lang", "lang": "fr"}."""
        user = authenticated_session(websocket.query_params.get("token"))
        await websocket.accept()
        if user is None:
            await websocket.close(code=4003, reason="Authentication required"); return
        from backend.room import get_room
        room_id = (websocket.query_params.get("room") or "")[:64]
        lang = (websocket.query_params.get("lang") or "en")[:8]
        if not room_id:
            await websocket.send_json({"type": "error", "message": "room is required"}); await websocket.close(); return
        room = get_room(room_id, create=False)     # listeners never create rooms (an id guess must not open one)
        if room is None:
            await websocket.send_json({"type": "error", "message": "no such room (the speaker has not started it, or it has ended)"}); await websocket.close(); return
        from backend.room import pair_unavailable
        why = pair_unavailable(room.src_lang, lang) if room.src_lang else None
        if why:   # tell the listener now, instead of a room that stays silent for them
            await websocket.send_json({"type": "error", "message": why}); await websocket.close(); return
        try:
            listener = room.add_listener(websocket, lang)
        except ValueError as e:
            await websocket.send_json({"type": "error", "message": str(e)}); await websocket.close(); return
        await websocket.send_json({"type": "joined", "room_id": room_id, "lang": lang, "listeners": len(room.listeners)})
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                        if data.get("type") == "set_lang":
                            room.set_language(listener.id, str(data.get("lang", lang))[:8]); lang = listener.lang
                            await websocket.send_json({"type": "lang_set", "lang": lang})
                    except ValueError as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            room.remove_listener(listener.id)

    # ------------------------------------------------------- meeting room mgmt
    @_app.post("/api/meeting/create", summary="Create a meeting room")
    async def meeting_create(room_id: Optional[str] = Query(None),
                             current_session: GuestSession = Depends(backend_main.get_current_session)):
        rid = room_id or f"room-{uuid.uuid4().hex[:8]}"
        room = await get_or_create_room(rid)
        return {"room_id": rid, "max_speakers": room.config.max_concurrent_sessions,
                "config": {
                    "stt": room.config.stt.model_size + "/" + room.config.stt.device,
                    "tts": room.config.tts.engine,
                }}

    @_app.get("/api/meeting/list", summary="List active meeting rooms")
    async def meeting_list():
        return {"rooms": await list_rooms()}

    @_app.delete("/api/meeting/{room_id}", summary="Close a meeting room")
    async def meeting_close(room_id: str,
                            current_session: GuestSession = Depends(backend_main.get_current_session)):
        await remove_room(room_id)
        return {"status": "closed", "room_id": room_id}

    # ---------------------------------------------------------- meeting websocket
    @_app.websocket("/ws/meeting")
    async def meeting_websocket_endpoint(websocket: WebSocket):
        """
        Multi-speaker meeting WebSocket.

        Client messages (JSON):
          {"type":"join","role":"speaker|listener","name":"...",
           "source_lang":"en","target_lang":"sk","voice":"en_US-personal-v2"}
          {"type":"audio","data": <base64 float32 PCM16k mono>}   (speaker only)
          {"type":"start"}                                         (speaker only)
          {"type":"stop"}                                          (speaker only)
          {"type":"leave"}

        Server messages (JSON):
          {"type":"joined","room_id":"...","session_id":"...","role":"speaker"}
          {"type":"translation_result", ...}   (listener: translated text + audio bytes)
          {"type":"transcription_result", ...} (listener: source text)
          {"type":"error","message":"..."}
        """
        token = websocket.query_params.get("token")
        user = authenticated_session(token)
        if user is None:
            await websocket.accept()
            await websocket.close(code=4003, reason="Authentication required")
            return

        await websocket.accept()
        room: Optional[MeetingRoom] = None
        session_id = f"{user.id}-{uuid.uuid4().hex[:8]}"
        role: Optional[str] = None
        pipeline: Optional[ContinuousStreamingPipeline] = None

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message["type"] != "websocket.receive":
                    continue
                if "text" not in message:
                    continue
                import json as _json
                data = _json.loads(message["text"])
                mtype = data.get("type")

                if mtype == "join":
                    role = data.get("role")
                    name = data.get("name", "Speaker")
                    if role == "speaker":
                        source_lang = data.get("source_lang", "en")
                        voice = data.get("voice")
                        tts_engine = data.get("tts_engine", "piper")
                        room = await get_or_create_room(data.get("room_id", "default"))
                        await room.add_speaker(session_id, name, source_lang, voice, tts_engine)
                        pipeline = ContinuousStreamingPipeline(
                            speaker_id=session_id,
                            source_lang=source_lang,
                            target_lang=data.get("target_lang", "sk"),
                            voice=voice or "",
                            tts_engine=tts_engine,
                            config=room.config,
                        )
                        await pipeline.initialize()
                        await websocket.send_json({
                            "type": "joined", "room_id": room.room_id, "session_id": session_id,
                            "role": "speaker",
                        })
                    elif role == "listener":
                        target_lang = data.get("target_lang", "sk")
                        output_device = data.get("output_device")
                        playback_rate = float(data.get("playback_rate", 1.0))
                        room = await get_or_create_room(data.get("room_id", "default"))
                        await room.add_listener(session_id, name, target_lang, output_device, playback_rate)
                        await websocket.send_json({
                            "type": "joined", "room_id": room.room_id, "session_id": session_id,
                            "role": "listener",
                        })
                    else:
                        await websocket.send_json({"type": "error", "message": "role must be 'speaker' or 'listener'"})

                elif mtype == "start":
                    if role != "speaker" or not room:
                        continue
                    session = room.speakers.get(session_id)
                    if session:
                        await session.start(websocket_send=None)

                elif mtype == "stop":
                    if role != "speaker" or not room:
                        continue
                    session = room.speakers.get(session_id)
                    if session:
                        await session.stop()

                elif mtype == "audio":
                    if role != "speaker" or not room or not pipeline:
                        continue
                    import base64 as _b64
                    raw = _b64.b64decode(data["data"])
                    audio_np = np.frombuffer(raw, dtype=np.float32)
                    result = await pipeline.process_audio_chunk(audio_np, AUDIO_SAMPLE_RATE)
                    if result:
                        # Broadcast to listeners matching target_lang
                        await room.broadcast_result(result)
                        await websocket.send_json({
                            "type": "translation_result",
                            "speaker_id": result.speaker_id,
                            "source_text": result.source_text,
                            "translated_text": result.translated_text,
                            "target_lang": result.target_lang,
                            "sequence": result.sequence,
                            "stt_ms": result.stt_ms, "mt_ms": result.mt_ms, "tts_ms": result.tts_ms,
                        })
                        # Send audio as binary message
                        await websocket.send_bytes(result.audio)

                elif mtype == "leave":
                    break

        except WebSocketDisconnect:
            pass
        except Exception:
            logging.exception("Meeting WebSocket session failed")
        finally:
            if role == "speaker" and room:
                await room.remove_speaker(session_id)
            elif role == "listener" and room:
                await room.remove_listener(session_id)
            if pipeline:
                await pipeline.close()
    @_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        # Browser sockets must originate from this host; native clients omit Origin.
        origin = websocket.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            expected_scheme = "https" if websocket.url.scheme == "wss" else "http"
            if parsed.netloc != websocket.headers.get("host") or parsed.scheme != expected_scheme:
                await websocket.close(code=4003, reason="Untrusted origin")
                return
        token = websocket.query_params.get("token")
        user = authenticated_session(token)
        if user is None:
            # Accept only to convey an actionable auth close code; no session/models exist.
            await websocket.accept()
            await websocket.close(code=4003, reason="Authentication required")
            return
        client_info = f"{websocket.client.host}:{websocket.client.port}"
        if len(guest_sockets) >= int(os.environ.get("BP_GUEST_MAX_SESSIONS", "4")):
            await websocket.accept()
            await websocket.close(code=4009, reason="The local service is busy. Try again in a minute.")
            return
        await manager.connect(websocket)
        guest_sockets.add(websocket)
        try:
            await websocket.send_json({"type": "models_loading_status", "fully_loaded": False,
                "route_status": {"pair": "en-sk", "status": "experimental",
                                 "reason": "Select a direction and initialize the installed profile."}})
            # Connecting authenticates only. Configuration explicitly starts model loading.
            await handle_audio_stream(websocket, token=token)
        except WebSocketDisconnect:
            pass
        except Exception:
            logging.exception("WebSocket session failed")
        finally:
            manager.disconnect(websocket)
            guest_sockets.discard(websocket)
            session = backend_main.active_sessions.pop(client_info, None)
            if session:
                task = session.get("active_pipeline_task")
                if task and not task.done():
                    task.cancel()
            backend_main._session_locks.pop(client_info, None)
    return _app


app = create_app()

if __name__ == "__main__":
    # Ensure piper_models and speaker_voices directories exist
    os.makedirs("backend/tts/piper_models", exist_ok=True)
    os.makedirs("speaker_voices", exist_ok=True)
    os.makedirs("processed/voice_qc", exist_ok=True)
    # Loopback by default; exposing the host grants network clients local-tool access.
    # For a conference/LAN demo opt in explicitly: BP_HOST=0.0.0.0
    host = os.environ.get("BP_HOST", "localhost")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"APP: WARNING - listening on {host}; anyone on the network can request a session and use this tool, including local voice references.")
    # Plain HTTP on localhost is a browser "secure context": the microphone works and there is no
    # certificate warning. HTTPS (self-signed certs/, needed for LAN/phone access) is opt-in: BP_HTTPS=1.
    use_tls = os.environ.get("BP_HTTPS") == "1" and os.environ.get("BP_NO_SSL") != "1"
    print(f"APP: serving {'https' if use_tls else 'http'}://{host}:{os.environ.get('BP_PORT', '8000')}")
    uvicorn.run(
        "app:app",
        host=host,
        port=int(os.environ.get("BP_PORT", "8000")),
        ssl_keyfile="./certs/key.pem" if use_tls else None,
        ssl_certfile="./certs/cert.pem" if use_tls else None,
        log_level="info",
        access_log=False,  # /ws?token=... would write session tokens into the log
        limit_concurrency=100,
        backlog=1024,
    )
