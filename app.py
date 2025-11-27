import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
import os
import sys # Import sys
import time # Import time for timing logs
from typing import Optional, List # Ensure List is imported
import threading

# Add the absolute path of the backend directory to sys.path to ensure backend modules are found
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import logging # Import logging
from backend.main import initialize_all_models, handle_audio_stream, get_initialized_models, router as backend_router
import backend.main as backend_main # Import backend.main as an alias
from backend.main import User, get_password_hash # Import User and get_password_hash for default user creation
from sqlalchemy.orm import Session # Import Session
from backend.utils.db_manager import get_db_session_and_engine, SQLALCHEMY_DATABASE_URL, get_db, init_db # Import init_db
from fastapi import UploadFile, File, Depends # Import Depends

# Define module-level engine and SessionLocal for non-test runs
_default_engine, _default_SessionLocal = get_db_session_and_engine(SQLALCHEMY_DATABASE_URL)

# Function to create a default user if no users exist, now accepts SessionLocal
async def create_default_user_if_empty(db_session_local):
    with db_session_local() as db:
        if db.query(User).count() == 0:
            print("APP: No users found in the database. Creating a default user.")
            default_email = "test@example.com"
            default_password = "password"
            hashed_password = get_password_hash(default_password)
            print(f"APP: Hashed password for default user: {hashed_password[:10]}...") # Log first 10 chars
            default_user = User(username="testuser", email=default_email, hashed_password=hashed_password)
            db.add(default_user)
            db.commit()
            db.refresh(default_user)
            print(f"APP: Default user '{default_email}' created successfully.")
        else:
            print("APP: Users already exist in the database. Skipping default user creation.")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI, db_session_local_override=None):
    """
    Handles startup and shutdown events for the FastAPI application.
    Uses db_session_local_override if provided, otherwise uses the default.
    """
    startup_time = time.time()
    print(f"APP: Application startup initiated at {time.strftime('%H:%M:%S', time.localtime(startup_time))}")
    
    current_db_session_local = db_session_local_override if db_session_local_override else _default_SessionLocal
    current_engine = current_db_session_local.kw["bind"] # Get engine from sessionmaker

    init_db(current_engine) # Initialize database tables using the correct engine
    await create_default_user_if_empty(current_db_session_local)
    yield
    print(f"APP: Application shutdown completed at {time.strftime('%H:%M:%S', time.localtime(time.time()))}")

def create_app(db_session_local_override=None) -> FastAPI:
    _app = FastAPI(lifespan=lambda app: lifespan(app, db_session_local_override))
    _app.include_router(backend_router, prefix="/api")

    # Dependency to get the database session for the app
    def get_db_dependency_for_app():
        db = db_session_local_override() if db_session_local_override else _default_SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    # Override the get_db_dependency in backend.main with the app's specific one
    _app.dependency_overrides[backend_main.get_db_dependency] = get_db_dependency_for_app

    class ConnectionManager:
        def __init__(self):
            self.active_connections: List[WebSocket] = []

        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket):
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

    # Mount static files for the UI
    _app.mount("/ui", StaticFiles(directory="ui"), name="ui")
    _app.mount("/ui/images", StaticFiles(directory="ui/images"), name="images") # Explicitly mount images
    _app.mount("/speaker_voices", StaticFiles(directory="speaker_voices"), name="speaker_voices")

    # Templates for serving HTML
    templates = Jinja2Templates(directory="ui")


    @_app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        return templates.TemplateResponse("home/home.html", {"request": request})



    @_app.post("/initialize")
    async def initialize_pipeline(
        source_lang: str,
        target_lang: str,
        tts_model_choice: str,
        stt_model_size: str = "base", # Add stt_model_size with default
        vad_enabled_param: bool = True, # Add vad_enabled_param with default
        speaker_wav_path: Optional[str] = None,
        speaker_text: Optional[str] = None,
        speaker_lang: Optional[str] = None
    ):
        init_start_time = time.time()
        print(f"APP: Received initialization request at {time.strftime('%H:%M:%S', time.localtime(init_start_time))}: source_lang={source_lang}, target_lang={target_lang}, tts_model_choice={tts_model_choice}, stt_model_size={stt_model_size}, vad_enabled_param={vad_enabled_param}, speaker_wav_path={speaker_wav_path}, speaker_text={speaker_text}, speaker_lang={speaker_lang}")
        
        # For the /initialize endpoint, we use a dummy client_info as it's not a WebSocket
        dummy_client_info = "http_init_client" 
        
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

        # After models are initialized, broadcast a status update to all connected websockets
        # This ensures that any client that connects after initialization, or is already connected,
        # receives the correct status.
        
        # Create a dummy session_config for get_initialized_models
        dummy_session_config = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "tts_model_choice": tts_model_choice,
            "stt_model_size": stt_model_size,
            "speaker_wav_path": speaker_wav_path,
            "speaker_text": speaker_text,
            "speaker_lang": speaker_lang,
            "vad_enabled": vad_enabled_param,
        }
        (
            current_stt_model,
            current_main_mt_model,
            current_tts_model_instance, # This will be either CoquiTTS or PiperTTS
            current_vad_instance,
            current_tts_choice_from_main,
        ) = get_initialized_models(dummy_client_info, dummy_session_config) # Pass dummy client_info
        
        status_check_time = time.time()
        print(f"APP: Model status check initiated at {time.strftime('%H:%M:%S', time.localtime(status_check_time))}")

        if (
            current_stt_model is not None
            and current_main_mt_model is not None
            and current_vad_instance is not None
            and current_tts_model_instance is not None # Check if any TTS model is initialized
        ):
            print(f"APP: Broadcasting models_loading_status: fully_loaded=True at {time.strftime('%H:%M:%S', time.localtime(time.time()))}")
            await manager.broadcast(
                json.dumps(
                    {
                        "type": "models_loading_status",
                        "loading_started": True,
                        "fully_loaded": True,
                    }
                )
            )
        else:
            # This case should ideally not be reached if initialize_models completes successfully
            print(
                f"APP: Broadcasting models_loading_status: fully_loaded=False (error or incomplete initialization) at {time.strftime('%H:%M:%S', time.localtime(time.time()))}"
            )
            await manager.broadcast(
                json.dumps(
                    {
                        "type": "models_loading_status",
                        "loading_started": False,
                        "fully_loaded": False,
                    }
                )
            )
        
        init_end_time = time.time()
        print(f"APP: Initialization endpoint finished at {time.strftime('%H:%M:%S', time.localtime(init_end_time))}. Total endpoint duration: {init_end_time - init_start_time:.2f}s")

        return result


    @_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown:unknown"
            print(
                f"App: WebSocket accepted from {client_info}"
            )
            
            # Initialize session-specific configuration for the new WebSocket
            # This will create an entry in active_sessions if it doesn't exist
            # and store the default session_config.
            await initialize_all_models(
                client_info=client_info,
                source_lang="en", # Default values for initial connection
                target_lang="sk",
                tts_model_choice="piper",
                stt_model_size="base",
                speaker_wav_path=None,
                speaker_text=None,
                speaker_lang=None,
                vad_enabled_param=True,
                websocket=websocket # Pass websocket for MT conversion status
            )

            # Retrieve models from the newly created/updated session
            session_data = backend_main.active_sessions.get(client_info)
            if session_data:
                current_stt_model = session_data.get("stt_model")
                current_main_mt_model = session_data["mt_models"].get(f"{session_data['session_config']['source_lang']}-{session_data['session_config']['target_lang']}")
                current_tts_model_instance = session_data.get("piper_tts_model") or session_data.get("coqui_tts_model") # Include Coqui TTS
                current_vad_instance = session_data.get("vad_instance")
                
                if (
                    current_stt_model is not None
                    and current_main_mt_model is not None
                    and current_vad_instance is not None
                    and current_tts_model_instance is not None
                ):
                    print(
                        f"App: Sending personal models_loading_status to new websocket {client_info}: fully_loaded=True"
                    )
                    await manager.send_personal_message(
                        json.dumps(
                            {
                                "type": "models_loading_status",
                                "loading_started": True,
                                "fully_loaded": True,
                            }
                        ),
                        websocket,
                    )
                else:
                    print(
                        f"App: Sending personal models_loading_status to new websocket {client_info}: fully_loaded=False"
                    )
                    await manager.send_personal_message(
                        json.dumps(
                            {
                                "type": "models_loading_status",
                                "loading_started": False,
                                "fully_loaded": False,
                            }
                        ),
                        websocket,
                    )
            else:
                print(f"App: Session data not found for {client_info} after initialization attempt.")
                await manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Failed to initialize session models."
                        }
                    ),
                    websocket,
                )

            await handle_audio_stream(websocket)
        except WebSocketDisconnect:
            manager.disconnect(websocket)
            # Clean up session data on disconnect
            if client_info in backend_main.active_sessions:
                del backend_main.active_sessions[client_info]
                logging.info(f"Backend: Session {client_info} data cleared on disconnect.")
            print(
                f"App: WebSocket disconnected from {client_info}"
            )
        except Exception as e:
            print(f"App: WebSocket error for {client_info}: {e}")
            if client_info in backend_main.active_sessions:
                del backend_main.active_sessions[client_info]
                logging.info(f"Backend: Session {client_info} data cleared due to error.")
    @_app.get("/test-route")
    async def test_route():
        return {"message": "Test route successful!"}

    return _app

app = create_app()

if __name__ == "__main__":
    # Ensure piper_models and speaker_voices directories exist
    os.makedirs("backend/tts/piper_models", exist_ok=True)
    os.makedirs("speaker_voices", exist_ok=True)
    uvicorn.run(
        "app:app", # Changed to reference the app object directly
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="./certs/key.pem",
        ssl_certfile="./certs/cert.pem",
        log_level="info", # Set log level to info for cleaner output
    )
