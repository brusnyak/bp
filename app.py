import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
import os

from backend.main import initialize_models, handle_audio_stream, get_initialized_models
from typing import List


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
        # Create a copy of the list to iterate over, in case connections are removed during iteration
        inactive_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                # Mark connection for removal if it's disconnected
                inactive_connections.append(connection)
            except RuntimeError as e:
                # Catch other potential runtime errors during send, e.g., connection already closed
                print(f"Error broadcasting to a connection: {e}")
                inactive_connections.append(connection)
        
        # Remove inactive connections after iteration
        for connection in inactive_connections:
            self.active_connections.remove(connection)


manager = ConnectionManager()

app = FastAPI()

# Mount static files for the UI
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Templates for serving HTML
templates = Jinja2Templates(directory="ui")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/initialize")
async def initialize_pipeline(
    source_lang: str, target_lang: str, tts_model_choice: str
):
    print(
        f"App: Received initialization request: source_lang={source_lang}, target_lang={target_lang}, tts_model_choice={tts_model_choice}"
    )
    # This will trigger the model loading in the background
    result = await initialize_models(source_lang, target_lang, tts_model_choice)

    # After models are initialized, broadcast a status update to all connected websockets
    # This ensures that any client that connects after initialization, or is already connected,
    # receives the correct status.
    (
        current_stt_model,
        current_mt_model,
        current_piper_tts_model,
        current_vad_instance,
    ) = get_initialized_models()
    if (
        current_stt_model is not None
        and current_mt_model is not None
        and current_piper_tts_model is not None
        and current_vad_instance is not None
    ):
        print("App: Broadcasting models_loading_status: fully_loaded=True")
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
            "App: Broadcasting models_loading_status: fully_loaded=False (error or incomplete initialization)"
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

    return result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        print(
            f"App: WebSocket accepted from {websocket.client.host}:{websocket.client.port}"
        )
        # When a new websocket connects, send the current model loading status
        (
            current_stt_model,
            current_mt_model,
            current_piper_tts_model,
            current_vad_instance,
        ) = get_initialized_models()
        if (
            current_stt_model is not None
            and current_mt_model is not None
            and current_piper_tts_model is not None
            and current_vad_instance is not None
        ):
            print(
                "App: Sending personal models_loading_status to new websocket: fully_loaded=True"
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
                "App: Sending personal models_loading_status to new websocket: fully_loaded=False"
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
        await handle_audio_stream(websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(
            f"App: WebSocket disconnected from {websocket.client.host}:{websocket.client.port}"
        )
    except Exception as e:
        print(f"App: WebSocket error: {e}")


if __name__ == "__main__":
    # Ensure piper_models directory exists for PiperTTS
    os.makedirs("backend/tts/piper_models", exist_ok=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="./certs/key.pem",
        ssl_certfile="./certs/cert.pem",
    )
