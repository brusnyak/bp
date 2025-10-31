import asyncio
import json
import os
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, Any

from backend.main import initialize_models, handle_audio_stream, current_source_lang, current_target_lang, current_tts_choice

app = FastAPI()

# Mount static files for the frontend
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Templates for serving HTML (if needed, though index.html is static)
templates = Jinja2Templates(directory="ui")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main frontend HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/initialize")
async def initialize_backend(source_lang: str = current_source_lang, 
                             target_lang: str = current_target_lang, 
                             tts_model_choice: str = current_tts_choice):
    """
    Initializes the STT, MT, and TTS models.
    This endpoint is called by the frontend to prepare the backend.
    """
    print(f"Received initialization request: source_lang={source_lang}, target_lang={target_lang}, tts_model_choice={tts_model_choice}")
    result = await initialize_models(source_lang, target_lang, tts_model_choice)
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles WebSocket connections for real-time audio streaming and translation.
    """
    await websocket.accept()
    print(f"WebSocket accepted connection from {websocket.client.host}:{websocket.client.port}")
    try:
        await handle_audio_stream(websocket, None) # Pass None for path as it's not used in handle_audio_stream
    except WebSocketDisconnect:
        print(f"WebSocket disconnected from {websocket.client.host}:{websocket.client.port}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print(f"WebSocket connection from {websocket.client.host}:{websocket.client.port} closed.")

if __name__ == "__main__":
    # Ensure Piper model files exist for local testing if Piper is chosen
    # In a production setup, these would be pre-downloaded or managed.
    PIPER_MODEL_PATH = "cs_CZ-fairseq-medium.onnx"
    PIPER_CONFIG_PATH = "cs_CZ-fairseq-medium.json"

    if not os.path.exists(PIPER_MODEL_PATH) or not os.path.exists(PIPER_CONFIG_PATH):
        print(f"WARNING: Piper model files not found in current directory.")
        print(f"Please download '{PIPER_MODEL_PATH}' and '{PIPER_CONFIG_PATH}' for PiperTTS to function.")
        print(f"Example Czech model (proxy for Slovak):")
        print(f"  Model: https://huggingface.co/rhasspy/piper-voices/resolve/main/cs/cs_CZ/vits/fairseq/medium/cs_CZ-fairseq-medium.onnx")
        print(f"  Config: https://huggingface.co/rhasspy/piper-voices/resolve/main/cs/cs_CZ/vits/fairseq/medium/cs_CZ-fairseq-medium.json")
        print(f"Using mock PiperTTS for now. Real PiperTTS will fail without these files.")

    # You can run the FastAPI app using: uvicorn app:app --reload --port 8000
    # Or directly from here for convenience during development:
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
