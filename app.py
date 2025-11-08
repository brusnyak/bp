import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
import os
from typing import Optional, List # Ensure List is imported

from backend.main import initialize_models, handle_audio_stream, get_initialized_models, get_mt_model_for_translation # Import get_mt_model_for_translation
from fastapi import UploadFile, File
# from backend.tts.xtts_tts import XTTS_TTS # Temporarily commented out due to dependency conflicts


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


@app.get("/list_speaker_voices")
async def list_speaker_voices():
    try:
        voices_dir = "speaker_voices"
        if not os.path.exists(voices_dir):
            return {"status": "success", "voices": []}
        
        voice_files = [f for f in os.listdir(voices_dir) if f.endswith(".wav")]
        return {"status": "success", "voices": voice_files}
    except Exception as e:
        print(f"Error listing speaker voices: {e}")
        return {"status": "error", "message": f"Failed to list voices: {e}"}


@app.post("/upload_speaker_voice")
async def upload_speaker_voice(file: UploadFile = File(...)):
    try:
        file_location = f"speaker_voices/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        print(f"Speaker voice file saved to {file_location}")
        return {"status": "success", "message": f"File '{file.filename}' uploaded successfully."}
    except Exception as e:
        print(f"Error uploading speaker voice file: {e}")
        return {"status": "error", "message": f"Failed to upload file: {e}"}


@app.post("/rename_speaker_voice")
async def rename_speaker_voice(request: Request):
    try:
        data = await request.json()
        old_name = data.get("old_name")
        new_name = data.get("new_name")

        if not old_name or not new_name:
            return {"status": "error", "message": "Old name and new name are required."}

        old_path = os.path.join("speaker_voices", old_name)
        new_path = os.path.join("speaker_voices", new_name)

        if not os.path.exists(old_path):
            return {"status": "error", "message": f"File '{old_name}' not found."}
        
        if os.path.exists(new_path):
            return {"status": "error", "message": f"File with name '{new_name}' already exists."}

        os.rename(old_path, new_path)
        print(f"Speaker voice file renamed from {old_name} to {new_name}")
        return {"status": "success", "message": f"File '{old_name}' renamed to '{new_name}' successfully."}
    except Exception as e:
        print(f"Error renaming speaker voice file: {e}")
        return {"status": "error", "message": f"Failed to rename file: {e}"}


@app.post("/translate_phrase")
async def translate_phrase(request: Request):
    try:
        data = await request.json()
        phrase = data.get("phrase")
        target_lang = data.get("target_lang")

        if not phrase or not target_lang:
            return {"status": "error", "message": "Phrase and target_lang are required."}

        # Ensure MT model is initialized
        (
            current_stt_model,
            current_mt_model,
            current_piper_tts_model,
            current_xtts_tts_model,
            current_vad_instance,
            current_tts_choice_from_main,
        ) = get_initialized_models()

        if not phrase or not target_lang:
            return {"status": "error", "message": "Phrase and target_lang are required."}

        # Get the specific MT model for en-target_lang
        mt_model_for_phrase = await get_mt_model_for_translation("en", target_lang) # Await the async function

        if mt_model_for_phrase is None:
            return {"status": "error", "message": f"MT model for en-{target_lang} not initialized."}

        translated_text, _ = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: mt_model_for_phrase.translate(phrase, "en", target_lang),
        )
        return {"status": "success", "translated_text": translated_text}
    except Exception as e:
        print(f"Error translating phrase: {e}")
        return {"status": "error", "message": f"Failed to translate phrase: {e}"}


@app.post("/delete_speaker_voice")
async def delete_speaker_voice(request: Request):
    try:
        data = await request.json()
        filename = data.get("filename")

        if not filename:
            return {"status": "error", "message": "Filename is required."}

        file_path = os.path.join("speaker_voices", filename)

        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File '{filename}' not found."}
        
        os.remove(file_path)
        print(f"Speaker voice file '{filename}' deleted successfully.")
        return {"status": "success", "message": f"File '{filename}' deleted successfully."}
    except Exception as e:
        print(f"Error deleting speaker voice file: {e}")
        return {"status": "error", "message": f"Failed to delete file: {e}"}


@app.post("/initialize")
async def initialize_pipeline(
    source_lang: str, target_lang: str, tts_model_choice: str, speaker_wav_path: Optional[str] = None
):
    print(
        f"App: Received initialization request: source_lang={source_lang}, target_lang={target_lang}, tts_model_choice={tts_model_choice}"
    )
    # This will trigger the model loading in the background
    result = await initialize_models(source_lang, target_lang, tts_model_choice, speaker_wav_path)

    # After models are initialized, broadcast a status update to all connected websockets
    # This ensures that any client that connects after initialization, or is already connected,
    # receives the correct status.
    (
        current_stt_model,
        current_main_mt_model, # Renamed to current_main_mt_model
        current_piper_tts_model,
        current_xtts_tts_model,
        current_vad_instance,
        current_tts_choice_from_main, # Get current_tts_choice from main
    ) = get_initialized_models()
    if (
        current_stt_model is not None
        and current_main_mt_model is not None # Use current_main_mt_model
        and current_piper_tts_model is not None
        and (current_tts_choice_from_main != "xtts" or current_xtts_tts_model is not None) # Check xtts_tts_model only if XTTS is chosen
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
            current_main_mt_model, # Renamed to current_main_mt_model
            current_piper_tts_model,
            current_xtts_tts_model,
            current_vad_instance,
            current_tts_choice_from_main, # Get current_tts_choice from main
        ) = get_initialized_models()
        
        # Use current_tts_choice_from_main for the check
        if (
            current_stt_model is not None
            and current_main_mt_model is not None # Use current_main_mt_model
            and current_piper_tts_model is not None
            and (current_tts_choice_from_main != "xtts" or current_xtts_tts_model is not None) # Check xtts_tts_model only if XTTS is chosen
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
    # Ensure piper_models and speaker_voices directories exist
    os.makedirs("backend/tts/piper_models", exist_ok=True)
    os.makedirs("speaker_voices", exist_ok=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="./certs/key.pem",
        ssl_certfile="./certs/cert.pem",
    )
