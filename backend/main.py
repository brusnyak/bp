import asyncio
import websockets
import json
import numpy as np
import base64
import io
import soundfile as sf
import time
from typing import Dict, Any, Optional
import os

# from chatterbox.mtl_tts import ChatterboxMultilingualTTS # Temporarily commented out
from backend.stt.mlx_whisper_stt import MLXWhisperSTT
from backend.mt.marian_mt import MarianMT # Using MarianMT as primary MT model
# from backend.mt.nllb_mt import NLLB_MT # Commented out NLLB
# from backend.mt.seamless_m4t_mt import SeamlessM4Tv2MT # Commented out SeamlessM4T v2
# from backend.tts.chatterbox_tts import ChatterboxTTS # Temporarily commented out
from backend.tts.piper_tts import PiperTTS
import piper # Corrected import for piper_tts package
from backend.utils.audio_utils import load_audio, save_audio, normalize_audio

# Configuration
STT_MODEL_SIZE = "mlx-community/whisper-tiny" # Updated for mlx-whisper
MT_MODEL_NAME = "Helsinki-NLP/opus-mt-en-sk" # Updated for MarianMT
PIPER_MODEL_PATH = "backend/tts/piper_models/cs_CZ-jirka-medium.onnx"
PIPER_CONFIG_PATH = "backend/tts/piper_models/cs_CZ-jirka-medium.onnx.json"
DEFAULT_TTS_MODEL = "chatterbox" # or "piper"
AUDIO_SAMPLE_RATE = 16000 # Standard sample rate for STT input

# Global model instances
stt_model: Optional[MLXWhisperSTT] = None
mt_model: Optional[MarianMT] = None # Changed to MarianMT
# chatterbox_tts_model: Optional[ChatterboxTTS] = None # Temporarily commented out
piper_tts_model: Optional[PiperTTS] = None
# The ChatterboxMultilingualTTS class is used directly, no need for MTLTTS alias.
# chatterbox_tts_model is an instance of ChatterboxTTS, which internally uses ChatterboxMultilingualTTS.
speaker_embeddings: Dict[str, Any] = {} # Stores speaker embeddings for voice cloning

# Current configuration for translation
current_source_lang = "en"
current_target_lang = "sk"
current_tts_choice = DEFAULT_TTS_MODEL

async def initialize_models(source_lang: str, target_lang: str, tts_model_choice: str):
    global stt_model, mt_model, chatterbox_tts_model, piper_tts_model
    global current_source_lang, current_target_lang, current_tts_choice

    current_source_lang = source_lang
    current_target_lang = target_lang
    current_tts_choice = tts_model_choice

    print("Initializing models...")
    
    # Initialize STT
    if stt_model is None:
        stt_model = MLXWhisperSTT(model_size=STT_MODEL_SIZE, compute_type="int8")
    
    # Initialize MT
    if mt_model is None:
        mt_model = MarianMT(model_name=MT_MODEL_NAME, device="auto") # Changed to MarianMT
    
    # Initialize TTS models
    # if chatterbox_tts_model is None: # Temporarily commented out
    #     chatterbox_tts_model = ChatterboxTTS(device="auto") # Temporarily commented out
    
    if piper_tts_model is None:
        if os.path.exists(PIPER_MODEL_PATH) and os.path.exists(PIPER_CONFIG_PATH):
            print(f"Attempting to initialize PiperTTS with model_path={PIPER_MODEL_PATH}")
            try:
                piper_tts_model = PiperTTS(model_path=PIPER_MODEL_PATH, speaker_id=0)
            except Exception as e:
                print(f"WARNING: Could not initialize PiperTTS. Error: {e}")
                piper_tts_model = None # Mark as not initialized if it fails
        else:
            print(f"WARNING: Piper TTS model files not found at {PIPER_MODEL_PATH} and {PIPER_CONFIG_PATH}. PiperTTS will not be initialized.")
            print("Please download the Piper model and config files (e.g., cs_CZ-jirka-medium.onnx and cs_CZ-jirka-medium.onnx.json) and place them in the project root.")
            piper_tts_model = None # Mark as not initialized if files are missing

    print("All models initialized (or attempted to initialize).")
    return {"status": "success", "message": "Models initialization triggered."}

async def handle_audio_stream(websocket, path):
    print(f"Client connected from {websocket.remote_address}")
    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)
                if data["type"] == "start":
                    print("Received 'start' command. Beginning transcription.")
                    # Reset any ongoing transcription state if necessary
                elif data["type"] == "stop":
                    print("Received 'stop' command. Stopping transcription.")
                    # Finalize transcription if necessary
                elif data["type"] == "config_update":
                    global current_source_lang, current_target_lang, current_tts_choice
                    current_source_lang = data.get("source_lang", current_source_lang)
                    current_target_lang = data.get("target_lang", current_target_lang)
                    current_tts_choice = data.get("tts_model_choice", current_tts_choice)
                    print(f"Configuration updated: Source={current_source_lang}, Target={current_target_lang}, TTS={current_tts_choice}")
                    await websocket.send(json.dumps({"type": "status", "message": "Configuration updated."}))
                elif data["type"] == "request_phonetic_prompt":
                    # This is a placeholder. Chatterbox doesn't directly provide phonetic prompts.
                    # In a real scenario, you'd use a separate library or a fixed set of prompts.
                    prompt_lang = data.get("language", "en")
                    # For now, provide a generic prompt.
                    if prompt_lang == "en":
                        prompt_text = "The quick brown fox jumps over the lazy dog."
                    elif prompt_lang == "sk":
                        prompt_text = "Rýchla hnedá líška preskočí lenivého psa."
                    elif prompt_lang == "cs":
                        prompt_text = "Rychlá hnědá liška přeskočí líného psa."
                    else:
                        prompt_text = "Please speak clearly into the microphone."
                    await websocket.send(json.dumps({"type": "phonetic_prompt_result", "prompt_text": prompt_text}))
                    print(f"Sent phonetic prompt for language: {prompt_lang}")
                elif data["type"] == "upload_reference_audio":
                    audio_data_base64 = data["audio_data_base64"]
                    mime_type = data["mime_type"]
                    provided_transcription = data.get("provided_transcription")

                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_data_base64)
                    
                    # Save to a temporary file
                    temp_audio_path = "temp_reference_audio.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    print(f"Received reference audio for voice training. Size: {len(audio_bytes)} bytes")
                    
                    if chatterbox_tts_model:
                        try:
                            # Load and resample audio to 16kHz for Chatterbox's get_speaker_embedding
                            audio_np, sr = load_audio(temp_audio_path, target_sr=16000)
                            
                            # Chatterbox's get_speaker_embedding expects a file path or (audio, sr)
                            # Let's assume it can take the path directly after saving.
                            # If it needs transcription, we pass the provided one.
                            
                            # The `get_speaker_embedding` method might also take `audio_np` and `sr` directly.
                            # For now, we'll pass the path.
                            
                            speaker_info = chatterbox_tts_model.train_voice(temp_audio_path, transcription=provided_transcription)
                            speaker_embeddings["default_speaker"] = speaker_info["speaker_embedding"]
                            print("Voice training successful. Speaker embedding stored.")
                            await websocket.send(json.dumps({"type": "status", "message": "Reference audio updated and voice trained."}))
                        except Exception as e:
                            print(f"Error during voice training: {e}")
                            await websocket.send(json.dumps({"type": "status", "message": f"Error processing reference audio: {e}"}))
                        finally:
                            if os.path.exists(temp_audio_path):
                                os.remove(temp_audio_path)
                    else:
                        await websocket.send(json.dumps({"type": "status", "message": "Chatterbox TTS model not initialized for voice training."}))
                elif data["type"] == "start_modal_recording":
                    print(f"Received 'start_modal_recording' command for language: {data.get('language')}")
                    # Here you might want to start a separate STT process for the modal
                    # to provide real-time transcription feedback for the prompt.
                    # For now, we'll just acknowledge.
                    await websocket.send(json.dumps({"type": "status", "message": "Modal recording started."}))
                elif data["type"] == "stop_modal_recording":
                    print("Received 'stop_modal_recording' command.")
                    # Stop the modal STT process.
                    await websocket.send(json.dumps({"type": "status", "message": "Modal recording stopped."}))

            elif isinstance(message, bytes):
                # Process raw audio bytes from the frontend
                if stt_model is None or mt_model is None:
                    print("Models not initialized. Skipping audio processing.")
                    continue

                # Convert bytes to numpy array (float32)
                audio_np = np.frombuffer(message, dtype=np.float32)
                
                # Transcribe
                stt_start_time = time.perf_counter()
                # Re-introducing language parameter
                transcribed_text, stt_time = stt_model.transcribe_audio(audio_np, AUDIO_SAMPLE_RATE, language=current_source_lang)
                stt_end_time = time.perf_counter()
                stt_total_time = stt_end_time - stt_start_time

                if transcribed_text:
                    await websocket.send(json.dumps({
                        "type": "transcription_result",
                        "transcribed": transcribed_text,
                        "metrics": {"stt_time": stt_total_time}
                    }))

                    # Translate
                    mt_start_time = time.perf_counter()
                    translated_text, _ = mt_model.translate_text(transcribed_text, current_source_lang, current_target_lang)
                    mt_end_time = time.perf_counter()
                    mt_total_time = mt_end_time - mt_start_time

                    await websocket.send(json.dumps({
                        "type": "translation_result",
                        "translated": translated_text,
                        "metrics": {"mt_time": mt_total_time}
                    }))

                    # Synthesize speech
                    tts_start_time = time.perf_counter()
                    audio_waveform, sample_rate, tts_time = None, None, 0.0

                    if current_tts_choice == "chatterbox": # and chatterbox_tts_model: # Temporarily commented out
                        print(f"WARNING: Chatterbox TTS is temporarily disabled due to dependency conflicts. Using Piper TTS.")
                        # Fallback to Piper if Chatterbox is selected but disabled
                        if piper_tts_model:
                            audio_waveform, sample_rate, tts_time = piper_tts_model.synthesize_speech(
                                translated_text, current_target_lang
                            )
                        else:
                            print(f"ERROR: Piper TTS model not initialized. Cannot fallback.")
                            await websocket.send(json.dumps({"type": "status", "message": f"Neither Chatterbox nor Piper TTS models are ready."}))
                    elif current_tts_choice == "piper" and piper_tts_model:
                        # Piper doesn't support voice cloning in the same way as Chatterbox
                        # Use default Piper voice for the target language (or proxy)
                        # Piper's language handling is model-specific, so we pass the target_lang for context
                        audio_waveform, sample_rate, tts_time = piper_tts_model.synthesize_speech(
                            translated_text, current_target_lang
                        )
                    else:
                        print(f"WARNING: Selected TTS model '{current_tts_choice}' not initialized or invalid.")
                        await websocket.send(json.dumps({"type": "status", "message": f"TTS model '{current_tts_choice}' not ready."}))

                    if audio_waveform is not None and sample_rate is not None:
                        # Convert numpy array to bytes for WebSocket
                        # Ensure audio is float32 and normalized for soundfile
                        audio_waveform = normalize_audio(audio_waveform)
                        
                        # Use BytesIO to avoid saving to disk
                        audio_buffer = io.BytesIO()
                        sf.write(audio_buffer, audio_waveform, sample_rate, format='WAV')
                        audio_bytes_to_send = audio_buffer.getvalue()
                        
                        await websocket.send(audio_bytes_to_send)
                        
                        tts_end_time = time.perf_counter()
                        tts_total_time = tts_end_time - tts_start_time

                        total_latency = stt_total_time + mt_total_time + tts_total_time
                        await websocket.send(json.dumps({
                            "type": "final_metrics",
                            "metrics": {
                                "stt_time": stt_total_time,
                                "mt_time": mt_total_time,
                                "tts_time": tts_total_time,
                                "total_latency": total_latency
                            }
                        }))
                    else:
                        print("No audio waveform generated by TTS.")

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Client {websocket.remote_address} disconnected normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Client {websocket.remote_address} disconnected with error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print(f"Client {websocket.remote_address} handler finished.")

# The main WebSocket server logic is now handled by app.py using FastAPI.
# This file (backend/main.py) now serves as a module for model initialization and audio stream handling.

# The `if __name__ == "__main__":` block is removed as this file is intended to be imported.
# Any direct testing should be done via `app.py` or a dedicated test script.
