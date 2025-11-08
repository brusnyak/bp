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
from fastapi import (
    WebSocket,
    WebSocketDisconnect,
)  # Import WebSocket and WebSocketDisconnect
import torchaudio  # Import torchaudio for resampling
import torch  # Import torch for tensor operations
import webrtcvad  # Import webrtcvad for Voice Activity Detection
from collections import deque  # Import deque for efficient buffering

from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.piper_tts import PiperTTS
import piper
import subprocess # Import subprocess to run download script
# from backend.tts.xtts_tts import XTTS_TTS # Temporarily commented out due to dependency conflicts
from backend.utils.audio_utils import load_audio, save_audio, normalize_audio

# Configuration
STT_MODEL_SIZE = "large-v3"  # Changed for FasterWhisper for better accuracy and robustness
def get_mt_model_name(source_lang: str, target_lang: str) -> str:
    """Constructs the MT model name based on source and target languages."""
    # Assuming models are named like "Helsinki-NLP/opus-mt-{source}-{target}"
    # and stored in ct2_models/Helsinki-NLP--opus-mt-{source}-{target}
    return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

# Piper TTS Model Mapping (example, extend as needed)
# Key: ISO 639-1 language code (from ui/index.html)
# Value: Piper model ID (e.g., from https://huggingface.co/rhasspy/piper-voices/tree/main)
PIPER_MODEL_MAPPING = {
    "en": "en_US-ryan-medium",
    "es": "es_ES-dave-medium",
    "fr": "fr_FR-upmc-medium",
    "de": "de_DE-thorsten-medium",
    "it": "it_IT-riccardo-medium",
    "pt": "pt_PT-caito-medium",
    "pl": "pl_PL-gosia-medium",
    "ru": "ru_RU-denis-medium",
    "nl": "nl_NL-rdh-medium",
    "cs": "cs_CZ-jirka-medium",
    "ar": "ar_JO-kareem-medium",
    "zh-cn": "zh_CN-huayan-medium",
    "ja": "ja_JP-kiritan-medium",
    "hu": "hu_HU-szabolcs-medium",
    "ko": "ko_KR-jinho-medium",
    "hi": "hi_IN-ananya-medium",
    "sk": "sk_SK-lili-medium",
    "tr": "tr_TR-emre-medium", # Added Turkish model
}

DEFAULT_TTS_MODEL = "piper"
AUDIO_SAMPLE_RATE = 16000  # Target sample rate for VAD and STT

# VAD Configuration
VAD_FRAME_DURATION = 30  # ms per frame for VAD
VAD_AGGRESSIVENESS = 2  # Adjusted to 2 for a balance between responsiveness and false positives
MIN_SPEECH_DURATION = 0.5  # seconds of speech to consider a valid segment, slightly increased
SILENCE_TIMEOUT = 0.5  # Adjusted to 0.5s to allow for natural pauses, but still responsive
STREAMING_CHUNK_LENGTH = 1.0 # seconds of audio to process for streaming STT/MT

# Global model instances
stt_model: Optional[FasterWhisperSTT] = None
mt_models: Dict[str, CTranslate2MT] = {} # Store multiple MT models
piper_tts_model: Optional[PiperTTS] = None
xtts_tts_model: Optional[Any] = None # Declare XTTS_TTS model globally (use Any as type is not imported)
vad_instance: Optional[webrtcvad.Vad] = None  # Global VAD instance

# Current configuration for translation
current_source_lang = "en"
current_target_lang = "sk"
current_tts_choice = DEFAULT_TTS_MODEL


async def initialize_models(source_lang: str, target_lang: str, tts_model_choice: str, speaker_wav_path: Optional[str] = None):
    global stt_model, mt_models, piper_tts_model, xtts_tts_model, vad_instance, current_source_lang, current_target_lang, current_tts_choice

    current_source_lang = source_lang
    current_target_lang = target_lang
    current_tts_choice = tts_model_choice

    print("Backend: Initializing models...")

    # Initialize STT
    if stt_model is None:
        print("Backend: Initializing FasterWhisperSTT...")
        stt_model = FasterWhisperSTT(model_size=STT_MODEL_SIZE, compute_type="int8")
        print("Backend: FasterWhisperSTT initialized.")

    # Initialize MT for the main pipeline
    await _initialize_mt_model(source_lang, target_lang)
    
    # Ensure MT model for en-target_lang is also initialized for phrase translation
    if source_lang != "en": # Only if the main source language is not English
        await _initialize_mt_model("en", target_lang)

    # Initialize TTS models
    if tts_model_choice == "piper":
        piper_model_id = PIPER_MODEL_MAPPING.get(target_lang, PIPER_MODEL_MAPPING["en"]) # Default to English if not found
        
        # Construct dynamic paths for Piper model
        base_model_dir = os.path.join("backend", "tts", "piper_models")
        onnx_model_path = os.path.join(base_model_dir, f"{piper_model_id}.onnx")
        json_config_path = os.path.join(base_model_dir, f"{piper_model_id}.onnx.json")

        print(f"Backend: Debugging Piper paths - ONNX: {os.path.abspath(onnx_model_path)}, JSON: {os.path.abspath(json_config_path)}")

        # Re-initialize Piper TTS if model_id changes or not yet initialized
        if piper_tts_model is None or (hasattr(piper_tts_model, 'model_id') and piper_tts_model.model_id != piper_model_id):
            if not os.path.exists(onnx_model_path) or not os.path.exists(json_config_path):
                print(f"Backend: Piper TTS model files not found for {piper_model_id} at {base_model_dir}.")
                print(f"Attempting to download Piper TTS model '{piper_model_id}'...")
                try:
                    # Call the download script
                    download_script_path = os.path.join("backend", "tts", "download_piper_models.py")
                    result = subprocess.run(
                        ["python", download_script_path, piper_model_id],
                        capture_output=True, text=True, check=True
                    )
                    print(result.stdout)
                    if result.stderr:
                        print(f"Backend: Download script stderr: {result.stderr}")
                    
                    if not os.path.exists(onnx_model_path) or not os.path.exists(json_config_path):
                        print(f"Backend: ERROR: Downloaded files for {piper_model_id} are still missing after running script.")
                        piper_tts_model = None
                    else:
                        print(f"Backend: Piper TTS model '{piper_model_id}' downloaded successfully.")
                        try:
                            print(f"Backend: Attempting to initialize PiperTTS with model_id={piper_model_id}")
                            piper_tts_model = PiperTTS(model_id=piper_model_id, speaker_id=0)
                            print(f"Backend: PiperTTS initialized with {piper_model_id}.")
                        except FileNotFoundError as e:
                            print(f"Backend: ERROR: Piper TTS model files not found for {piper_model_id} after download: {e}")
                            piper_tts_model = None
                        except RuntimeError as e:
                            print(f"Backend: ERROR: Failed to load Piper TTS model '{piper_model_id}' after download: {e}")
                            piper_tts_model = None
                        except Exception as e:
                            print(f"Backend: WARNING: An unexpected error occurred during PiperTTS initialization for {piper_model_id} after download: {e}")
                            piper_tts_model = None
                except subprocess.CalledProcessError as e:
                    print(f"Backend: ERROR: Piper TTS model download script failed for {piper_model_id}: {e.stderr}")
                    piper_tts_model = None
                except Exception as e:
                    print(f"Backend: ERROR: An unexpected error occurred during Piper TTS model download for {piper_model_id}: {e}")
                    piper_tts_model = None
            else:
                try:
                    print(f"Backend: Attempting to initialize PiperTTS with model_id={piper_model_id}")
                    piper_tts_model = PiperTTS(model_id=piper_model_id, speaker_id=0)
                    print(f"Backend: PiperTTS initialized with {piper_model_id}.")
                except FileNotFoundError as e:
                    print(f"Backend: ERROR: Piper TTS model files not found for {piper_model_id}: {e}")
                    print(f"Please ensure '{piper_model_id}.onnx' and '{piper_model_id}.onnx.json' are present in 'backend/tts/piper_models'.")
                    piper_tts_model = None
                except RuntimeError as e:
                    print(f"Backend: ERROR: Failed to load Piper TTS model '{piper_model_id}': {e}")
                    piper_tts_model = None
                except Exception as e:
                    print(f"Backend: WARNING: An unexpected error occurred during PiperTTS initialization for {piper_model_id}: {e}")
                    piper_tts_model = None
        else:
            print(f"Backend: PiperTTS for {piper_model_id} already initialized.")
        
        # Ensure XTTS is not active if Piper is chosen
        xtts_tts_model = None

    elif tts_model_choice == "xtts":
        # XTTS is temporarily disabled due to dependency conflicts.
        print("Backend: WARNING: XTTS TTS is temporarily disabled due to dependency conflicts. Please select Piper TTS.")
        xtts_tts_model = None
        piper_tts_model = None # Ensure Piper is not active if XTTS is chosen (even if disabled)
        return {"status": "error", "message": "XTTS TTS is temporarily disabled."}
        # if xtts_tts_model is None:
        #     print("Backend: Initializing XTTS_TTS...")
        #     xtts_tts_model = XTTS_TTS(device="auto") # Use "auto" for device detection
        #     await xtts_tts_model.load_model()
        #     print("Backend: XTTS_TTS initialized.")
        # else:
        #     print("Backend: XTTS_TTS already initialized.")

        # if speaker_wav_path:
        #     try:
        #         # Ensure speaker_wav_path is relative to the current working directory
        #         full_speaker_wav_path = os.path.join(os.getcwd(), speaker_wav_path)
        #         xtts_tts_model.set_speaker_wav(full_speaker_wav_path)
        #         print(f"Backend: XTTS_TTS speaker WAV set to {full_speaker_wav_path}.")
        #     except FileNotFoundError as e:
        #         print(f"Backend: WARNING: {e}. XTTS_TTS will not be able to synthesize.")
        #         xtts_tts_model = None
        #     except Exception as e:
        #         print(f"Backend: ERROR: An unexpected error occurred while setting XTTS_TTS speaker WAV: {e}")
        #         xtts_tts_model = None
        # else:
        #     print("Backend: WARNING: No speaker WAV path provided for XTTS_TTS. XTTS_TTS will not be able to synthesize.")
        #     xtts_tts_model = None
        
        # # Ensure Piper is not active if XTTS is chosen
        # piper_tts_model = None

    else:
        # Handle other TTS models or set both to None if not using Piper or XTTS
        piper_tts_model = None
        xtts_tts_model = None
        print(f"Backend: Skipping TTS initialization as '{tts_model_choice}' is selected or invalid.")

    # Initialize VAD
    global vad_instance
    if vad_instance is None:
        print(
            f"Backend: Initializing WebRTC VAD with aggressiveness={VAD_AGGRESSIVENESS}..."
        )
        vad_instance = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        print("Backend: WebRTC VAD initialized.")

    print("Backend: All models initialized (or attempted to initialize).")
    return {"status": "success", "message": "Models initialization triggered."}

async def _initialize_mt_model(source_lang: str, target_lang: str):
    """Initializes a specific MT model and stores it in the mt_models dictionary."""
    global mt_models
    model_key = f"{source_lang}-{target_lang}"
    mt_model_name = get_mt_model_name(source_lang, target_lang)
    mt_model_path = f"ct2_models/{mt_model_name.replace('/', '--')}"

    if model_key not in mt_models or mt_models[model_key].model_path != mt_model_path:
        print(f"Backend: Initializing CTranslate2MT for {model_key}...")
        if not os.path.exists(mt_model_path):
            print(f"Backend: WARNING: MT model not found locally at {mt_model_path}. Please ensure the model is converted and available.")
            print(f"You can convert it using 'python backend/mt/convert_opus_mt_to_ct2.py --model_name {mt_model_name}'")
        try:
            mt_models[model_key] = CTranslate2MT(model_path=mt_model_name, device="cpu")
            print(f"Backend: CTranslate2MT initialized for {model_key} on CPU (MPS fallback).")
        except Exception as e:
            print(f"Backend: ERROR: Failed to initialize CTranslate2MT for {model_key}: {e}")
            if model_key in mt_models:
                del mt_models[model_key] # Remove from dictionary if initialization fails
    else:
        print(f"Backend: CTranslate2MT for {model_key} already initialized.")

def get_initialized_models():
    """Returns the currently initialized model instances and current TTS choice."""
    # Return the MT model for the current main pipeline, if available
    main_mt_model = mt_models.get(f"{current_source_lang}-{current_target_lang}")
    return stt_model, main_mt_model, piper_tts_model, xtts_tts_model, vad_instance, current_tts_choice

async def get_mt_model_for_translation(source_lang: str, target_lang: str) -> Optional[CTranslate2MT]:
    """Retrieves or initializes an MT model for a specific source-target pair."""
    global mt_models
    model_key = f"{source_lang}-{target_lang}"
    if model_key not in mt_models:
        # Asynchronously initialize if not already loaded (for phrase translation)
        await _initialize_mt_model(source_lang, target_lang)
    return mt_models.get(model_key)


async def handle_audio_stream(websocket: WebSocket):
    client_info = "unknown:unknown"
    if hasattr(websocket, 'client') and websocket.client:
        client_host = getattr(websocket.client, 'host', 'unknown')
        client_port = getattr(websocket.client, 'port', 'unknown')
        client_info = f"{client_host}:{client_port}"
    print(f"Client connected from {client_info}")

    # VAD-related state
    audio_queue = deque()
    speech_frames = []
    streaming_buffer = deque() # Buffer for streaming STT/MT
    last_speech_time = time.time()
    in_speech_segment = False
    last_streaming_process_time = time.time() # Track last time a streaming chunk was processed

    # Frame size for VAD (30ms at 16kHz)
    frame_size_samples = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
    streaming_chunk_samples = int(AUDIO_SAMPLE_RATE * STREAMING_CHUNK_LENGTH)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "text" in message:
                    data = json.loads(message["text"])
                    if data["type"] == "start":
                        print("Received 'start' command. Beginning transcription.")
                        # Reset VAD state on start
                        audio_queue.clear()
                        speech_frames.clear()
                        in_speech_segment = False
                        last_speech_time = time.time()
                    elif data["type"] == "stop":
                        print("Received 'stop' command. Stopping transcription.")
                        # Process any remaining speech on stop
                        if speech_frames:
                            await process_speech_segment(
                                websocket, np.concatenate(speech_frames), last_speech_time, is_final=True
                            )
                            speech_frames.clear()
                        in_speech_segment = False
                    elif data["type"] == "config_update":
                        global current_source_lang, current_target_lang, current_tts_choice
                        new_source_lang = data.get("source_lang", current_source_lang)
                        new_target_lang = data.get("target_lang", current_target_lang)
                        new_tts_model_choice = data.get("tts_model_choice", current_tts_choice)
                        new_speaker_wav_path = data.get("speaker_wav_path", None) # Get speaker_wav_path if sent

                        # Only re-initialize if a relevant configuration has changed
                        if (new_source_lang != current_source_lang or
                            new_target_lang != current_target_lang or
                            new_tts_model_choice != current_tts_choice or
                            (new_tts_model_choice == "xtts" and new_speaker_wav_path != xtts_tts_model.speaker_wav_path if xtts_tts_model else False)): # Check if XTTS speaker path changed
                            
                            print(f"Backend: Configuration changed. Re-initializing models with: Source={new_source_lang}, Target={new_target_lang}, TTS={new_tts_model_choice}, Speaker WAV={new_speaker_wav_path}")
                            await initialize_models(new_source_lang, new_target_lang, new_tts_model_choice, new_speaker_wav_path)
                            
                            current_source_lang = new_source_lang
                            current_target_lang = new_target_lang
                            current_tts_choice = new_tts_model_choice
                            
                            await websocket.send_text(
                                json.dumps(
                                    {"type": "status", "message": "Configuration updated and models re-initialized."}
                                )
                            )
                        else:
                            print(f"Backend: Configuration updated, but no model re-initialization needed.")
                            await websocket.send_text(
                                json.dumps(
                                    {"type": "status", "message": "Configuration updated."}
                                )
                            )
                elif "bytes" in message:
                    if stt_model is None or mt_model is None or vad_instance is None:
                        print("Models not initialized. Skipping audio processing.")
                        continue

                    audio_np = np.frombuffer(message["bytes"], dtype=np.float32)

                    # Calculate RMS for mic level visualization
                    rms = np.sqrt(np.mean(audio_np**2))
                    await websocket.send_text(
                        json.dumps({"type": "audio_level", "level": float(rms)})
                    )

                    # Add incoming audio to the queue
                    audio_queue.extend(audio_np)

                    # Process audio from the queue in VAD frames
                    while len(audio_queue) >= frame_size_samples:
                        frame_np = np.array(
                            [audio_queue.popleft() for _ in range(frame_size_samples)],
                            dtype=np.float32,
                        )

                        # Convert to int16 for WebRTC VAD
                        frame_int16 = np.clip(frame_np * 32767, -32768, 32767).astype(
                            np.int16
                        )

                        is_speech = vad_instance.is_speech(
                            frame_int16.tobytes(), AUDIO_SAMPLE_RATE
                        )

                        if is_speech:
                            speech_frames.append(frame_np)
                            streaming_buffer.extend(frame_np) # Add to streaming buffer
                            in_speech_segment = True
                            last_speech_time = time.time()
                            # print("•", end="", flush=True) # Speech indicator

                            # Accumulate speech frames for final processing
                            # Removed duplicate append: speech_frames.append(frame_np)
                            
                            # Add to streaming buffer for continuous, non-final processing
                            streaming_buffer.extend(frame_np) 
                            in_speech_segment = True
                            last_speech_time = time.time()

                            # Process streaming chunk if buffer is large enough
                            if len(streaming_buffer) >= streaming_chunk_samples:
                                chunk_to_process = np.array(list(streaming_buffer), dtype=np.float32)
                                await process_speech_segment(
                                    websocket, chunk_to_process, last_speech_time, is_final=False
                                )
                                streaming_buffer.clear() # Clear the streaming buffer after processing a chunk
                                last_streaming_process_time = time.time()

                        elif in_speech_segment:
                            # If in speech segment and silence detected, check for timeout
                            if (time.time() - last_speech_time) > SILENCE_TIMEOUT:
                                # Process any remaining streaming buffer as a non-final chunk before the final segment
                                if streaming_buffer:
                                    await process_speech_segment(
                                        websocket, np.array(list(streaming_buffer), dtype=np.float32), last_speech_time, is_final=False
                                    )
                                    streaming_buffer.clear()

                                # Process the accumulated speech_frames as a final segment
                                if speech_frames:
                                    await process_speech_segment(
                                        websocket, np.concatenate(speech_frames), last_speech_time, is_final=True
                                    )
                                    speech_frames.clear()
                                in_speech_segment = False
                        else:
                            # If not in speech and not in a segment, clear streaming buffer
                            streaming_buffer.clear()
                            speech_frames.clear() # Also clear speech_frames if no speech is detected for a while

    except WebSocketDisconnect:
        client_info = "unknown:unknown"
        if hasattr(websocket, 'client') and websocket.client:
            client_host = getattr(websocket.client, 'host', 'unknown')
            client_port = getattr(websocket.client, 'port', 'unknown')
            client_info = f"{client_host}:{client_port}"
        print(
            f"Client {client_info} disconnected normally."
        )
        # Process any remaining speech on disconnect
        if speech_frames:
            await process_speech_segment(websocket, np.concatenate(speech_frames), last_speech_time, is_final=True)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        await websocket.send_text(
            json.dumps({"type": "error", "message": f"Backend error: {e}"})
        )
    finally:
        client_info = "unknown:unknown"
        if hasattr(websocket, 'client') and websocket.client:
            client_host = getattr(websocket.client, 'host', 'unknown')
            client_port = getattr(websocket.client, 'port', 'unknown')
            client_info = f"{client_host}:{client_port}"
        print(
            f"Client {client_info} handler finished."
        )


async def process_speech_segment(
    websocket: WebSocket, audio_segment_np: np.ndarray, segment_start_time: float, is_final: bool = True
):
    """Processes a detected speech segment: STT -> MT -> TTS -> Send audio."""
    if audio_segment_np.size == 0:
        return

    print(
        f"\nBackend: Processing speech segment ({audio_segment_np.size/AUDIO_SAMPLE_RATE:.2f}s, Final: {is_final})..."
    )

    stt_start_time = time.perf_counter()
    loop = asyncio.get_event_loop()
    transcribed_text_list, stt_time = await loop.run_in_executor(
        None,
        lambda: stt_model.transcribe_audio(
            audio_segment_np,
            AUDIO_SAMPLE_RATE,
            language=None if current_source_lang == "auto" else current_source_lang,
        ),
    )
    stt_end_time = time.perf_counter()
    stt_total_time = stt_end_time - stt_start_time
    
    # FasterWhisper returns a list of segments, we need to concatenate them for the full text
    transcribed_text = " ".join([s.text for s in transcribed_text_list]) if transcribed_text_list else ""

    # For streaming, we might want to send partial results more frequently
    # The `transcribed_text_list` contains segments with start/end times, which could be useful for UI.
    # For now, we'll just send the concatenated text.


    if transcribed_text:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "transcription_result",
                    "transcribed": transcribed_text,
                    "is_final": is_final, # Indicate if this is a final transcription
                    "metrics": {"stt_time": stt_total_time},
                }
            )
        )

        # Use the MT model corresponding to the current main pipeline configuration
        main_mt_model = mt_models.get(f"{current_source_lang}-{current_target_lang}")
        if main_mt_model is None:
            print(f"WARNING: MT model for {current_source_lang}-{current_target_lang} not initialized. Skipping translation.")
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "message": f"MT model for {current_source_lang}-{current_target_lang} not initialized. Skipping translation."}
                )
            )
            return # Exit if no MT model is available

        mt_start_time = time.perf_counter()
        translated_text, _ = await loop.run_in_executor(
            None,
            lambda: main_mt_model.translate(
                transcribed_text, current_source_lang, current_target_lang
            ),
        )
        mt_end_time = time.perf_counter()
        mt_total_time = mt_end_time - mt_start_time

        await websocket.send_text(
            json.dumps(
                {
                    "type": "translation_result",
                    "translated": translated_text,
                    "is_final": is_final, # Indicate if this is a final translation
                    "metrics": {"mt_time": mt_total_time},
                }
            )
        )

        tts_start_time = time.perf_counter()
        audio_waveform, sample_rate, tts_time = None, None, 0.0

        if current_tts_choice == "piper" and piper_tts_model:
            if piper_tts_model:
                audio_waveform, sample_rate, tts_time = await loop.run_in_executor(
                    None,
                    lambda: piper_tts_model.synthesize(
                        translated_text, current_target_lang
                    ),
                )
                if audio_waveform is not None and sample_rate is not None:
                    audio_buffer = io.BytesIO()
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda: sf.write(
                                audio_buffer, audio_waveform, sample_rate, format="WAV"
                            ),
                        )
                        audio_bytes_to_send = audio_buffer.getvalue()
                        await websocket.send_bytes(audio_bytes_to_send)
                    except Exception as e:
                        print(f"ERROR: Failed to write audio to buffer for Piper TTS: {e}")
                        await websocket.send_text(
                            json.dumps(
                                {"type": "error", "message": f"Backend audio processing error (Piper): {e}"}
                            )
                        )
                        return
            else:
                print(f"WARNING: Piper TTS model not initialized. No TTS will be performed for '{translated_text}'.")
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "message": "Piper TTS model not initialized. No TTS output."}
                    )
                )
                return
        elif current_tts_choice == "xtts": # XTTS is now handled as a disabled option in initialize_models
            print(f"WARNING: XTTS TTS is disabled. No TTS will be performed for '{translated_text}'.")
            await websocket.send_text(
                json.dumps(
                    {"type": "error", "message": "XTTS TTS is temporarily disabled. No TTS output."}
                )
            )
            return
        else:
            print(
                f"WARNING: Selected TTS model '{current_tts_choice}' not initialized or invalid. No TTS will be performed."
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "status",
                        "message": f"TTS model '{current_tts_choice}' not ready or invalid. No TTS output.",
                    }
                )
            )
            return # Exit if no TTS can be performed

        tts_end_time = time.perf_counter()
        tts_total_time = tts_end_time - tts_start_time

        # Send metrics for both final and non-final segments
        metrics_payload = {
            "stt_time": stt_total_time,
            "mt_time": mt_total_time,
            "tts_time": tts_total_time,
        }
        if is_final:
            total_latency = stt_total_time + mt_total_time + tts_total_time
            metrics_payload["total_latency"] = total_latency

        await websocket.send_text(
            json.dumps(
                {
                    "type": "final_metrics",
                    "metrics": metrics_payload,
                    "is_final": is_final, # Add is_final to metrics for frontend to distinguish
                }
            )
        )
