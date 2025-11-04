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
from backend.utils.audio_utils import load_audio, save_audio, normalize_audio

# Configuration
STT_MODEL_SIZE = "tiny"  # Changed for FasterWhisper
MT_MODEL_NAME = (
    "Helsinki-NLP/opus-mt-en-sk"  # This is a CTranslate2 compatible model name
)
PIPER_MODEL_PATH = "backend/tts/piper_models/cs_CZ-jirka-medium.onnx"
PIPER_CONFIG_PATH = "backend/tts/piper_models/cs_CZ-jirka-medium.onnx.json"
DEFAULT_TTS_MODEL = "piper"
AUDIO_SAMPLE_RATE = 16000  # Target sample rate for VAD and STT

# VAD Configuration
VAD_FRAME_DURATION = 30  # ms per frame for VAD
VAD_AGGRESSIVENESS = 3  # 0 (least aggressive) to 3 (most aggressive) - Increased for more responsiveness
MIN_SPEECH_DURATION = 0.1  # seconds of speech to consider a valid segment (further reduced for responsiveness)
SILENCE_TIMEOUT = 0.2  # seconds of silence to mark end of speech (further reduced for responsiveness)

# Global model instances
stt_model: Optional[FasterWhisperSTT] = None
mt_model: Optional[CTranslate2MT] = None
piper_tts_model: Optional[PiperTTS] = None
vad_instance: Optional[webrtcvad.Vad] = None  # Global VAD instance

# Current configuration for translation
current_source_lang = "en"
current_target_lang = "sk"
current_tts_choice = DEFAULT_TTS_MODEL


async def initialize_models(source_lang: str, target_lang: str, tts_model_choice: str):
    global stt_model, mt_model, piper_tts_model
    global current_source_lang, current_target_lang, current_tts_choice

    current_source_lang = source_lang
    current_target_lang = target_lang
    current_tts_choice = tts_model_choice

    print("Backend: Initializing models...")

    # Initialize STT
    if stt_model is None:
        print("Backend: Initializing FasterWhisperSTT...")
        stt_model = FasterWhisperSTT(model_size=STT_MODEL_SIZE, compute_type="int8")
        print("Backend: FasterWhisperSTT initialized.")

    # Initialize MT
    if mt_model is None:
        print("Backend: Initializing CTranslate2MT...")
        mt_model = CTranslate2MT(model_path=MT_MODEL_NAME, device="auto")
        print("Backend: CTranslate2MT initialized.")

    # Initialize TTS models
    if piper_tts_model is None:
        piper_model_id = "cs_CZ-jirka-medium"

        if not os.path.exists(PIPER_MODEL_PATH):
            print(
                f"Backend: WARNING: Piper TTS model file not found at {PIPER_MODEL_PATH}. Skipping PiperTTS initialization."
            )
            piper_tts_model = None
        elif not os.path.exists(PIPER_CONFIG_PATH):
            print(
                f"Backend: WARNING: Piper TTS config file not found at {PIPER_CONFIG_PATH}. Skipping PiperTTS initialization."
            )
            piper_tts_model = None
        else:
            try:
                print(
                    f"Backend: Attempting to initialize PiperTTS with model_id={piper_model_id}"
                )
                piper_tts_model = PiperTTS(model_id=piper_model_id, speaker_id=0)
                print("Backend: PiperTTS initialized.")
            except Exception as e:
                print(
                    f"Backend: WARNING: An unexpected error occurred during PiperTTS initialization: {e}"
                )
                piper_tts_model = None

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


def get_initialized_models():
    """Returns the currently initialized model instances."""
    return stt_model, mt_model, piper_tts_model, vad_instance


async def handle_audio_stream(websocket: WebSocket):
    print(f"Client connected from {websocket.client.host}:{websocket.client.port}")

    # VAD-related state
    audio_queue = deque()
    speech_frames = []
    last_speech_time = time.time()
    in_speech_segment = False

    # Frame size for VAD (30ms at 16kHz)
    frame_size_samples = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)

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
                                websocket, speech_frames, last_speech_time
                            )
                            speech_frames.clear()
                        in_speech_segment = False
                    elif data["type"] == "config_update":
                        global current_source_lang, current_target_lang, current_tts_choice
                        current_source_lang = data.get(
                            "source_lang", current_source_lang
                        )
                        current_target_lang = data.get(
                            "target_lang", current_target_lang
                        )
                        current_tts_choice = data.get(
                            "tts_model_choice", current_tts_choice
                        )
                        print(
                            f"Configuration updated: Source={current_source_lang}, Target={current_target_lang}, TTS={current_tts_choice}"
                        )
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
                            in_speech_segment = True
                            last_speech_time = time.time()
                            # print("•", end="", flush=True) # Speech indicator
                        elif in_speech_segment:
                            # print(".", end="", flush=True) # Silence indicator
                            # If in speech segment and silence detected, check for timeout
                            if (time.time() - last_speech_time) > SILENCE_TIMEOUT:
                                await process_speech_segment(
                                    websocket, speech_frames, last_speech_time
                                )
                                speech_frames.clear()
                                in_speech_segment = False
                        # If not in speech segment and silence, just continue

    except WebSocketDisconnect:
        print(
            f"Client {websocket.client.host}:{websocket.client.port} disconnected normally."
        )
        # Process any remaining speech on disconnect
        if speech_frames:
            await process_speech_segment(websocket, speech_frames, last_speech_time)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        await websocket.send_text(
            json.dumps({"type": "error", "message": f"Backend error: {e}"})
        )
    finally:
        print(
            f"Client {websocket.client.host}:{websocket.client.port} handler finished."
        )


async def process_speech_segment(
    websocket: WebSocket, speech_frames: list, segment_start_time: float
):
    """Processes a detected speech segment: STT -> MT -> TTS -> Send audio."""
    if not speech_frames:
        return

    audio_segment_np = np.concatenate(speech_frames)

    # Process segment regardless of length to ensure real-time responsiveness
    # The SILENCE_TIMEOUT will define the segment boundaries.
    print(
        f"\nBackend: Processing speech segment ({len(audio_segment_np)/AUDIO_SAMPLE_RATE:.2f}s)..."
    )

    stt_start_time = time.perf_counter()
    loop = asyncio.get_event_loop()
    transcribed_text, stt_time = await loop.run_in_executor(
        None,
        lambda: stt_model.transcribe_audio(
            audio_segment_np, AUDIO_SAMPLE_RATE, language=current_source_lang
        ),
    )
    stt_end_time = time.perf_counter()
    stt_total_time = stt_end_time - stt_start_time

    if transcribed_text:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "transcription_result",
                    "transcribed": transcribed_text,
                    "metrics": {"stt_time": stt_total_time},
                }
            )
        )

        mt_start_time = time.perf_counter()
        translated_text, _ = await loop.run_in_executor(
            None,
            lambda: mt_model.translate(
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
                    "metrics": {"mt_time": mt_total_time},
                }
            )
        )

        tts_start_time = time.perf_counter()
        audio_waveform, sample_rate, tts_time = None, None, 0.0

        if current_tts_choice == "piper" and piper_tts_model:
            audio_waveform, sample_rate, tts_time = await loop.run_in_executor(
                None,
                lambda: piper_tts_model.synthesize(
                    translated_text, current_target_lang
                ),
            )
        else:
            print(
                f"WARNING: Selected TTS model '{current_tts_choice}' not initialized or invalid. Falling back to Piper TTS."
            )
            if piper_tts_model:
                audio_waveform, sample_rate, tts_time = await loop.run_in_executor(
                    None,
                    lambda: piper_tts_model.synthesize(
                        translated_text, current_target_lang
                    ),
                )
            else:
                print(f"ERROR: Piper TTS model not initialized. Cannot fallback.")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "status",
                            "message": f"TTS model '{current_tts_choice}' not ready and Piper TTS fallback failed.",
                        }
                    )
                )

        if audio_waveform is not None and sample_rate is not None:
            audio_buffer = io.BytesIO()
            try:
                # sf.write is blocking, run in executor
                await loop.run_in_executor(
                    None,
                    lambda: sf.write(
                        audio_buffer, audio_waveform, sample_rate, format="WAV"
                    ),
                )
                audio_bytes_to_send = audio_buffer.getvalue()
            except Exception as e:
                print(f"ERROR: Failed to write audio to buffer: {e}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Backend audio processing error: {e}",
                        }
                    )
                )
                return

            await websocket.send_bytes(audio_bytes_to_send)

            tts_end_time = time.perf_counter()
            tts_total_time = tts_end_time - tts_start_time

            total_latency = stt_total_time + mt_total_time + tts_total_time
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "final_metrics",
                        "metrics": {
                            "stt_time": stt_total_time,
                            "mt_time": mt_total_time,
                            "tts_time": tts_total_time,
                            "total_latency": total_latency,
                        },
                    }
                )
            )
        else:
            print("No audio waveform generated by TTS.")
