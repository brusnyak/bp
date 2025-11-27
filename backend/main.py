import logging
import sys
import asyncio
import threading
import websockets
import json
import numpy as np
import base64
import io
import soundfile as sf
import time
import os
from typing import Dict, Any, Optional, Tuple, List
from fastapi import FastAPI, Depends # Import Depends
from pydantic import BaseModel, Field # Import Field
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm # Import security components
from jose import jwt, JWTError # Import JWT components (even if using mock for now)
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# Initialize the database for the main application
from backend.utils.db_manager import SQLALCHEMY_DATABASE_URL, get_db_session_and_engine, init_db, User # Import init_db and User
from backend.utils.auth import get_password_hash, verify_password # Import auth functions

# Get engine and SessionLocal
engine, SessionLocal = get_db_session_and_engine(SQLALCHEMY_DATABASE_URL)
# init_db(engine) # Initialize database tables - Removed to prevent premature table creation outside of app lifespan or test setup

# Pydantic Models for Request Bodies
class RenameVoiceRequest(BaseModel):
    old_name: str
    new_name: str

class TranslatePhraseRequest(BaseModel):
    phrase: str
    target_lang: str

class DeleteVoiceRequest(BaseModel):
    filename: str

# New Pydantic Models for User Authentication
class UserCreate(BaseModel):
    username: str = Field(..., min_length=1) # Added username field with min_length validation
    email: str
    password: str

class TranslatePhraseRequest(BaseModel): # Moved here from above
    phrase: str
    target_lang: str

class UserLogin(BaseModel):
    email: str
    password: str

from sqlalchemy.orm import Session
from backend.utils.db_manager import get_db # Import get_db

# Dependency to get the database session
def get_db_dependency():
    yield from get_db(SessionLocal)

# Security dependency for user authentication
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db_dependency)):
    # For now, we'll just extract the email from the mock token
    # In a real app, you'd decode the JWT and verify it
    token = credentials.credentials
    if not token.startswith("mock-jwt-token-for-"):
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    email = token.replace("mock-jwt-token-for-", "")
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Removed explicit FFmpeg environment variable settings as soundfile is now used for audio I/O
# and torchaudio is no longer a core dependency for the pipeline.
# os.environ["TORCH_USE_LIBAV"] = "0"
# os.environ["TORCH_USE_EXTERNAL_FFMPEG"] = "1"
# logging.info("Backend: TORCH_USE_LIBAV set to 0. TORCH_USE_EXTERNAL_FFMPEG set to 1.")

# Removed explicit FFmpeg environment variable settings as soundfile is now used for audio I/O
# and torchaudio is no longer a core dependency for the pipeline.
# os.environ["TORCH_USE_LIBAV"] = "0"
# os.environ["TORCH_USE_EXTERNAL_FFMPEG"] = "1"
# logging.info("Backend: TORCH_USE_LIBAV set to 0. TORCH_USE_EXTERNAL_FFMPEG set to 1.")

from fastapi import (
    WebSocket,
    WebSocketDisconnect,
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Body,
    Depends # Import Depends for dependency injection
)
from starlette.responses import JSONResponse
import torch
import webrtcvad # Re-enabled
from collections import deque
import websockets.exceptions # Explicitly import websockets.exceptions
import ffmpeg # Import ffmpeg for audio conversion
from starlette.websockets import WebSocketState # Import WebSocketState

from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.piper_tts import PiperTTS
from backend.tts.coqui_tts import CoquiTTS # Import CoquiTTS
from backend.utils.audio_utils import load_audio, save_audio

# Router for FastAPI
router = APIRouter()

# Configuration
DEFAULT_STT_MODEL_SIZE = "base" # Default STT model size
DEFAULT_TTS_MODEL = "piper"
AUDIO_SAMPLE_RATE = 16000 # Standard sample rate for VAD and STT (Reverted to 16000 Hz for VAD/STT compatibility)

# VAD Configuration (matching BP xtts)
VAD_FRAME_DURATION = 20 # ms - BP xtts uses 20ms frames
VAD_AGGRESSIVENESS = 3 # Mode 3 (Most Aggressive) - Increased from 1 for noisy environments
# MIN_SPEECH_DURATION removed - BP xtts doesn't filter by duration
SILENCE_TIMEOUT = 0.3 # seconds (reduced from 1.0s for better responsiveness)
STREAMING_CHUNK_LENGTH = 0.5 # seconds

# Pre-VAD Silence Detection Configuration
# Threshold calibrated based on user testing:
# - Finger clicks: ~0.015 (should block)
# - User speech: ~0.05-0.06 (should allow)
# - Background noise: ~0.002 (should block)
SILENCE_RMS_THRESHOLD = 0.02  # Lowered to allow quieter speech through (was 0.04, then 0.08)
PRE_VAD_BUFFER_DURATION = 0.5 # seconds, how much audio to buffer before checking RMS

# Speaker Voices Directory and Metadata
SPEAKER_VOICES_DIR = "speaker_voices"
SPEAKER_VOICES_METADATA_FILE = os.path.join(SPEAKER_VOICES_DIR, "speaker_voices.json")
os.makedirs(SPEAKER_VOICES_DIR, exist_ok=True)

# Dictionary to store active sessions and their models
# Key: client_info (str), Value: Dict[str, Any] containing session_config and model instances
active_sessions: Dict[str, Dict[str, Any]] = {}

# Dictionary to track ongoing model conversions
_ongoing_conversions: Dict[str, asyncio.Lock] = {}

# Dictionary for per-session locks (prevents race conditions)
_session_locks: Dict[str, asyncio.Lock] = {}

def get_session_lock(client_info: str) -> asyncio.Lock:
    """
    Get or create a lock for a specific session to prevent race conditions.
    
    THESIS NOTE: This lock ensures thread-safe access to session_config during
    concurrent config updates from the same client.
    """
    if client_info not in _session_locks:
        _session_locks[client_info] = asyncio.Lock()
    return _session_locks[client_info]

# --- Helper Functions for Speaker Metadata ---
def _read_speaker_voices_metadata() -> list[Dict[str, Any]]:
    """Reads the speaker voices metadata from the JSON file."""
    logging.debug(f"Backend: Checking for metadata file at: {SPEAKER_VOICES_METADATA_FILE}")
    if not os.path.exists(SPEAKER_VOICES_METADATA_FILE):
        logging.warning(f"Backend: Metadata file not found at: {SPEAKER_VOICES_METADATA_FILE}")
        return []
    try:
        with open(SPEAKER_VOICES_METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Backend: Error decoding speaker voices metadata file: {e}")
        return []
    except Exception as e:
        logging.error(f"Backend: Error reading speaker voices metadata file: {e}")
        return []

def _write_speaker_voices_metadata(metadata: list[Dict[str, Any]]):
    """Writes the speaker voices metadata to the JSON file."""
    try:
        with open(SPEAKER_VOICES_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logging.error(f"Backend: Error writing speaker voices metadata file: {e}")

def get_mt_model_name(source_lang: str, target_lang: str) -> str:
    """Constructs the MT model name based on source and target languages."""
    return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

async def _convert_mt_model(model_name: str, websocket: Optional[WebSocket] = None):
    """Converts an Opus-MT model to CTranslate2 format using a subprocess."""
    model_key = model_name.replace('Helsinki-NLP/', '').replace('/', '-')
    ct2_model_dir = os.path.join("ct2_models", model_key)

    if os.path.exists(ct2_model_dir):
        logging.info(f"Backend: CTranslate2 model already exists for {model_name} at {ct2_model_dir}. Skipping conversion.")
        return

    logging.info(f"Backend: Starting conversion of {model_name} to CTranslate2 format...")
    command = [
        sys.executable,
        "backend/mt/convert_opus_mt_to_ct2.py",
        "--model_name", model_name
    ]
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        error_message = f"Failed to convert MT model {model_name}. STDOUT: {stdout.decode().strip()}, STDERR: {stderr.decode().strip()}"
        logging.error(f"Backend: ERROR: {error_message}")
        raise RuntimeError(error_message)
    
    logging.info(f"Backend: Successfully converted {model_name} to CTranslate2 format.")

def _save_audio_segment_for_debug(audio_segment_np: np.ndarray, filename_prefix: str, is_final: bool):
    """Saves an audio segment to a WAV file for debugging."""
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    final_tag = "final" if is_final else "streaming"
    filename = os.path.join(output_dir, f"{filename_prefix}_{final_tag}_{timestamp}.wav")
    try:
        sf.write(filename, audio_segment_np, AUDIO_SAMPLE_RATE)
        logging.debug(f"Backend: Saved debug audio segment to {filename}")
    except Exception as e:
        logging.error(f"Backend: Failed to save debug audio segment to {filename}: {e}")

# --- Model Initialization Functions ---
async def _initialize_stt_model(session_data: Dict[str, Any], stt_model_size: str = DEFAULT_STT_MODEL_SIZE):
    """Initializes the FasterWhisperSTT model for a given session."""
    current_stt_model = session_data.get("stt_model")
    if current_stt_model is None or current_stt_model.model_size != stt_model_size:
        init_start = time.perf_counter()
        logging.info(f"Backend: Session {session_data['client_info']}: Initializing FasterWhisperSTT model '{stt_model_size}' at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
        session_data["stt_model"] = FasterWhisperSTT(model_size=stt_model_size, compute_type="int8")
        init_end = time.perf_counter()
        logging.info(f"Backend: Session {session_data['client_info']}: FasterWhisperSTT model '{stt_model_size}' initialized at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s")
    else:
        logging.info(f"Backend: Session {session_data['client_info']}: FasterWhisperSTT model '{stt_model_size}' already initialized.")

async def _initialize_mt_model(session_data: Dict[str, Any], source_lang: str, target_lang: str, websocket: Optional[WebSocket] = None):
    """Initializes a specific CTranslate2MT model for a given session."""
    mt_models_for_session = session_data.setdefault("mt_models", {})
    model_key = f"{source_lang}-{target_lang}"

    if source_lang == target_lang:
        logging.info(f"Backend: Session {session_data['client_info']}: Skipping MT model initialization for {model_key} as source and target languages are the same.")
        if model_key in mt_models_for_session:
            del mt_models_for_session[model_key]
        return

    mt_model_name = get_mt_model_name(source_lang, target_lang)
    mt_model_path = f"ct2_models/{mt_model_name.replace('/', '--')}"

    if model_key not in mt_models_for_session or mt_models_for_session[model_key].ctranslate2_model_dir != mt_model_path:
        # Acquire a lock for this model key to prevent multiple simultaneous conversions
        if model_key not in _ongoing_conversions:
            _ongoing_conversions[model_key] = asyncio.Lock()
        
        async with _ongoing_conversions[model_key]:
            # Re-check if model is already initialized or converted after acquiring lock
            if model_key in mt_models_for_session and mt_models_for_session[model_key].ctranslate2_model_dir == mt_model_path:
                logging.info(f"Backend: Session {session_data['client_info']}: CTranslate2MT for {model_key} already initialized by another task.")
                return
            if os.path.exists(mt_model_path):
                logging.info(f"Backend: Session {session_data['client_info']}: CTranslate2 model for {mt_model_name} already exists at {mt_model_path}.")
            else:
                logging.info(f"Backend: Session {session_data['client_info']}: MT model not found locally at {mt_model_path}. Attempting dynamic conversion for {model_name}...")
                if websocket:
                    await websocket.send_text(json.dumps({"type": "mt_conversion_status", "status": "started", "model_name": model_name}))
                try:
                    await _convert_mt_model(model_name, websocket)
                    logging.info(f"Backend: Session {session_data['client_info']}: Dynamic conversion for {model_name} completed successfully.")
                    if websocket:
                        await websocket.send_text(json.dumps({"type": "mt_conversion_status", "status": "completed", "model_name": model_name}))
                except Exception as e:
                    logging.error(f"Backend: Session {session_data['client_info']}: ERROR: Dynamic conversion failed for {model_name}: {e}")
                    if websocket:
                        await websocket.send_text(json.dumps({"type": "mt_conversion_status", "status": "failed", "model_name": model_name, "error": str(e)}))
                    raise HTTPException(status_code=500, detail=f"Failed to dynamically convert MT model {model_name}: {e}")

            init_start = time.time()
            logging.info(f"Backend: Session {session_data['client_info']}: Initializing CTranslate2MT for {model_key} at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
            try:
                mt_models_for_session[model_key] = CTranslate2MT(model_path=mt_model_path, device="cpu") # Corrected model_path
                init_end = time.time()
                logging.info(f"Backend: Session {session_data['client_info']}: CTranslate2MT initialized for {model_key} on CPU (MPS fallback) at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s")
            except Exception as e:
                logging.error(f"Backend: Session {session_data['client_info']}: ERROR: Failed to initialize CTranslate2MT for {model_key}: {e}")
                if model_key in mt_models_for_session:
                    del mt_models_for_session[model_key]
                raise HTTPException(status_code=500, detail=f"Failed to initialize CTranslate2MT for {model_key} after conversion: {e}")
    else:
        logging.info(f"Backend: Session {session_data['client_info']}: CTranslate2MT for {model_key} already initialized.")

async def _initialize_tts_models(session_data: Dict[str, Any], tts_model_choice: str, speaker_wav_path: Optional[str], speaker_text: Optional[str], speaker_lang: Optional[str]):
    """Initializes the selected TTS model (Piper or CoquiTTS) for a given session."""
    
    # Ensure only the selected TTS model is initialized for this session
    if tts_model_choice == "piper":
        current_piper_tts_model = session_data.get("piper_tts_model")
        if current_piper_tts_model is None:
            init_start = time.time()
            logging.info(f"Backend: Session {session_data['client_info']}: Initializing PiperTTS with 'cs_CZ-jirka-medium' at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
            try:
                session_data["piper_tts_model"] = PiperTTS(model_id="cs_CZ-jirka-medium", device="mps" if torch.backends.mps.is_available() else "cpu")
                init_end = time.time()
                logging.info(f"Backend: Session {session_data['client_info']}: PiperTTS 'cs_CZ-jirka-medium' initialized at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s.")
            except Exception as e:
                logging.error(f"Backend: Session {session_data['client_info']}: ERROR: Failed to initialize PiperTTS 'cs_CZ-jirka-medium': {e}")
                session_data["piper_tts_model"] = None
                raise # Re-raise if Piper also fails, as there's no other fallback
        else:
            logging.info(f"Backend: Session {session_data['client_info']}: PiperTTS model 'cs_CZ-jirka-medium' already initialized.")
        session_data["coqui_tts_model"] = None # Ensure Coqui is not active for this session
    elif tts_model_choice == "xtts": # Condition for Coqui XTTS
        current_coqui_tts_model = session_data.get("coqui_tts_model")
        if current_coqui_tts_model is None:
            init_start = time.time()
            logging.info(f"Backend: Session {session_data['client_info']}: Initializing CoquiTTS (XTTS v2) at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
            try:
                # Auto-detect device (CPU/MPS/CUDA) via CoquiTTS logic
                session_data["coqui_tts_model"] = CoquiTTS() 
                init_end = time.time()
                logging.info(f"Backend: Session {session_data['client_info']}: CoquiTTS (XTTS v2) initialized at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s.")
            except Exception as e:
                logging.error(f"Backend: Session {session_data['client_info']}: ERROR: Failed to initialize CoquiTTS (XTTS v2): {e}")
                session_data["coqui_tts_model"] = None
                raise HTTPException(status_code=500, detail=f"Failed to initialize CoquiTTS (XTTS v2): {e}")
        else:
            logging.info(f"Backend: Session {session_data['client_info']}: CoquiTTS (XTTS v2) model already initialized.")
        session_data["piper_tts_model"] = None # Ensure Piper is not active for this session
    else:
        logging.warning(f"Backend: Session {session_data['client_info']}: Skipping TTS initialization as '{tts_model_choice}' is selected or invalid.")
        session_data["piper_tts_model"] = None
        session_data["coqui_tts_model"] = None # Ensure Coqui is not active for this session

async def _initialize_vad_instance(session_data: Dict[str, Any]):
    """Initializes the WebRTC VAD instance for a given session."""
    current_vad_instance = session_data.get("vad_instance")
    if current_vad_instance is None:
        init_start = time.time()
        logging.info(f"Backend: Session {session_data['client_info']}: Initializing WebRTC VAD with aggressiveness={VAD_AGGRESSIVENESS} at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
        session_data["vad_instance"] = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        init_end = time.time()
        logging.info(f"Backend: Session {session_data['client_info']}: WebRTC VAD initialized at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s")
    else:
        logging.info(f"Backend: Session {session_data['client_info']}: WebRTC VAD already initialized.")


async def initialize_all_models(client_info: str, source_lang: str, target_lang: str, tts_model_choice: str, stt_model_size: str = DEFAULT_STT_MODEL_SIZE, speaker_wav_path: Optional[str] = None, speaker_text: Optional[str] = None, speaker_lang: Optional[str] = None, vad_enabled_param: bool = True, websocket: Optional[WebSocket] = None):
    """Initializes all necessary models (STT, MT, TTS, VAD) for a given session."""
    
    session_data = active_sessions.setdefault(client_info, {
        "client_info": client_info,
        "stt_model": None,
        "mt_models": {},
        "piper_tts_model": None,
        "coqui_tts_model": None, # Add Coqui TTS model to session data
        "vad_instance": None,
        "session_config": { # Store a copy of the config for easy access
            "source_lang": source_lang,
            "target_lang": target_lang,
            "tts_model_choice": tts_model_choice,
            "stt_model_size": stt_model_size,
            "speaker_wav_path": speaker_wav_path,
            "speaker_text": speaker_text,
            "speaker_lang": speaker_lang,
            "vad_enabled": vad_enabled_param,
        }
    })

    logging.info(f"Backend: Session {client_info}: Initializing models at {time.strftime('%H:%M:%S', time.localtime(time.time()))}...")

    await _initialize_stt_model(session_data, stt_model_size)
    await _initialize_mt_model(session_data, source_lang, target_lang, websocket)
    if source_lang != "en" and source_lang != target_lang: # Also initialize en-target for auto-detection fallback, but only if not already the target
        await _initialize_mt_model(session_data, "en", target_lang, websocket)
    await _initialize_tts_models(session_data, tts_model_choice, speaker_wav_path, speaker_text, speaker_lang)
    await _initialize_vad_instance(session_data)

    logging.info(f"Backend: Session {client_info}: All models initialized (or attempted to initialize) at {time.strftime('%H:%M:%S', time.localtime(time.time()))}.")
    return {"status": "success", "message": "Models initialization triggered."}

# --- Voice Management Endpoints ---
@router.post("/register", summary="Register a new user")
async def register_user(user: UserCreate, db: Session = Depends(get_db_dependency)):
    db_user_email = db.query(User).filter(User.email == user.email).first()
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    db_user_username = db.query(User).filter(User.username == user.username).first()
    if db_user_username:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"message": "User registered successfully"}

@router.post("/token", summary="Login and get an access token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db_dependency)):
    user = db.query(User).filter(User.email == form_data.username).first() # username field is used for email
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # For now, we'll return a mock JWT token
    mock_token = f"mock-jwt-token-for-{user.email}"
    return {"access_token": mock_token, "token_type": "bearer"}

@router.post("/voices/upload", summary="Upload a new speaker voice", response_model=Dict[str, str])
async def upload_voice(
    file: UploadFile = File(...),
    voice_name: str = Form(...),
    speaker_lang: str = Form(...), # Add speaker_lang parameter
    current_user: User = Depends(get_current_user) # Authenticated user
):
    """
    Uploads a new speaker voice file and saves its metadata.
    """
    if not voice_name:
        raise HTTPException(status_code=400, detail="Voice name cannot be empty.")

    # Sanitize voice_name to create a valid filename
    sanitized_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
    if not sanitized_voice_name:
        raise HTTPException(status_code=400, detail="Sanitized voice name is empty. Please use a valid name.")

    # Ensure the voice name is unique for the current user
    metadata = _read_speaker_voices_metadata()
    user_voices = [v for v in metadata if v.get("user_id") == current_user.id]
    if any(v.get("name") == sanitized_voice_name for v in user_voices):
        raise HTTPException(status_code=409, detail=f"Voice with name '{sanitized_voice_name}' already exists for this user.")

    # Save the uploaded file
    file_extension = os.path.splitext(file.filename)[1]
    if not file_extension:
        file_extension = ".wav" # Default to .wav if no extension
    
    # Construct a unique filename using user_id and sanitized voice name
    filename = f"{current_user.id}_{sanitized_voice_name}{file_extension}"
    file_path = os.path.join(SPEAKER_VOICES_DIR, filename)

    try:
        contents = await file.read()
        
        # Use a BytesIO object for the input to ffmpeg
        input_audio_buffer = io.BytesIO(contents)
        output_audio_buffer = io.BytesIO()

        try:
            # Use ffmpeg to convert the input audio (which might be webm) to WAV format
            # and ensure it's 16kHz mono.
            # Use ffmpeg to convert the input audio (which might be webm) to WAV format
            # and ensure it's 16kHz mono.
            # Construct the ffmpeg command directly
            command = [
                "ffmpeg",
                "-i", "pipe:0", # Input from stdin
                "-f", "wav", # Output format WAV
                "-acodec", "pcm_s16le", # PCM 16-bit little-endian
                "-ar", str(AUDIO_SAMPLE_RATE), # Sample rate
                "-ac", "1", # Mono audio
                "pipe:1" # Output to stdout
            ]

            # If the input is webm, specify the input format and codec
            if 'webm' in file.content_type:
                command.insert(1, "webm")
                command.insert(1, "-f")
                command.insert(3, "opus")
                command.insert(3, "-acodec")

            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate(input=input_audio_buffer.getvalue())

            if process.returncode != 0:
                error_message = f"FFmpeg conversion failed. STDOUT: {stdout.decode().strip()}, STDERR: {stderr.decode().strip()}" # type: ignore
                logging.error(f"Backend: {error_message}") # type: ignore
                raise RuntimeError(error_message) # type: ignore
            
            output_audio_buffer.write(stdout)
            output_audio_buffer.seek(0) # Rewind buffer for reading

            # Now load the converted WAV audio using soundfile
            audio_np, sr = load_audio(output_audio_buffer)
            
            if sr != AUDIO_SAMPLE_RATE:
                logging.warning(f"Backend: Audio after FFmpeg conversion is not {AUDIO_SAMPLE_RATE}Hz. Resampling with soundfile.") # type: ignore
                # load_audio already handles resampling if needed, but this log indicates a potential issue with ffmpeg command if it doesn't output 16kHz
            
            save_audio(file_path, audio_np, AUDIO_SAMPLE_RATE)
            logging.info(f"Backend: Uploaded voice '{voice_name}' saved to {file_path}") # type: ignore

        except Exception as e:
            logging.error(f"Backend: Error during FFmpeg conversion or audio processing for voice '{voice_name}': {e}") # type: ignore
            raise HTTPException(status_code=500, detail=f"Failed to process uploaded audio: {e}") # type: ignore

        # Update metadata
        new_voice_entry = {
            "id": str(len(metadata) + 1), # Simple ID generation
            "user_id": current_user.id,
            "name": sanitized_voice_name,
            "filename": filename,
            "path": file_path,
            "language": speaker_lang, # Store the speaker language
            "upload_time": time.time()
        }
        metadata.append(new_voice_entry)
        _write_speaker_voices_metadata(metadata)

        return {"status": "success", "message": f"Voice '{sanitized_voice_name}' uploaded successfully."}
    except Exception as e:
        logging.error(f"Backend: Error uploading voice '{voice_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {e}")

@router.put("/voices/rename", summary="Rename an existing speaker voice", response_model=Dict[str, str])
async def rename_voice(
    request: RenameVoiceRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Renames an existing speaker voice file and updates its metadata.
    """
    metadata = _read_speaker_voices_metadata()
    voice_found = False
    for voice in metadata:
        if voice.get("user_id") == current_user.id and voice.get("name") == request.old_name:
            # Check if new name already exists for this user
            if any(v.get("user_id") == current_user.id and v.get("name") == request.new_name for v in metadata):
                raise HTTPException(status_code=409, detail=f"Voice with name '{request.new_name}' already exists for this user.")

            old_filename = voice["filename"]
            file_extension = os.path.splitext(old_filename)[1]
            new_filename = f"{current_user.id}_{request.new_name}{file_extension}"
            
            old_file_path = os.path.join(SPEAKER_VOICES_DIR, old_filename)
            new_file_path = os.path.join(SPEAKER_VOICES_DIR, new_filename)

            try:
                os.rename(old_file_path, new_file_path)
                voice["name"] = request.new_name
                voice["filename"] = new_filename
                voice["path"] = new_file_path
                _write_speaker_voices_metadata(metadata)
                logging.info(f"Backend: Renamed voice from '{request.old_name}' to '{request.new_name}' for user {current_user.id}")
                voice_found = True
                break
            except FileNotFoundError:
                logging.error(f"Backend: File not found for voice '{request.old_name}' at {old_file_path}")
                raise HTTPException(status_code=404, detail=f"Voice file '{request.old_name}' not found.")
            except Exception as e:
                logging.error(f"Backend: Error renaming voice '{request.old_name}': {e}")
                raise HTTPException(status_code=500, detail=f"Failed to rename voice: {e}")
    
    if not voice_found:
        raise HTTPException(status_code=404, detail=f"Voice '{request.old_name}' not found for this user.")
    
    return {"status": "success", "message": f"Voice '{request.old_name}' renamed to '{request.new_name}' successfully."}

@router.delete("/voices/delete", summary="Delete a speaker voice", response_model=Dict[str, str])
async def delete_voice(
    request: DeleteVoiceRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Deletes a speaker voice file and removes its metadata.
    """
    metadata = _read_speaker_voices_metadata()
    updated_metadata = []
    voice_deleted = False
    for voice in metadata:
        if voice.get("user_id") == current_user.id and voice.get("filename") == request.filename:
            file_path = os.path.join(SPEAKER_VOICES_DIR, request.filename)
            try:
                os.remove(file_path)
                logging.info(f"Backend: Deleted voice file {file_path}")
                voice_deleted = True
            except FileNotFoundError:
                logging.warning(f"Backend: File not found for deletion: {file_path}. Removing metadata anyway.")
                voice_deleted = True # Consider it deleted if file is already gone
            except Exception as e:
                logging.error(f"Backend: Error deleting voice file {file_path}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete voice file: {e}")
        else:
            updated_metadata.append(voice)
    
    if not voice_deleted:
        raise HTTPException(status_code=404, detail=f"Voice file '{request.filename}' not found for this user.")
    
    _write_speaker_voices_metadata(updated_metadata)
    return {"status": "success", "message": f"Voice '{request.filename}' deleted successfully."}

@router.get("/voices", summary="Get all speaker voices for the current user", response_model=List[Dict[str, Any]])
async def get_voices(
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves a list of all speaker voices uploaded by the current authenticated user.
    """
    metadata = _read_speaker_voices_metadata()
    user_voices = [
        {"id": v.get("id", str(uuid.uuid4())), "name": v["name"], "filename": v.get("filename", "unknown_filename.wav"), "path": v.get("path", "unknown_path"), "language": v.get("language", "unknown"), "upload_time": v.get("upload_time", 0)}
        for v in metadata if v.get("user_id") is None or v.get("user_id") == current_user.id
    ]
    logging.info(f"Backend: Retrieved {len(user_voices)} voices for user {current_user.id}")
    return user_voices

def get_initialized_models(client_info: str, session_config: Dict[str, Any]) -> Tuple[Optional[FasterWhisperSTT], Optional[CTranslate2MT], Optional[Any], Optional[Any], str]:
    """Returns the currently initialized model instances and current TTS choice based on session config."""
    session_data = active_sessions.get(client_info)
    if not session_data:
        logging.warning(f"Backend: Session {client_info} not found in active_sessions. Cannot retrieve models.")
        return None, None, None, None, session_config["tts_model_choice"] # Return None for models if session not found

    stt_model_instance = session_data.get("stt_model")
    main_mt_model = session_data["mt_models"].get(f"{session_config['source_lang']}-{session_config['target_lang']}")
    vad_instance = session_data.get("vad_instance")
    
    tts_model_instance = None
    if session_config["tts_model_choice"] == "piper":
        tts_model_instance = session_data.get("piper_tts_model")
    elif session_config["tts_model_choice"] == "xtts": # Coqui XTTS
        tts_model_instance = session_data.get("coqui_tts_model")
    
    return stt_model_instance, main_mt_model, tts_model_instance, vad_instance, session_config["tts_model_choice"]

# --- Core Speech Processing Pipeline ---
async def _process_speech_segment_pipeline(
    websocket: WebSocket, audio_segment_np: np.ndarray, segment_start_time: float, is_final: bool = True,
    session_config: Dict[str, Any] = None # Add session_config parameter
):
    """Processes a detected speech segment: STT -> MT -> TTS -> Send audio."""
    if session_config is None:
        logging.error("Backend: _process_speech_segment_pipeline called without session_config.")
        return

    loop = asyncio.get_event_loop()

    source_lang = session_config["source_lang"]
    target_lang = session_config["target_lang"]
    tts_model_choice = session_config["tts_model_choice"]
    vad_enabled = session_config["vad_enabled"]
    speaker_wav_path = session_config["speaker_wav_path"]
    speaker_text = session_config["speaker_text"]
    speaker_lang = session_config["speaker_lang"]

    if audio_segment_np.size == 0:
        logging.warning(f"Backend: Received empty audio segment (Final: {is_final}). Skipping processing.")
        return

    logging.debug(f"Backend: Audio segment size for STT: {audio_segment_np.size} samples ({audio_segment_np.size/AUDIO_SAMPLE_RATE:.2f}s). Final: {is_final}")
    logging.info(
        f"Backend: Processing speech segment (Final: {is_final}). Session Config: source_lang={source_lang}, target_lang={target_lang}, TTS choice={tts_model_choice}, VAD enabled: {vad_enabled}, Speaker: {speaker_wav_path}"
    )
    # _save_audio_segment_for_debug(audio_segment_np, "stt_input", is_final) # Removed for less clutter

    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown:unknown"
    session_data = active_sessions.get(client_info)
    if not session_data:
        logging.error(f"Backend: Session {client_info} not found in active_sessions. Cannot process speech segment.")
        # Do not send message if session is not found, as websocket might be closed
        return

    # Helper to safely send messages
    async def safe_send(message_type: str, payload: Dict[str, Any], is_bytes: bool = False):
        # Check if the WebSocket is still connected before sending
        if websocket.client_state == WebSocketState.DISCONNECTED:
            logging.debug(f"Backend: WebSocket for {websocket.scope['client'][0]}:{websocket.scope['client'][1]} is disconnected. Skipping send of {message_type}.")
            return
        try:
            if is_bytes:
                await websocket.send_bytes(payload)
            else:
                await websocket.send_text(json.dumps(payload))
        except websockets.exceptions.ConnectionClosedOK:
            logging.info(f"Backend: WebSocket connection already closed for {websocket.scope['client'][0]}:{websocket.scope['client'][1]}. Skipping send of {message_type}.")
        except Exception as e:
            logging.error(f"Backend: Error sending {message_type} to client {websocket.scope['client'][0]}:{websocket.scope['client'][1]}: {e}")

    stt_model_instance = session_data.get("stt_model")
    mt_models_for_session = session_data.get("mt_models", {})
    piper_tts_model_instance = session_data.get("piper_tts_model")
    coqui_tts_model_instance = session_data.get("coqui_tts_model") # Get Coqui TTS instance

    # --- STT ---
    websocket.timestamps["stt_start"].append(time.perf_counter())
    stt_start_time = time.perf_counter()
    # Use the session-specific STT model
    transcribed_segments, stt_time, detected_source_lang = await loop.run_in_executor(
        None,
        lambda: stt_model_instance.transcribe_audio(
            audio_segment_np,
            AUDIO_SAMPLE_RATE,
            language=None if source_lang == "auto" else source_lang,
            vad_filter=False if vad_enabled else True # Disable FasterWhisper's VAD if WebRTC VAD is enabled
        ),
    )
    if source_lang == "auto" and detected_source_lang:
        session_config["source_lang"] = detected_source_lang # Update session config with detected language
        logging.info(f"Backend: Session {client_info}: Source language detected as: {session_config['source_lang']}")
    stt_end_time = time.perf_counter()
    websocket.timestamps["stt_end"].append(stt_end_time)
    stt_total_time = stt_end_time - stt_start_time
    input_to_stt_latency = stt_end_time - segment_start_time # Calculate input->STT latency
    transcribed_text = " ".join([s.text for s in transcribed_segments]) if transcribed_segments else ""
    logging.debug(f"Backend: Session {client_info}: STT raw segments: {transcribed_segments}")
    logging.debug(f"Backend: Session {client_info}: Concatenated transcribed_text: '{transcribed_text}'")
    logging.info(f"Backend: Session {client_info}: STT completed. Transcribed: '{transcribed_text}' (Detected Lang: {detected_source_lang}). Time: {stt_total_time:.2f}s. Input->STT Latency: {input_to_stt_latency:.2f}s")

    # Filter out transcriptions that are just periods/punctuation (background noise artifacts)
    cleaned_text = transcribed_text.strip().replace('.', '').replace(',', '').replace('!', '').replace('?', '').strip()
    if not cleaned_text:
        logging.info(f"Backend: Session {client_info}: Transcription contains only punctuation (likely background noise). Skipping translation and TTS.")
        return

    if not transcribed_text.strip():
        logging.warning(f"Backend: Session {client_info}: STT produced no meaningful text for segment (Final: {is_final}). Skipping translation and TTS.")
        await safe_send("transcription_result", {"type": "transcription_result", "transcribed": "", "is_final": is_final, "metrics": {"stt_time": stt_total_time}})
        return
    await safe_send("transcription_result", {"type": "transcription_result", "transcribed": transcribed_text, "is_final": is_final, "metrics": {"stt_time": stt_total_time}})

    # --- MT ---
    translated_text = transcribed_text
    mt_total_time = 0.0
    if source_lang != target_lang:
        websocket.timestamps["mt_start"].append(time.perf_counter())
        main_mt_model = mt_models_for_session.get(f"{source_lang}-{target_lang}")
        if main_mt_model is None:
            logging.warning(f"WARNING: Session {client_info}: MT model for {source_lang}-{target_lang} not initialized. Skipping translation.")
            await safe_send("error", {"type": "error", "message": f"MT model for {source_lang}-{target_lang} not initialized. Skipping translation."})
            return
        mt_start_time = time.perf_counter()
        translated_text, _ = await loop.run_in_executor(None, lambda: main_mt_model.translate(transcribed_text, source_lang, target_lang))
        mt_total_time = time.perf_counter() - mt_start_time
        websocket.timestamps["mt_end"].append(time.perf_counter())
    else:
        logging.info(f"Backend: Session {client_info}: Source and target languages are the same ({source_lang}). Skipping machine translation.")
    logging.info(f"Backend: Session {client_info}: MT completed for {source_lang}-{target_lang}. Translated: '{translated_text}'. Time: {mt_total_time:.2f}s")

    if not translated_text:
        logging.warning(f"Backend: Session {client_info}: MT produced no translated text for '{transcribed_text}' (Final: {is_final}). Skipping TTS.")
        await safe_send("translation_result", {"type": "translation_result", "translated": "", "is_final": is_final, "metrics": {"mt_time": mt_total_time}})
        return
    
    # Send translation result to frontend
    await safe_send("translation_result", {"type": "translation_result", "translated": translated_text, "is_final": is_final, "metrics": {"mt_time": mt_total_time}})
    logging.debug(f"Backend: Session {client_info}: Sent translation_result: '{translated_text}'")

    # --- TTS ---
    # Standard non-streaming synthesis (Piper or fallback)
    logging.info(f"Backend: Session {client_info}: Starting TTS synthesis...")

    # Handle Slovak fallback for XTTS if it's the chosen model but target_lang is 'sk'
    if tts_model_choice == "xtts" and target_lang == "sk":
        logging.warning(f"Backend: Session {client_info}: XTTS does not support Slovak. Falling back to Piper for generic voice.")
        await safe_send("error", {"type": "error", "message": "XTTS does not support Slovak. Using generic voice (Piper) instead."})
        if piper_tts_model_instance:
            tts_model_instance = piper_tts_model_instance # Use Piper for fallback
        else:
                logging.error(f"Backend: Session {client_info}: Piper TTS model not initialized for Slovak fallback.")
                await safe_send("error", {"type": "error", "message": "Piper TTS not ready for Slovak fallback."})
                return
        
        # Check for speaker_wav_path requirement for XTTS if it's still the chosen model
        if tts_model_choice == "xtts" and not speaker_wav_path:
            logging.warning(f"Backend: Session {client_info}: XTTS selected but no valid speaker voice provided. Sending error to client.")
            await safe_send("error", {"type": "error", "message": "XTTS requires a selected voice. Please choose or record one."})
            return
     # --- TTS ---
    websocket.timestamps["tts_start"].append(time.perf_counter())
    tts_start_time = time.perf_counter()
    
    # Initialize tts_model_instance outside the conditional blocks
    tts_model_instance = None
    audio_wav = None
    sample_rate = None
    tts_total_time = 0.0
    
    # COQUI TTS (XTTS) - Streaming Synthesis
    if tts_model_choice == "xtts" and coqui_tts_model_instance:
        logging.info(f"Backend: Session {client_info}: Starting XTTS streaming synthesis...")
        
        queue = asyncio.Queue()
        
        def run_streaming_tts():
            try:
                # synthesize_stream yields numpy arrays
                for chunk in coqui_tts_model_instance.synthesize_stream(
                    text=translated_text,
                    language=target_lang,
                    speaker_wav_path=speaker_wav_path
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel
            except Exception as e:
                logging.error(f"Backend: Error in XTTS streaming thread: {e}")
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Run the blocking generator in a separate thread
        threading.Thread(target=run_streaming_tts, daemon=True).start()
        
        chunk_count = 0
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            
            chunk_count += 1
            # Convert numpy chunk to WAV bytes
            audio_buffer = io.BytesIO()
            # Write chunk to buffer as WAV
            sf.write(audio_buffer, chunk, 24000, format="WAV") 
            await safe_send("tts_audio", audio_buffer.getvalue(), is_bytes=True)
        
        tts_total_time = time.perf_counter() - tts_start_time
        logging.info(f"Backend: Session {client_info}: XTTS streaming completed. Time: {tts_total_time:.2f}s. Chunks: {chunk_count}")
        
    elif tts_model_choice == "piper" and piper_tts_model_instance:
        # Piper TTS (Non-streaming)
        tts_model_instance = piper_tts_model_instance
        logging.info(f"Backend: Session {client_info}: Starting Piper TTS synthesis...")
        audio_wav, sample_rate, tts_latency = await loop.run_in_executor(
            None,
            lambda: tts_model_instance.synthesize(translated_text)
        )
        tts_total_time = time.perf_counter() - tts_start_time
    
    else:
        logging.warning(f"Backend: Session {client_info}: Selected TTS model '{tts_model_choice}' not initialized or invalid. No TTS will be performed.")
        await safe_send("error", {"type": "error", "message": f"TTS model '{tts_model_choice}' not ready or invalid. No TTS output."})
        return

    if audio_wav is not None and sample_rate is not None:
        # Ensure audio_wav is a numpy array
        if isinstance(audio_wav, torch.Tensor):
            audio_wav = audio_wav.cpu().numpy()
        
        _save_audio_segment_for_debug(audio_wav, "tts_output", is_final)
        audio_buffer = io.BytesIO()
        # Use soundfile to write the audio to a WAV buffer
        await loop.run_in_executor(None, lambda: sf.write(audio_buffer, audio_wav, sample_rate, format="WAV"))
        # Send the WAV data as bytes over the WebSocket
        await safe_send("tts_audio", audio_buffer.getvalue(), is_bytes=True)
    elif tts_model_choice != "xtts":
        # Only log warning if we expected audio (i.e. not xtts which handled it above)
        logging.warning(f"Backend: Session {client_info}: {tts_model_choice} TTS model not initialized or produced no audio. No TTS will be performed for '{translated_text}'.")
        await safe_send("error", {"type": "error", "message": f"{tts_model_choice} TTS model not ready or invalid. No TTS output."})
        return
    websocket.timestamps["tts_end"].append(time.perf_counter()) # Corrected to log end time
    logging.info(f"Backend: Session {client_info}: TTS completed. Time: {tts_total_time:.2f}s")

    metrics_payload = {"stt_time": stt_total_time, "mt_time": mt_total_time, "tts_time": tts_total_time, "input_to_stt_latency": input_to_stt_latency}
    if is_final:
        metrics_payload["total_latency"] = stt_total_time + mt_total_time + tts_total_time
        logging.info(f"Backend: Sending final_metrics: {metrics_payload}")
    else:
        logging.debug(f"Backend: Sending streaming metrics: {metrics_payload}")
    await safe_send("final_metrics", {"type": "final_metrics", "metrics": metrics_payload, "is_final": is_final})

# --- WebSocket Handler ---
async def handle_audio_stream(websocket: WebSocket):
    # Use websocket.scope["client"] to get client host and port
    client_host, client_port = websocket.scope["client"]
    client_info = f"{client_host}:{client_port}"
    logging.info(f"Client connected from {client_info}")
    
    # Initialize timestamps attribute for this WebSocket connection
    websocket.timestamps = {
        "audio_input_start": [], "audio_input_end": [],
        "stt_start": [], "stt_end": [],
        "mt_start": [], "mt_end": [],
        "tts_start": [], "tts_end": [],
        "tts_audio_sent": []
    }

    # Retrieve or initialize session data
    session_data = active_sessions.setdefault(client_info, {
        "client_info": client_info,
        "stt_model": None,
        "mt_models": {},
        "piper_tts_model": None,
        "coqui_tts_model": None, # Add Coqui TTS model to session data
        "vad_instance": None,
        "session_config": { # Default config, will be updated by client
            "source_lang": "en",
            "target_lang": "sk",
            "tts_model_choice": DEFAULT_TTS_MODEL,
            "stt_model_size": DEFAULT_STT_MODEL_SIZE,
            "speaker_wav_path": None,
            "speaker_text": None,
            "speaker_lang": None,
            "vad_enabled": True,
        }
    })
    session_config = session_data["session_config"] # Reference to the session's config

    audio_queue = deque()
    speech_frames = [] # This will accumulate all VAD-detected speech frames for the entire utterance
    last_speech_time = time.perf_counter()
    in_speech_segment = False
    last_streaming_process_time = time.perf_counter() # Track last time a streaming chunk was processed
    
    # New variables for pre-VAD silence detection
    frame_size_samples = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
    streaming_chunk_samples = int(AUDIO_SAMPLE_RATE * STREAMING_CHUNK_LENGTH)
    pre_vad_buffer_samples = int(AUDIO_SAMPLE_RATE * PRE_VAD_BUFFER_DURATION)

    # New variables for pre-VAD silence detection
    pre_vad_buffer = deque()
    pre_vad_threshold_met = False

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive": # Check for the correct message type
                if "text" in message:
                    data = json.loads(message["text"])
                    if data["type"] == "start":
                        logging.info("Received 'start' command. Beginning transcription.")
                        audio_queue.clear()
                        speech_frames.clear()
                        pre_vad_buffer.clear() # Clear pre-VAD buffer on start
                        pre_vad_threshold_met = False # Reset threshold flag
                        in_speech_segment = False
                        last_speech_time = time.time()
                        last_streaming_process_time = time.perf_counter()
                    elif data["type"] == "stop":
                        logging.info("Received 'stop' command. Stopping transcription.")
                        if speech_frames:
                            await _process_speech_segment_pipeline(websocket, np.concatenate(speech_frames), last_speech_time, is_final=True, session_config=session_config)
                            speech_frames.clear()
                        in_speech_segment = False
                        pre_vad_buffer.clear() # Clear pre-VAD buffer on stop
                        pre_vad_threshold_met = False # Reset threshold flag
                    elif data["type"] == "config_update":
                        # THESIS NOTE: Use session lock to prevent race conditions when
                        # multiple config updates arrive concurrently for the same session
                        session_lock = get_session_lock(client_info)
                        async with session_lock:
                            new_source_lang = data.get("source_lang", session_config["source_lang"])
                            new_target_lang = data.get("target_lang", session_config["target_lang"])
                            new_tts_model_choice = data.get("tts_model_choice", session_config["tts_model_choice"])
                            new_speaker_wav_path = data.get("speaker_wav_path", session_config["speaker_wav_path"])
                            new_speaker_text = data.get("speaker_text", session_config["speaker_text"])
                            new_speaker_lang = data.get("speaker_lang", session_config["speaker_lang"])
                            new_vad_enabled = data.get("vad_enabled", session_config["vad_enabled"]) # Get new VAD enabled state
                            logging.debug(f"Backend: Received config_update: {data}")

                            config_changed = (
                                new_source_lang != session_config["source_lang"] or
                                new_target_lang != session_config["target_lang"] or
                                new_tts_model_choice != session_config["tts_model_choice"] or
                                new_speaker_wav_path != session_config["speaker_wav_path"] or
                                new_speaker_text != session_config["speaker_text"] or
                                new_speaker_lang != session_config["speaker_lang"] or
                                new_vad_enabled != session_config["vad_enabled"]
                            )

                            if config_changed:
                                logging.info(f"Backend: Configuration changed. Re-initializing models with: Source={new_source_lang}, Target={new_target_lang}, TTS={new_tts_model_choice}, SpeakerWav={new_speaker_wav_path}, VAD_Enabled={new_vad_enabled}")
                                await initialize_all_models(
                                    client_info,
                                    new_source_lang, new_target_lang, new_tts_model_choice,
                                    stt_model_size=session_config["stt_model_size"], # STT model size is not updated via config_update, keep current
                                    speaker_wav_path=new_speaker_wav_path, speaker_text=new_speaker_text,
                                    speaker_lang=new_speaker_lang, vad_enabled_param=new_vad_enabled, websocket=websocket
                                )
                                
                                # Check if source language changed and send a translated notification
                                if new_source_lang != session_config["source_lang"]:
                                    notification_phrase = f"Input language changed to {new_source_lang}."
                                    translated_notification = notification_phrase
                                    
                                    # Attempt to translate the notification phrase
                                    if new_source_lang != new_target_lang:
                                        # Use session-specific mt_models for notification translation
                                        mt_models_for_notification = session_data.get("mt_models", {})
                                        mt_model_instance = mt_models_for_notification.get(f"en-{new_target_lang}")
                                        if mt_model_instance is None:
                                            # Try to initialize the en-target MT model for this session if not already
                                            try:
                                                await _initialize_mt_model(session_data, "en", new_target_lang, websocket)
                                                mt_model_instance = session_data["mt_models"].get(f"en-{new_target_lang}")
                                            except Exception as e:
                                                logging.warning(f"Backend: Session {client_info}: Could not initialize en-{new_target_lang} MT model for notification: {e}")

                                        if mt_model_instance:
                                            loop = asyncio.get_event_loop()
                                            translated_notification, _ = await loop.run_in_executor(
                                                None,
                                                lambda: mt_model_instance.translate(notification_phrase, "en", new_target_lang)
                                            )
                                            logging.info(f"Backend: Session {client_info}: Translated notification: '{translated_notification}'")
                                        else:
                                            logging.warning(f"Backend: Session {client_info}: MT model for en-{new_target_lang} not available for notification translation.")
                                    
                                    await websocket.send_text(json.dumps({"type": "status", "message": translated_notification}))
                                    await websocket.send_text(json.dumps({"type": "notification", "message": translated_notification, "type": "info"})) # Send as a notification type
                                
                                session_config["source_lang"] = new_source_lang
                                session_config["target_lang"] = new_target_lang
                                session_config["tts_model_choice"] = new_tts_model_choice
                                session_config["speaker_wav_path"] = new_speaker_wav_path
                                session_config["speaker_text"] = new_speaker_text
                                session_config["speaker_lang"] = new_speaker_lang
                                session_config["vad_enabled"] = new_vad_enabled # Update session VAD state
                                
                                await websocket.send_text(json.dumps({"type": "status", "message": "Configuration updated and models re-initialized."}))
                            else:
                                logging.info(f"Backend: Session {client_info}: Configuration updated, but no model re-initialization needed.")
                                await websocket.send_text(json.dumps({"type": "status", "message": "Configuration updated."}))
                elif "bytes" in message:
                    stt_model_instance = session_data.get("stt_model")
                    vad_instance_session = session_data.get("vad_instance") # Get session-specific VAD instance

                    if stt_model_instance is None:
                        logging.warning(f"Backend: Session {client_info}: STT model not initialized. Skipping audio processing.")
                        continue

                    audio_np = np.frombuffer(message["bytes"], dtype=np.float32)
                    
                    # Sanitize audio_np: replace NaN/Inf with 0
                    if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                        logging.warning("Backend: Detected NaN or Inf in audio_np. Sanitizing to 0.")
                        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)

                    logging.debug(f"Backend: Received {audio_np.size} audio samples from frontend. First 10 samples: {audio_np[:10]}")
                    logging.debug(f"Backend: audio_np min: {np.min(audio_np)}, max: {np.max(audio_np)}")
                    logging.debug(f"Backend: audio_np contains NaN: {np.isnan(audio_np).any()}, Inf: {np.isinf(audio_np).any()}")
                    logging.debug(f"Backend: audio_np shape: {audio_np.shape}, dtype: {audio_np.dtype}")
                    # Skip all-zero audio chunks entirely to prevent false VAD triggers
                    if not audio_np.any():
                        logging.warning("Backend: Received an all-zero audio chunk. Skipping processing.")
                        continue

                    # Calculate RMS on raw audio (not normalized) to get accurate level readings
                    # Clip to ensure values are within valid range before RMS calculation
                    audio_np_clipped = np.clip(audio_np, -1.0, 1.0)
                    rms = np.sqrt(np.mean(audio_np_clipped**2))
                    
                    # Ensure RMS is a valid number before sending
                    if np.isnan(rms) or np.isinf(rms):
                        logging.warning(f"Backend: Calculated RMS is NaN or Inf ({rms}). Sending 0.0 to frontend.")
                        rms = 0.0
                    
                    await websocket.send_text(json.dumps({"type": "audio_level", "level": float(rms)}))

                    # Pre-VAD silence detection logic
                    if not pre_vad_threshold_met:
                        pre_vad_buffer.extend(audio_np)
                        if len(pre_vad_buffer) >= pre_vad_buffer_samples:
                            current_pre_vad_buffer_np = np.array(list(pre_vad_buffer), dtype=np.float32)
                            
                            # Sanitize pre-VAD buffer: replace NaN/Inf with 0
                            if np.isnan(current_pre_vad_buffer_np).any() or np.isinf(current_pre_vad_buffer_np).any():
                                logging.warning("Backend: Detected NaN or Inf in pre-VAD buffer. Sanitizing to 0.")
                                current_pre_vad_buffer_np = np.nan_to_num(current_pre_vad_buffer_np, nan=0.0, posinf=0.0, neginf=0.0)

                            buffer_rms = np.sqrt(np.mean(current_pre_vad_buffer_np**2))
                            
                            # Ensure buffer_rms is a valid number
                            if np.isnan(buffer_rms) or np.isinf(buffer_rms):
                                logging.warning(f"Backend: Calculated buffer_rms is NaN or Inf ({buffer_rms}). Treating as 0.0.")
                                buffer_rms = 0.0

                            logging.info(f"Backend: Pre-VAD buffer RMS: {buffer_rms:.4f} (Threshold: {SILENCE_RMS_THRESHOLD})")
                            if buffer_rms > SILENCE_RMS_THRESHOLD:
                                logging.info(f"Backend: Pre-VAD silence threshold met. Starting VAD processing.")
                                pre_vad_threshold_met = True
                                # Add the buffered audio to the main audio_queue for VAD processing
                                audio_queue.extend(pre_vad_buffer)
                            pre_vad_buffer.clear() # Clear buffer after check
                    
                    if pre_vad_threshold_met: # Only process with VAD if threshold met
                        audio_queue.extend(audio_np)

                    if session_config["vad_enabled"] and pre_vad_threshold_met: # Only run VAD if enabled and threshold met
                        if vad_instance_session is None:
                            logging.warning(f"Backend: Session {client_info}: VAD is enabled but vad_instance is not initialized. Skipping VAD processing.")
                            # Fallback to processing all audio if VAD is not ready
                            if len(audio_queue) >= streaming_chunk_samples:
                                chunk_to_process = np.array(list(audio_queue), dtype=np.float32)
                                # Sanitize chunk_to_process before pipeline
                                if np.isnan(chunk_to_process).any() or np.isinf(chunk_to_process).any():
                                    logging.warning("Backend: Detected NaN or Inf in chunk_to_process (VAD disabled fallback). Sanitizing to 0.")
                                    chunk_to_process = np.nan_to_num(chunk_to_process, nan=0.0, posinf=0.0, neginf=0.0)
                                await _process_speech_segment_pipeline(websocket, chunk_to_process, time.perf_counter(), is_final=False, session_config=session_config)
                                audio_queue.clear()
                            continue

                        while len(audio_queue) >= frame_size_samples:
                            frame_float33 = np.array([audio_queue.popleft() for _ in range(frame_size_samples)], dtype=np.float32)
                            
                            # Sanitize frame_float33: replace NaN/Inf with 0
                            if np.isnan(frame_float33).any() or np.isinf(frame_float33).any():
                                logging.warning("Backend: Detected NaN or Inf in VAD frame_float33. Sanitizing to 0.")
                                frame_float33 = np.nan_to_num(frame_float33, nan=0.0, posinf=0.0, neginf=0.0)

                            logging.debug(f"Backend: VAD frame_float33 min: {np.min(frame_float33)}, max: {np.max(frame_float33)}")
                            logging.debug(f"Backend: VAD frame_float33 contains NaN: {np.isnan(frame_float33).any()}, Inf: {np.isinf(frame_float33).any()}")
                            logging.debug(f"Backend: VAD frame_float33 shape: {frame_float33.shape}, dtype: {frame_float33.dtype}")
                            # Add a check for all zeros
                            if not frame_float33.any():
                                logging.warning("Backend: VAD received an all-zero audio frame.")

                            # DO NOT Normalize frame_float33 to [-1.0, 1.0] range individually!
                            # This destroys relative volume information and makes noise look like speech.
                            # Just clip and convert to int16.
                            
                            # Clip to ensure values are within [-1.0, 1.0] before converting to int16
                            frame_float33_clipped = np.clip(frame_float33, -1.0, 1.0)
                            frame_int16 = (frame_float33_clipped * 32767).astype(np.int16) # Use 32767 for max int16 value
                            logging.debug(f"Backend: Session {client_info}: VAD processing frame. AUDIO_SAMPLE_RATE: {AUDIO_SAMPLE_RATE}, frame_int16.dtype: {frame_int16.dtype}")

                            vad_start_time = time.perf_counter()
                            is_speech = vad_instance_session.is_speech(frame_int16.tobytes(), AUDIO_SAMPLE_RATE)
                            vad_end_time = time.perf_counter()
                            logging.debug(f"Backend: Session {client_info}: VAD processing took: {(vad_end_time - vad_start_time)*1000:.2f}ms")
                            
                            frame_rms_float = np.sqrt(np.mean(frame_float33_clipped**2)) # Use clipped frame for RMS
                            # Ensure frame_rms_float is a valid number
                            if np.isnan(frame_rms_float) or np.isinf(frame_rms_float):
                                logging.warning(f"Backend: Calculated frame_rms_float is NaN or Inf ({frame_rms_float}). Treating as 0.0.")
                                frame_rms_float = 0.0

                            logging.debug(f"Backend: Session {client_info}: VAD is_speech: {is_speech}. Frame RMS (float32): {frame_rms_float:.4f}")

                            if is_speech:
                                speech_frames.append(frame_float33)
                                if not in_speech_segment:
                                    logging.info(f"Backend: Speech segment STARTED at {time.perf_counter():.3f}s")
                                    in_speech_segment = True
                                last_speech_time = time.perf_counter()
                                
                                # Safety Valve: Force process if speech segment is too long (e.g., stuck due to noise)
                                current_speech_duration = len(speech_frames) * (VAD_FRAME_DURATION / 1000.0)
                                if current_speech_duration > 15.0: # 15 seconds max
                                    logging.warning(f"Backend: Speech segment exceeded 15s ({current_speech_duration:.2f}s). Forcing processing.")
                                    final_speech_segment_np = np.concatenate(speech_frames)
                                    await _process_speech_segment_pipeline(websocket, final_speech_segment_np, last_speech_time, is_final=True, session_config=session_config)
                                    speech_frames.clear()
                                    in_speech_segment = False
                                    # Reset buffers
                                    audio_queue.clear()
                                    pre_vad_buffer.clear()
                                    pre_vad_threshold_met = False
                                    last_speech_time = 0 # Reset

                                
                                # DISABLED: Streaming chunk processing causes Whisper hallucinations on short non-speech sounds
                                # Only process after silence timeout (matching BP xtts behavior)
                                # # Check if enough speech has accumulated for a streaming chunk
                                # if in_speech_segment and (time.perf_counter() - last_streaming_process_time) >= STREAMING_CHUNK_LENGTH:
                                #     # Process the current speech_frames for streaming
                                #     streaming_segment_np = np.concatenate(speech_frames)
                                #     # Sanitize streaming_segment_np before pipeline
                                #     if np.isnan(streaming_segment_np).any() or np.isinf(streaming_segment_np).any():
                                #         logging.warning("Backend: Detected NaN or Inf in streaming_segment_np. Sanitizing to 0.")
                                #         streaming_segment_np = np.nan_to_num(streaming_segment_np, nan=0.0, posinf=0.0, neginf=0.0)
                                #     await _process_speech_segment_pipeline(websocket, streaming_segment_np, last_speech_time, is_final=False, session_config=session_config)
                                #     last_streaming_process_time = time.perf_counter()
                                #     # CRITICAL FIX: Clear speech_frames after processing to prevent accumulation
                                #     speech_frames.clear()
                                #     logging.debug(f"Backend: Cleared speech_frames after streaming chunk to prevent accumulation")

                            elif in_speech_segment:
                                # Silence detected after speech, or speech ended
                                if (time.perf_counter() - last_speech_time) > SILENCE_TIMEOUT:
                                    if speech_frames:
                                        final_speech_segment_np = np.concatenate(speech_frames)
                                        # Sanitize final_speech_segment_np before pipeline
                                        if np.isnan(final_speech_segment_np).any() or np.isinf(final_speech_segment_np).any():
                                            logging.warning("Backend: Detected NaN or Inf in final_speech_segment_np. Sanitizing to 0.")
                                            final_speech_segment_np = np.nan_to_num(final_speech_segment_np, nan=0.0, posinf=0.0, neginf=0.0)
                                        logging.info(f"Backend: Final speech segment ENDED at {time.perf_counter():.3f}s. Duration: {len(final_speech_segment_np) / AUDIO_SAMPLE_RATE:.3f}s")
                                        await _process_speech_segment_pipeline(websocket, final_speech_segment_np, last_speech_time, is_final=True, session_config=session_config)
                                        speech_frames.clear()
                                    in_speech_segment = False
                                    # Ensure all buffers are cleared after a full speech segment ends
                                    audio_queue.clear()
                                    speech_frames.clear() # Redundant but safe
                                    pre_vad_buffer.clear()
                                    pre_vad_threshold_met = False
                            else:
                                # Not in speech segment and current frame is not speech, clear buffers
                                speech_frames.clear() # Redundant but safe
                    elif not session_config["vad_enabled"]: # VAD is disabled, process all incoming audio as one continuous segment
                        if len(audio_queue) >= streaming_chunk_samples:
                            chunk_to_process = np.array(list(audio_queue), dtype=np.float32)
                            # Sanitize chunk_to_process before pipeline
                            if np.isnan(chunk_to_process).any() or np.isinf(chunk_to_process).any():
                                logging.warning("Backend: Detected NaN or Inf in chunk_to_process (VAD disabled). Sanitizing to 0.")
                                chunk_to_process = np.nan_to_num(chunk_to_process, nan=0.0, posinf=0.0, neginf=0.0)
                            await _process_speech_segment_pipeline(websocket, chunk_to_process, time.perf_counter(), is_final=False, session_config=session_config)
                            audio_queue.clear()

    except WebSocketDisconnect:
        logging.info(f"Client {client_info} disconnected normally. Final speech_frames size: {len(speech_frames)}")
        if speech_frames:
            try:
                final_speech_segment_np = np.concatenate(speech_frames)
                await _process_speech_segment_pipeline(websocket, final_speech_segment_np, last_speech_time, is_final=True, session_config=session_config)
            except websockets.exceptions.ConnectionClosedOK:
                logging.info(f"Backend: WebSocket connection already closed for {client_info}. Skipping final segment processing.")
            except Exception as e:
                logging.warning(f"Backend: Error processing final speech segment on disconnect for {client_info}: {e}")
    except websockets.exceptions.ConnectionClosed as e: # Catch specific websockets connection closed errors
        logging.info(f"Backend: WebSocket connection closed unexpectedly for {client_info}: {e}")
        # No need to send message back, connection is already closed
    except Exception as e:
        logging.error(f"An unexpected error occurred in handle_audio_stream for {client_info}: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": f"Backend error: {e}"}))
        except websockets.exceptions.ConnectionClosedOK:
            logging.info(f"Backend: WebSocket connection already closed for {client_info}. Cannot send error message.")
        except Exception as send_error:
            logging.error(f"Backend: Error sending error message to client {client_info}: {send_error}")
    finally:
        logging.info(f"Client {client_info} handler finished. Cleaning up session.")
        
        # THESIS NOTE: Critical cleanup to prevent memory leaks and ensure proper resource management
        # 1. Remove session from active_sessions to prevent unbounded dictionary growth
        # 2. Explicitly delete model references to free GPU/CPU memory
        # 3. Clean up session lock to prevent lock dictionary growth
        
        if client_info in active_sessions:
            session_data = active_sessions.pop(client_info)
            
            # Explicitly delete model references to trigger garbage collection
            if session_data.get("stt_model"):
                logging.debug(f"Backend: Deleting STT model for session {client_info}")
                del session_data["stt_model"]
            
            if session_data.get("piper_tts_model"):
                logging.debug(f"Backend: Deleting Piper TTS model for session {client_info}")
                del session_data["piper_tts_model"]
                
            if session_data.get("coqui_tts_model"):
                logging.debug(f"Backend: Deleting Coqui TTS model for session {client_info}")
                del session_data["coqui_tts_model"]
            
            # Clean up all MT models
            mt_models = session_data.get("mt_models", {})
            for mt_key in list(mt_models.keys()):
                logging.debug(f"Backend: Deleting MT model {mt_key} for session {client_info}")
                del mt_models[mt_key]
            
            # Clean up VAD instance
            if session_data.get("vad_instance"):
                del session_data["vad_instance"]
            
            logging.info(f"Backend: Session {client_info} removed from active_sessions. Models unloaded.")
        else:
            logging.warning(f"Backend: Session {client_info} not found in active_sessions during cleanup.")
        
        # Clean up session lock
        if client_info in _session_locks:
            del _session_locks[client_info]
            logging.debug(f"Backend: Session lock removed for {client_info}")
        
        # Clean up audio buffers
        audio_queue.clear()
        speech_frames.clear()
        pre_vad_buffer.clear()
        
        logging.info(f"Backend: Cleanup complete for client {client_info}.")
