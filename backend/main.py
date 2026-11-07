import logging
import sys
import asyncio
import threading
import json
import numpy as np
import base64
import io
import soundfile as sf
import time
import os
from typing import Dict, Any, Optional, Tuple, List, Callable
from fastapi import FastAPI, Depends, Request, WebSocket, WebSocketDisconnect, APIRouter, UploadFile, File, Form, HTTPException # Import Depends, Request
from pydantic import BaseModel, Field # Import Field
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm # Import security components
import uuid

# Rate limiter import is optional: rate limiting only enabled via BP_ENABLE_RATE_LIMIT=1.
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    _SLOWAPI_AVAILABLE = True
except ImportError:
    Limiter = None
    get_remote_address = None
    _SLOWAPI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
# Keep a rotating copy on disk so a live UI test can be reviewed afterwards (BP_LOG_FILE=0 disables). Texts are already
# redacted by _log_text, so the file holds timings and lengths, not speech content.
if os.environ.get("BP_LOG_FILE", "1") != "0":
    try:
        from logging.handlers import RotatingFileHandler
        _log_path = os.environ.get("BP_LOG_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "bp-server.log"))
        os.makedirs(os.path.dirname(_log_path), exist_ok=True)
        _fh = RotatingFileHandler(_log_path, maxBytes=3_000_000, backupCount=3, encoding="utf-8")
        _fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(_fh)
    except OSError:
        pass  # read-only checkout: console logging only

# Initialize the database for the main application
from backend.utils.db_manager import SQLALCHEMY_DATABASE_URL, get_db_session_and_engine, init_db, User # Import init_db and User
from backend.utils.auth import get_password_hash, verify_password # Import auth functions
from backend.tts.piper_tts import PiperTTS
try:
    from backend.tts.coqui_tts import CoquiTTS
except ImportError:
    # SANDBOX-NOTE (2026-09-28): see backend/tts/base.py -- Coqui TTS is optional
    # on machines without it (no Windows wheels). Capabilities endpoint below is
    # None-safe via getattr; TTS_ENGINES simply lacks "xtts".
    CoquiTTS = None
# Lite install has no torch: these engines import it at module level, so guard like CoquiTTS.
from backend.tts.base import TTS_ENGINES, HybridTTS, OmniVoiceTTS
try:
    from backend.mt.nllb_mt import NLLBMT
except ImportError:
    NLLBMT = None

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
    username: str = Field(..., min_length=1, max_length=64)
    email: str = Field(..., max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(..., min_length=8, max_length=256)

class UserLogin(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str

from sqlalchemy.orm import Session
from backend.utils.db_manager import get_db # Import get_db

# Dependency to get the database session
def get_db_dependency():
    yield from get_db(SessionLocal)

# Security dependency for user authentication
security = HTTPBearer()

# Rate limiter (P1) -- protect auth & upload endpoints
# ONLY enable if explicitly requested via BP_ENABLE_RATE_LIMIT=1
from backend.utils.auth import rate_limit_enabled
if rate_limit_enabled() and Limiter is not None:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# Helper to conditionally apply rate limiting
def rate_limit(limit: str):
    """Conditionally apply rate limiting based on whether limiter is enabled."""
    if limiter is None:
        def noop_decorator(func):
            return func
        return noop_decorator
    return limiter.limit(limit)

# Engine registry (T010) -- selection + licence gate + instantiation
from backend.registry import instantiate, select, license_ok, engines as reg_engines, target_tts_engine

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db_dependency)):
    from backend.utils.auth import decode_access_token
    email = decode_access_token(credentials.credentials)
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Removed explicit FFmpeg environment variable settings as soundfile is now used for audio I/O
# and torchaudio is no longer a core dependency for the pipeline.
# os.environ["TORCH_USE_LIBAV"] = "0"
# os.environ["TORCH_USE_EXTERNAL_FFMPEG"] = "1"
# logging.info("Backend: TORCH_USE_LIBAV set to 0. TORCH_USE_EXTERNAL_FFMPEG set to 1.")

from fastapi.responses import FileResponse
from starlette.responses import JSONResponse
try:
    import torch  # optional: only for Tensor->numpy conversion of TTS output
except ImportError:
    torch = None
import webrtcvad # Re-enabled
from collections import deque
import websockets.exceptions # Explicitly import websockets.exceptions
import ffmpeg # Import ffmpeg for audio conversion
from starlette.websockets import WebSocketState # Import WebSocketState

from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend import prosody as _prosody
from backend.stt.streaming_captions import SessionCaptioner, STREAMING_CAPTIONS_AVAILABLE  # T044
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.piper_tts import PiperTTS
from backend.utils.audio_utils import load_audio, save_audio

# Router for FastAPI
router = APIRouter()

# Configuration
DEFAULT_STT_MODEL_SIZE = "base" # Default STT model size
DEFAULT_STT_ENGINE = os.environ.get("BP_STT_ENGINE", "faster_whisper")  # "mlx_whisper" on Apple Silicon (see backend/stt/mlx_whisper_stt.py)


def _effective_stt_engine(config: Dict[str, Any]) -> str:
    """Engine for this session. mlx_whisper has no Slovak fine-tune, so a Slovak source with BP_SK_STT_MODEL set
    (e.g. the converted kinit/whisper-large-v3-turbo-sk, 6.5% vs 10.7% plain turbo on FLEURS-sk) goes to faster-whisper."""
    engine = config.get("stt_engine", DEFAULT_STT_ENGINE)
    if engine == "mlx_whisper" and str(config.get("source_lang", "")).startswith("sk") and os.environ.get("BP_SK_STT_MODEL"):
        return "faster_whisper"
    return engine
DEFAULT_TTS_MODEL = "piper"
AUDIO_SAMPLE_RATE = 16000 # Standard sample rate for VAD and STT (Reverted to 16000 Hz for VAD/STT compatibility)
MAX_VOICE_UPLOAD_BYTES = 50 * 1024 * 1024 # /voices/upload had no size cap - unbounded file.read() into memory

# Machine-aware device selection so the pipeline runs anywhere
# (Mac M1 Pro MPS, NVIDIA CUDA, CPU-only Windows, Kaggle T4).
# Override via env: BP_DEVICE=cpu|cuda|mps|auto,
# BP_COMPUTE_TYPE=int8|float16|float32.
# "auto" self-verifies: probes torch for CUDA/MPS, falls back to cpu.
def _resolve_device() -> str:
    # faster-whisper uses CTranslate2, not Torch's MPS backend.
    requested = os.environ.get("BP_DEVICE", "auto").lower()
    if requested not in ("auto", "cpu", "cuda"):
        logging.warning("Unsupported CTranslate2 device %s; using CPU", requested)
        return "cpu"
    if requested == "cpu":
        return "cpu"
    try:
        import ctranslate2
        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda_available = False
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("BP_DEVICE=cuda requested but CTranslate2 cannot find a CUDA device")
    return "cuda" if cuda_available else "cpu"


BP_DEVICE = _resolve_device()
BP_COMPUTE_TYPE = os.environ.get("BP_COMPUTE_TYPE", "int8")
# CTranslate2 MT is CPU/CUDA only (no MPS); default cpu unless explicitly set.
BP_MT_DEVICE = os.environ.get("BP_MT_DEVICE", "cpu")


# Slovak-fine-tuned Whisper small (NaiveNeuron/whisper-small-sk, MIT) converted by scripts/convert_models.py.
SK_STT_LOCAL_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ct2_models", "whisper-small-sk")


def _pick_stt_model(source_lang: str, requested: str) -> str:
    """Pick the STT model for a given source language.

    SK source (SK→EN): whisper-small-sk is NOT a safe default — it produces
    Czech-like hallucinations (WER 0.25–0.32 > 0.2 quality gate, measured on
    bench/results 2026-11-02). Require an explicit quality tier via the
    BP_SK_STT_MODEL env var (e.g. kinit/whisper-large-v3-turbo-sk, ~6–13%
    WER per SloPal 2026, ~800 MB model). Falls back to whisper-small-sk only
    when the tier is not set, and the caller marks the direction experimental.

    EN source (EN→SK): requested size (base/small) is fine — English STT is
    accurate and fast.

    Order: BP_SK_STT_MODEL env var (model name or CT2 directory) >
           per-pack requested model > local fine-tuned small > small-sk.
    Set BP_SK_STT_MODEL=kinit/whisper-large-v3-turbo-sk for the quality tier.
    Faster-whisper downloads HF model names on first use; CT2 dirs must be
    converted with scripts/convert_models.py.
    """
    if source_lang.startswith("sk") and requested in ("base", "small", DEFAULT_STT_MODEL_SIZE):
        env = os.environ.get("BP_SK_STT_MODEL")
        if env:
            return env
        if os.path.exists(os.path.join(SK_STT_LOCAL_MODEL, "model.bin")):
            return SK_STT_LOCAL_MODEL
        return "small-sk"
    return requested

# VAD Configuration (matching BP xtts)
VAD_FRAME_DURATION = 20 # ms - BP xtts uses 20ms frames
VAD_AGGRESSIVENESS = 2 # Mode 2 (Balanced) - sensitive enough for quiet speech, still rejects noise
# Max open-segment bound (measured 2026-09-29): dense pause-free speech never trips
# SILENCE_TIMEOUT, so transcripts waited for teardown (~24s). Force-close every
# MAX_SEGMENT_S of continuous speech; next voiced frame opens a chained segment.
# This bounds the worst case, it is not the streaming mechanism (pauses + interim
# path give the real-time feel). 6s keeps typical sentences whole (MT quality)
# while capping dense-speech wait at ~7s incl. STT.
MAX_SEGMENT_S = 3.0
# MIN_SPEECH_DURATION removed - BP xtts doesn't filter by duration
SILENCE_TIMEOUT = 0.25 # seconds (reduced from 1.0s for better responsiveness)
# STT window for streaming chunk processing. Whisper needs sentence-level
# context (>= ~1.5s) for accurate transcription; 0.5s chunks produce
# fragmented, low-accuracy output (the "word at a time" effect). 2.0s keeps
# near-real-time feel while giving Whisper enough context for coherent phrases.
# VAD itself runs on 20ms frames; this window only gates STT input.
STREAMING_CHUNK_LENGTH = float(os.environ.get("BP_STREAMING_CHUNK_S", "2.0"))

# Pre-VAD Silence Detection Configuration
# Threshold calibrated based on user testing:
# - Finger clicks: ~0.015 (should block)
# - User speech: ~0.05-0.06 (should allow)
# - Background noise: ~0.002 (should block)
SILENCE_RMS_THRESHOLD = float(os.environ.get("BP_SILENCE_RMS", "0.005"))  # RMS pre-filter: blocks background noise from triggering VAD/STT false positives
PRE_VAD_BUFFER_DURATION = 0.5 # seconds, how much audio to buffer before checking RMS

# Speaker Voices Directory and Metadata
SPEAKER_VOICES_DIR = "speaker_voices"
SPEAKER_VOICES_METADATA_FILE = os.path.join(SPEAKER_VOICES_DIR, "speaker_voices.json")
os.makedirs(SPEAKER_VOICES_DIR, exist_ok=True)


def _safe_voice_name(name: str) -> str:
    """Same character policy as /voices/upload; rejects names that are empty or only dots."""
    cleaned = "".join(c for c in name if c.isalnum() or c in (" ", ".", "_")).strip()
    if not cleaned or not cleaned.strip("."):
        raise HTTPException(status_code=400, detail="Invalid voice name.")
    return cleaned


def _voice_path(filename: str) -> str:
    """Resolve `filename` inside SPEAKER_VOICES_DIR or raise 400 (blocks ../, backslash traversal and absolute paths)."""
    if "\\" in filename:  # not a separator on POSIX, but never produced by _safe_voice_name and is one on Windows
        raise HTTPException(status_code=400, detail="Invalid voice path.")
    base = os.path.realpath(SPEAKER_VOICES_DIR)
    path = os.path.realpath(os.path.join(base, filename))
    try:
        inside = os.path.commonpath([base, path]) == base
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise HTTPException(status_code=400, detail="Invalid voice path.")
    return path


def _safe_speaker_wav(path: Optional[str]) -> Optional[str]:
    """speaker_wav_path comes from the unauthenticated /initialize endpoint and the WebSocket.
    Only files inside SPEAKER_VOICES_DIR are allowed; anything else is dropped (no cloning, no file read)."""
    if not path:
        return path
    base = os.path.realpath(SPEAKER_VOICES_DIR)
    resolved = os.path.realpath(path)
    try:
        inside = os.path.commonpath([base, resolved]) == base
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        logging.warning("Backend: Ignoring speaker_wav_path outside the speaker voices directory.")
        return None
    return path

# Dictionary to store active sessions and their models
# Key: client_info (str), Value: Dict[str, Any] containing session_config and model instances
active_sessions: Dict[str, Dict[str, Any]] = {}

# Dictionary to track ongoing model conversions
_ongoing_conversions: Dict[str, asyncio.Lock] = {}

# Dictionary for per-session locks (prevents race conditions)
_session_locks: Dict[str, asyncio.Lock] = {}

# ===== Model Pool with Bounds (P0) =============================================
# One instance per (stage, engine, config) per process, shared across sessions.
# Configurable limits per stage; LRU eviction when limits exceeded.
MODEL_POOL_CONFIG = {
    "stt": {"max_instances": 3, "idle_ttl_s": 300},      # 3 STT models max, 5 min idle TTL
    "mt": {"max_instances": 5, "idle_ttl_s": 300},       # 5 MT models max
    "tts": {"max_instances": 5, "idle_ttl_s": 300},      # 5 TTS models max
    "vad": {"max_instances": 2, "idle_ttl_s": 600},      # VAD is tiny, rarely pooled
}

# Model pool: key=(stage, engine, config_hash) -> {"instance": obj, "ref_count": int, "last_used": float, "created_at": float}
_model_pool: Dict[tuple, Dict[str, Any]] = {}
_model_pool_locks: Dict[tuple, asyncio.Lock] = {}

def _pool_key(stage: str, engine: str, **config) -> tuple:
    """Stable key for a model instance based on stage, engine, and config."""
    import hashlib
    import json
    cfg_str = json.dumps(config, sort_keys=True)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    return (stage, engine, cfg_hash)

async def get_pooled_model(stage: str, engine: str, factory: Callable, **config) -> tuple:
    """
    Get or create a pooled model instance.
    Returns (model_instance, is_new) where is_new=True if freshly created.
    """
    key = _pool_key(stage, engine, **config)
    
    # Get or create lock for this key
    if key not in _model_pool_locks:
        _model_pool_locks[key] = asyncio.Lock()
    
    async with _model_pool_locks[key]:
        if key in _model_pool:
            # Existing instance - increment ref count
            _model_pool[key]["ref_count"] += 1
            _model_pool[key]["last_used"] = time.time()
            return _model_pool[key]["instance"], False
        
        # Check if we need to evict (per-stage limit)
        stage_limit = MODEL_POOL_CONFIG.get(stage, {}).get("max_instances", 10)
        current_stage_count = sum(1 for k in _model_pool if k[0] == stage)
        
        if current_stage_count >= stage_limit:
            # LRU eviction: find oldest unused model in this stage
            candidates = [(k, v) for k, v in _model_pool.items() if k[0] == stage and v["ref_count"] == 0]
            if candidates:
                oldest = min(candidates, key=lambda x: x[1]["last_used"])
                evict_key = oldest[0]
                logging.info(f"Backend: Evicting LRU model {evict_key} (stage={stage}, limit={stage_limit})")
                del _model_pool[evict_key]
            else:
                logging.warning(f"Backend: Model pool limit reached for {stage} ({stage_limit}), but all instances in use")
        
        # Create new instance
        logging.info(f"Backend: Creating new pooled {stage}/{engine} instance (key={key})")
        model = factory()
        _model_pool[key] = {
            "instance": model,
            "ref_count": 1,
            "last_used": time.time(),
            "created_at": time.time(),
            "stage": stage,
            "engine": engine,
            "config": config
        }
        return model, True


def release_pooled_model(stage: str, engine: str, **config) -> bool:
    """
    Decrement ref count; delete if zero and idle TTL expired.
    Returns True if model was actually released (ref_count hit 0 and TTL expired).
    """
    key = _pool_key(stage, engine, **config)
    if key not in _model_pool:
        return False
    
    _model_pool[key]["ref_count"] -= 1
    
    if _model_pool[key]["ref_count"] <= 0:
        # Check idle TTL against last_used (do not reset it here — that would
        # make the TTL check always ~0 and the delete dead code).
        idle_ttl = MODEL_POOL_CONFIG.get(stage, {}).get("idle_ttl_s", 300)
        if time.time() - _model_pool[key]["last_used"] >= idle_ttl:
            del _model_pool[key]
            if key in _model_pool_locks:
                del _model_pool_locks[key]
            logging.info(f"Backend: Released pooled {stage}/{engine} (key={key}) after idle TTL")
            return True
    return False


def cleanup_session_models(client_info: str):
    """Called when a session disconnects - releases its model references."""
    session_data = active_sessions.get(client_info)
    if not session_data:
        return
    
    config = session_data.get("session_config", {})
    
    # Release STT
    if session_data.get("stt_model"):
        stt_size = config.get("stt_model_size", "base")
        release_pooled_model("stt", _effective_stt_engine(config), **session_data.get("stt_pool_config", {"model_size": stt_size}))
    
    # Release MT models
    for mt_key, mt_model in session_data.get("mt_models", {}).items():
        if mt_model:
            src, tgt = mt_key.split("-", 1)
            release_pooled_model("mt", config.get("mt_engine", "opus-mt"), source=src, target=tgt, device=BP_MT_DEVICE)
    
    # Release TTS
    if session_data.get("tts_engine"):
        tts_choice = config.get("tts_model_choice", "piper")
        release_pooled_model("tts", tts_choice)
    
    # VAD is tiny, not pooled
    
    logging.info(f"Backend: Session {client_info} model refs released")


def get_model_pool_stats() -> Dict[str, Any]:
    """Return current pool stats for /metrics."""
    return {
        "total_instances": len(_model_pool),
        "by_stage": {
            stage: sum(1 for k in _model_pool if k[0] == stage)
            for stage in ("vad", "stt", "mt", "tts", "unified")
        },
        "details": [
            {
                "stage": k[0], "engine": k[1], "config_hash": k[2],
                "ref_count": v["ref_count"],
                "age_s": round(time.time() - v["created_at"], 1),
                "idle_s": round(time.time() - v["last_used"], 1),
            }
            for k, v in _model_pool.items()
        ]
    }


def _cleanup_idle_models():
    """Background task to clean up idle models. Called periodically."""
    now = time.time()
    for key, info in list(_model_pool.items()):
        stage = key[0]
        idle_ttl = MODEL_POOL_CONFIG.get(stage, {}).get("idle_ttl_s", 300)
        if info["ref_count"] == 0 and (now - info["last_used"]) >= idle_ttl:
            logging.info(f"Backend: Cleaning up idle model {key} (stage={stage})")
            del _model_pool[key]
            if key in _model_pool_locks:
                del _model_pool_locks[key]


# Available MT models for hybrid system
AVAILABLE_MT_MODELS = {
    "opus_mt": "CTranslate2 optimized Opus-MT models",
    "nllb": "NLLB-200 model for low-resource language pairs"
}

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
def authorize_speaker_wav(path: Optional[str], user_id: int) -> Optional[str]:
    """Client-selected voice references require ownership, not just a safe directory."""
    if not path:
        return None
    safe = _safe_speaker_wav(path)
    if not safe:
        raise HTTPException(status_code=403, detail="Voice reference is not available")
    name = os.path.basename(safe)
    entry = next((v for v in _read_speaker_voices_metadata()
                  if (v.get("filename") or os.path.basename(v.get("path", ""))) == name), None)
    if not entry or not (entry.get("public_demo") is True or entry.get("user_id") == user_id):
        raise HTTPException(status_code=403, detail="Voice reference is not owned by this user")
    return safe


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

# --- Model Initialization Functions ---
def local_stt_source(model: str) -> str:
    """Resolve STT model: local CT2 dir first, then HF name (for BP_SK_STT_MODEL quality tier)."""
    from faster_whisper.utils import download_model
    candidates = (model, os.path.join("ct2_models", f"whisper-{model}"))
    for directory in candidates:
        if os.path.isfile(os.path.join(directory, "model.bin")):
            return os.path.abspath(directory)
    # Not a local CT2 dir — treat as a HuggingFace model name (e.g. kinit/whisper-large-v3-turbo-sk)
    try:
        return download_model(model, local_files_only=False)
    except Exception:
        raise HTTPException(status_code=409, detail=f"STT assets missing for {model}. Run approved setup first; no runtime download was started.")


async def _initialize_stt_model(session_data: Dict[str, Any], stt_model_size: str = DEFAULT_STT_MODEL_SIZE):
    """Initializes the STT model for a given session via the engine registry.
    Engine name comes from session_config['stt_engine'] or defaults to 'faster_whisper'."""
    config = session_data.get("session_config", {})
    stt_engine = _effective_stt_engine(config)
    if stt_engine == "faster_whisper":
        stt_model_size = local_stt_source(stt_model_size)
        config["stt_model_size"] = stt_model_size
    beam = int(config.get("beam_size", os.environ.get("BP_STT_BEAM_SIZE", "5")))
    pool_config = {"model_size": stt_model_size, "device": config.get("stt_device", BP_DEVICE),
                   "compute_type": config.get("stt_compute_type", BP_COMPUTE_TYPE), "beam_size": beam}
    current_stt_model = session_data.get("stt_model")
    if current_stt_model is None or session_data.get("stt_pool_config") != pool_config:
        init_start = time.perf_counter()
        logging.info(f"Backend: Session {session_data['client_info']}: Getting pooled {stt_engine} STT model '{stt_model_size}' at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
        def stt_factory():
            return instantiate("stt", stt_engine, **pool_config)
        if session_data.get("stt_pool_config"):
            release_pooled_model("stt", stt_engine, **session_data["stt_pool_config"])
        session_data["stt_model"], is_new = await get_pooled_model(
            "stt", stt_engine, stt_factory, **pool_config)
        session_data["stt_pool_config"] = pool_config
        init_end = time.perf_counter()
        logging.info(f"Backend: Session {session_data['client_info']}: STT {'created' if is_new else 'reused'} at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s")
    else:
        logging.info(f"Backend: Session {session_data['client_info']}: {stt_engine} STT model '{stt_model_size}' already initialized.")

async def _initialize_captioner(session_data: Dict[str, Any], source_lang: str):
    """T044: lazily create this session's SessionCaptioner (streaming partial-text captions,
    additive only — see backend/stt/streaming_captions.py). Best-effort: any failure here
    (missing dep, model load error) disables captions for this session but must never block or
    break the real STT/MT/TTS pipeline, which doesn't depend on this at all."""
    if session_data.get("captioner") is not None:
        return
    if not STREAMING_CAPTIONS_AVAILABLE or os.environ.get("BP_ENABLE_STREAMING_CAPTIONS") != "1":
        return
    try:
        base_model_dir = local_stt_source("base")
        init_start = time.perf_counter()
        session_data["captioner"] = SessionCaptioner(source_lang=source_lang, model_dir=base_model_dir)
        logging.info(f"Backend: Session {session_data['client_info']}: SessionCaptioner initialized in {time.perf_counter() - init_start:.2f}s.")
    except Exception as e:
        logging.error(f"Backend: Session {session_data['client_info']}: SessionCaptioner init failed, captions disabled for this session: {e}")
        session_data["captioner"] = None

async def _initialize_mt_model(session_data: Dict[str, Any], source_lang: str, target_lang: str, websocket: Optional[WebSocket] = None):
    """Initializes a specific CTranslate2MT model for a given session via the engine registry."""
    mt_models_for_session = session_data.setdefault("mt_models", {})
    model_key = f"{source_lang}-{target_lang}"

    if source_lang == target_lang:
        logging.info(f"Backend: Session {session_data['client_info']}: Skipping MT model initialization for {model_key} as source and target languages are the same.")
        if model_key in mt_models_for_session:
            del mt_models_for_session[model_key]
        return

    mt_model_name = get_mt_model_name(source_lang, target_lang)
    mt_model_path = f"ct2_models/{mt_model_name.replace('/', '--')}"

    if model_key not in mt_models_for_session or getattr(mt_models_for_session[model_key], "ctranslate2_model_dir", None) != mt_model_path:
        # Acquire a lock for this model key to prevent multiple simultaneous conversions
        if model_key not in _ongoing_conversions:
            _ongoing_conversions[model_key] = asyncio.Lock()
        
        async with _ongoing_conversions[model_key]:
            # Re-check if model is already initialized or converted after acquiring lock
            if model_key in mt_models_for_session and getattr(mt_models_for_session[model_key], "ctranslate2_model_dir", None) == mt_model_path:
                logging.info(f"Backend: Session {session_data['client_info']}: CTranslate2MT for {model_key} already initialized by another task.")
                return
            if not all(os.path.isfile(os.path.join(mt_model_path, name)) for name in ("model.bin", "tokenizer_config.json")):
                raise HTTPException(status_code=409, detail=f"MT assets missing for {model_key}. Run the approved setup/conversion before initializing; no runtime download was started.")

            init_start = time.time()
            logging.info(f"Backend: Session {session_data['client_info']}: Getting pooled MT for {model_key} at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
            try:
                # Licence gate: allow_noncommercial from session config (default False)
                config = session_data.get("session_config", {})
                mt_engine = config.get("mt_engine", "opus-mt")
                allow_nc = config.get("allow_noncommercial", False)
                allow_unk = config.get("allow_unknown_license", False)
                
                def mt_factory():
                    return instantiate("mt", mt_engine, model_path=mt_model_path, device=BP_MT_DEVICE,
                                      allow_noncommercial=allow_nc, allow_unknown=allow_unk)
                
                mt_models_for_session[model_key], is_new = await get_pooled_model(
                    "mt", mt_engine, mt_factory, source=source_lang, target=target_lang, device=BP_MT_DEVICE
                )
                init_end = time.time()
                logging.info(f"Backend: Session {session_data['client_info']}: MT {'created' if is_new else 'reused'} on CPU at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s")
            except PermissionError as pe:
                # Licence gate blocked it — log and raise with context
                logging.error(f"Backend: Session {session_data['client_info']}: MT licence gate blocked '{model_key}': {pe}")
                raise HTTPException(status_code=403, detail=str(pe))
            except Exception as e:
                logging.error(f"Backend: Session {session_data['client_info']}: ERROR: Failed to initialize CTranslate2MT for {model_key}: {e}")
                if model_key in mt_models_for_session:
                    del mt_models_for_session[model_key]
                raise HTTPException(status_code=500, detail=f"Failed to initialize CTranslate2MT for {model_key} after conversion: {e}")
    else:
        logging.info(f"Backend: Session {session_data['client_info']}: CTranslate2MT for {model_key} already initialized.")

async def _initialize_tts_models(session_data: Dict[str, Any], tts_model_choice: str, speaker_wav_path: Optional[str], speaker_text: Optional[str], speaker_lang: Optional[str]):
    """Initializes the selected TTS engine for a session via the TTS_ENGINES registry
    (backend/tts/base.py) — to add/remove an engine, edit that registry, not this function."""
    target = (session_data.get("session_config") or {}).get("target_lang", "")
    tts_model_choice = target_tts_engine(tts_model_choice, target)
    session_data["session_config"]["tts_model_choice"] = tts_model_choice
    # Licence gate flags from session config
    allow_nc = session_data.get("session_config", {}).get("allow_noncommercial", False)
    allow_unk = session_data.get("session_config", {}).get("allow_unknown_license", False)

    try:
        # Use the central registry — it validates availability + licence
        factory = lambda: instantiate("tts", tts_model_choice,
                                       allow_noncommercial=allow_nc, allow_unknown=allow_unk)
    except PermissionError as pe:
        logging.warning(f"Backend: Session {session_data['client_info']}: TTS licence gate blocked '{tts_model_choice}': {pe}")
        session_data["tts_engine"] = None
        session_data["tts_engine_name"] = None
        return
    except KeyError:
        logging.warning(f"Backend: Session {session_data['client_info']}: Skipping TTS initialization as '{tts_model_choice}' is unknown.")
        session_data["tts_engine"] = None
        session_data["tts_engine_name"] = None
        return
    except RuntimeError as re:
        # Engine not installed here (e.g., XTTS without torch, OmniVoice without GPU deps)
        logging.warning(f"Backend: Session {session_data['client_info']}: TTS engine '{tts_model_choice}' not available here: {re}")
        session_data["tts_engine"] = None
        session_data["tts_engine_name"] = None
        return

    if session_data.get("tts_engine") is not None and session_data.get("tts_engine_name") == tts_model_choice:
        logging.info(f"Backend: Session {session_data['client_info']}: TTS engine '{tts_model_choice}' already initialized.")
        return

    init_start = time.time()
    logging.info(f"Backend: Session {session_data['client_info']}: Getting pooled TTS engine '{tts_model_choice}' at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
    try:
        session_data["tts_engine"], is_new = await get_pooled_model(
            "tts", tts_model_choice, factory
        )
        session_data["tts_engine_name"] = tts_model_choice
        session_data["session_config"]["tts_model_choice"] = tts_model_choice
        init_end = time.time()
        logging.info(f"Backend: Session {session_data['client_info']}: TTS {'created' if is_new else 'reused'} at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s.")
    except Exception as e:
        logging.error(f"Backend: Session {session_data['client_info']}: ERROR: Failed to initialize TTS engine '{tts_model_choice}': {e}")
        session_data["tts_engine"] = None
        session_data["tts_engine_name"] = None
        raise

async def _initialize_vad_instance(session_data: Dict[str, Any]):
    """Initializes the WebRTC VAD instance for a given session via the engine registry."""
    current_vad_instance = session_data.get("vad_instance")
    if current_vad_instance is None:
        init_start = time.time()
        logging.info(f"Backend: Session {session_data['client_info']}: Getting pooled WebRTC VAD with aggressiveness={VAD_AGGRESSIVENESS} at {time.strftime('%H:%M:%S', time.localtime(init_start))}...")
        
        def vad_factory():
            vad_obj = instantiate("vad", "webrtcvad")
            if hasattr(vad_obj, "set_aggressiveness"):
                vad_obj.set_aggressiveness(VAD_AGGRESSIVENESS)
            return vad_obj
        
        session_data["vad_instance"], is_new = await get_pooled_model("vad", "webrtcvad", vad_factory)
        # Override aggressiveness if the pooled instance was reused
        if hasattr(session_data["vad_instance"], "set_aggressiveness"):
            session_data["vad_instance"].set_aggressiveness(VAD_AGGRESSIVENESS)
        init_end = time.time()
        logging.info(f"Backend: Session {session_data['client_info']}: VAD {'created' if is_new else 'reused'} at {time.strftime('%H:%M:%S', time.localtime(init_end))}. Duration: {init_end - init_start:.2f}s")
    else:
        logging.info(f"Backend: Session {session_data['client_info']}: WebRTC VAD already initialized.")


async def initialize_all_models(client_info: str, source_lang: str, target_lang: str, tts_model_choice: str, stt_model_size: str = DEFAULT_STT_MODEL_SIZE, speaker_wav_path: Optional[str] = None, speaker_text: Optional[str] = None, speaker_lang: Optional[str] = None, vad_enabled_param: bool = True, websocket: Optional[WebSocket] = None):
    """Initializes all necessary models (STT, MT, TTS, VAD) for a given session."""
    speaker_wav_path = _safe_speaker_wav(speaker_wav_path)  # shared choke point: HTTP /initialize and WebSocket

    # --- Resolve language pack for per-language VAD/timing config ---
    pack = None
    try:
        from scripts.language_packs import load as _load_pack
        pack = _load_pack(f"{source_lang}-{target_lang}")   # the pack file itself: carries the `vad` block
    except Exception:
        pack = None  # no pack for this pair: global defaults below

    from backend.latency_profiles import resolve_profile
    pack, profile_name = resolve_profile(pack or {}, os.environ.get("BP_PROFILE", "default"))
    profile_stt = pack.get("stt", {}) if profile_name != "default" else {}
    stt_model_size = profile_stt.get("model", stt_model_size)
    pack_vad = pack.get("vad", {})
    # Precedence: env override > language pack > module default. The session usually exists already
    # (handle_audio_stream creates it first), so these must also be applied in the .update() below.
    vad_silence_s = float(os.environ.get("BP_VAD_SILENCE_S") or pack_vad.get("silence_timeout_s", SILENCE_TIMEOUT))
    vad_max_segment_s = float(os.environ.get("BP_VAD_MAX_SEGMENT_S") or pack_vad.get("max_segment_s", MAX_SEGMENT_S))
    barge_in = os.environ.get("BP_BARGE_IN") == "1"
    # Incremental (simultaneous) mode: translate agreed words while the speaker is still talking.
    incremental = (os.environ.get("BP_INCREMENTAL") or str(pack_vad.get("incremental", ""))).lower() in ("1", "true")

    session_data = active_sessions.setdefault(client_info, {
        "client_info": client_info,
        "stt_model": None,
        "mt_models": {},
        "tts_engine": None,
        "tts_engine_name": None,
        "vad_instance": None,
        "active_pipeline_task": None,  # T043: see handle_audio_stream's session dict for why. This is the OTHER session-init path (app.py's /ws route calls initialize_all_models before handle_audio_stream, so its setdefault runs first) — same keys required here or they're silently missing.
        "model_call_lock": asyncio.Lock(),  # T046
        "captioner": None,  # T044
        "captioner_lock": asyncio.Lock(),  # T044
        "last_transcription_text": None,  # Last non-empty streaming transcription, promoted to final if the tail segment is too short
        "session_config": { # Store a copy of the config for easy access
            "source_lang": source_lang,
            "target_lang": target_lang,
            "tts_model_choice": tts_model_choice,
            "stt_model_size": stt_model_size,
            "speaker_wav_path": speaker_wav_path,
            "speaker_text": speaker_text,
            "speaker_lang": speaker_lang,
            "vad_enabled": vad_enabled_param,
            # Per-language VAD/timing config (German-style end-of-sentence gating for all languages)
            "vad_silence_timeout_s": vad_silence_s,
            "vad_max_segment_s": vad_max_segment_s,
            "barge_in": barge_in,
            "incremental": incremental,
            "vad_end_of_sentence_gating": pack_vad.get("end_of_sentence_gating", True),
            "language_pack": pack or {},
        }
    })

    session_data["session_config"].update({
        "source_lang": source_lang,
        "target_lang": target_lang,
        "tts_model_choice": tts_model_choice,
        "stt_model_size": stt_model_size,
        "speaker_wav_path": speaker_wav_path,
        "speaker_text": speaker_text,
        "speaker_lang": speaker_lang,
        "vad_enabled": vad_enabled_param,
        "vad_silence_timeout_s": vad_silence_s,
        "vad_max_segment_s": vad_max_segment_s,
        "vad_end_of_sentence_gating": pack_vad.get("end_of_sentence_gating", True),
        "barge_in": barge_in,
        "incremental": incremental,
        "language_pack": pack or {},
    })

    # Set and clear profile-only options even when a session already exists.
    config = session_data["session_config"]
    if config.get("latency_profile", "default") != "default":
        for key in ("stt_device", "stt_compute_type", "beam_size"):
            config.pop(key, None)
    config["latency_profile"] = profile_name
    for source, dest in (("device", "stt_device"), ("compute_type", "stt_compute_type"), ("beam_size", "beam_size")):
        if source in profile_stt:
            config[dest] = profile_stt[source]
    # Language pack engine takes precedence over hardware-scan default:
    # Parakeet for EN source, Whisper small-sk for SK/CS source.
    if "engine" in profile_stt:
        config["stt_engine"] = profile_stt["engine"]

    logging.info(f"Backend: Session {client_info}: Initializing models at {time.strftime('%H:%M:%S', time.localtime(time.time()))}...")

    upgraded = _pick_stt_model(source_lang, stt_model_size)
    if upgraded != stt_model_size:
        stt_model_size = upgraded
        session_data["session_config"]["stt_model_size"] = stt_model_size

    # A model override is not an evaluation receipt. Every route stays experimental
    # until its full current STT/MT/TTS path has matching reviewed evidence.
    session_data["session_config"]["stt_experimental_sk"] = source_lang.startswith("sk")
    session_data["route_status"] = {"pair": f"{source_lang}-{target_lang}",
        "status": "experimental", "reason": "Models may be ready; full live route quality and cloned-voice acceptance remain unverified."}
    await _initialize_stt_model(session_data, stt_model_size)
    await _initialize_mt_model(session_data, source_lang, target_lang, websocket)
    if source_lang != "en" and source_lang != target_lang: # Also initialize en-target for auto-detection fallback, but only if not already the target
        await _initialize_mt_model(session_data, "en", target_lang, websocket)
    await _initialize_tts_models(session_data, tts_model_choice, speaker_wav_path, speaker_text, speaker_lang)
    await _initialize_vad_instance(session_data)
    await _initialize_captioner(session_data, source_lang)  # T044, best-effort, never blocks the above

    logging.info(f"Backend: Session {client_info}: All models initialized (or attempted to initialize) at {time.strftime('%H:%M:%S', time.localtime(time.time()))}.")
    return {"status": "success", "message": "Models initialized.", "route_status": session_data["route_status"]}

@router.get("/routes", summary="Declared routes and installed assets; no model loading")
async def route_inventory():
    from scripts.language_packs import all_packs, resolve
    routes = []
    for pack in all_packs():
        assets = resolve(pack)
        stt_cfg = pack.get("stt", {})
        # local_stt_source only applies to faster_whisper; Parakeet models are downloaded
        # from HuggingFace at runtime by transformers and don't live under ct2_models/
        if stt_cfg.get("engine") == "faster_whisper":
            try:
                assets["stt"] = local_stt_source(stt_cfg.get("model", ""))
            except HTTPException as exc:
                assets["stt"] = None
                assets["errors"].append(exc.detail)
        else:
            assets["stt"] = stt_cfg.get("model")
        assets["resolved"] = not assets["errors"]
        routes.append({"pair": pack["pair"], "source": pack["source"], "target": pack["target"],
            "assets": assets, "route_status": {"pair": pack["pair"], "status": "experimental",
            "reason": "Full live route acceptance is pending."}})
    return {"routes": routes}


# --- Voice Management Endpoints ---
@router.post("/register", summary="Register a new user")
@rate_limit("5/minute")
async def register_user(request: Request, user: UserCreate, db: Session = Depends(get_db_dependency)):
    if user.email.strip().lower().endswith("@guest.local"):
        raise HTTPException(status_code=400, detail="That email address is reserved.")
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
@rate_limit("10/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db_dependency)):
    user = db.query(User).filter(User.email == form_data.username).first() # username field is used for email
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    from backend.utils.auth import create_access_token
    return {"access_token": create_access_token(user.email), "token_type": "bearer"}

@router.post("/login", summary="Login endpoint for frontend")
@rate_limit("10/minute")
async def login(request: Request, user_login: UserLogin, db: Session = Depends(get_db_dependency)):
    """
    Login endpoint that matches the frontend's expected format.
    Accepts JSON with email and password, returns token and user info.
    """
    user = db.query(User).filter(User.email == user_login.email).first()
    if not user or not verify_password(user_login.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    from backend.utils.auth import create_access_token
    return {
        "message": "Login successful",
        "token": create_access_token(user.email),
        "username": user.username,
        "email": user.email
    }

@router.get("/auth/config", summary="Public auth config (Google client id or null)")
async def auth_config():
    """Lets the frontend decide whether to render the real Google button.
    Inert when GOOGLE_CLIENT_ID is unset — password login keeps working."""
    return {"google_client_id": os.environ.get("GOOGLE_CLIENT_ID"), "guest_enabled": guest_enabled()}


GUEST_EMAIL = "guest@guest.local"


def guest_enabled() -> bool:
    """Guest sessions let anyone try the translator without an account. On by default; BP_GUEST=0 turns them off.
    Abuse limits: 20 guest tokens/minute per client, 2-hour tokens, at most BP_GUEST_MAX_SESSIONS live sockets."""
    flag = (os.environ.get("BP_GUEST") or "").strip().lower()
    return flag not in ("0", "false", "no")


def _log_text(text) -> str:
    """Spoken words are personal data: server logs carry lengths, not content, unless BP_LOG_SESSIONS=1."""
    if os.environ.get("BP_LOG_SESSIONS") == "1":
        return repr(text)
    return f"<{len(text or '')} chars>"


def _deny_guest(user: "User") -> None:
    """Guests may translate but never touch stored voices (recordings belong to real accounts)."""
    if user.email == GUEST_EMAIL:
        raise HTTPException(status_code=403, detail="Guest sessions cannot manage voices. Create an account to prepare a personal voice.")


@router.post("/guest", summary="Start a guest demo session (no account)")
@rate_limit("20/minute")
async def guest_session(request: Request, db: Session = Depends(get_db_dependency)):
    if not guest_enabled():
        raise HTTPException(status_code=403, detail="Guest sessions are disabled on this server.")
    user = db.query(User).filter(User.email == GUEST_EMAIL).first()
    if user is None:
        # Empty password hash: password login can never match (see verify_password), so this row is token-only.
        user = User(username="guest", email=GUEST_EMAIL, hashed_password="")
        db.add(user)
        db.commit()
        db.refresh(user)
    from backend.utils.auth import create_access_token
    return {"message": "Guest session", "token": create_access_token(user.email, hours=2), "username": "Guest", "email": user.email, "guest": True}

@router.post("/auth/google", summary="Login with a Google ID token")
@rate_limit("10/minute")
async def login_with_google(request: Request, body: GoogleAuthRequest, db: Session = Depends(get_db_dependency)):
    """Verifies a Google Identity Services ID token, links/creates the user by
    Google `sub`, and returns the same mock-token shape as /login so the
    frontend storage flow is unchanged."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    if not client_id:
        raise HTTPException(status_code=501, detail="Google login not configured. Set GOOGLE_CLIENT_ID (see documentation/supabase_auth_db_guide.md).")
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests
        claims = google_id_token.verify_oauth2_token(body.id_token, google_requests.Request(), client_id)
    except ImportError:
        raise HTTPException(status_code=501, detail="google-auth package not installed. Run: pip install -r requirements.txt")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google ID token")
    sub = claims.get("sub")
    email = claims.get("email")
    if not sub or not email:
        raise HTTPException(status_code=401, detail="Google token missing sub/email")
    # Linking by email below would let an unverified address take over an existing account.
    if claims.get("email_verified") is not True:
        raise HTTPException(status_code=401, detail="Google email is not verified")
    user = db.query(User).filter(User.google_sub == sub).first()
    if user is None:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            base = email.split("@")[0] or "user"
            username = base
            n = 1
            while db.query(User).filter(User.username == username).first():
                n += 1
                username = f"{base}{n}"
            user = User(username=username, email=email, hashed_password="", google_sub=sub)
            db.add(user)
            db.commit()
            db.refresh(user)
        elif user.hashed_password and not user.google_sub:
            # A password account created with an unverified address must not be taken over (or kept alive) by linking.
            raise HTTPException(status_code=409, detail="An account with this email already uses a password. Sign in with email and password.")
        else:
            user.google_sub = sub
            db.commit()
    from backend.utils.auth import create_access_token
    return {
        "message": "Login successful",
        "token": create_access_token(user.email),
        "username": user.username,
        "email": user.email
    }

@router.post("/voices/upload", summary="Upload a new speaker voice", response_model=Dict[str, str])
@rate_limit("10/minute")
async def upload_voice(
    request: Request,
    file: UploadFile = File(...),
    voice_name: str = Form(...),
    speaker_lang: str = Form(...), # Add speaker_lang parameter
    current_user: User = Depends(get_current_user) # Authenticated user
):
    """
    Uploads a new speaker voice file and saves its metadata.
    """
    _deny_guest(current_user)
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
        chunks, total = [], 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_VOICE_UPLOAD_BYTES:   # stop reading at the cap instead of buffering the whole body
                raise HTTPException(status_code=413, detail=f"Voice upload too large (limit {MAX_VOICE_UPLOAD_BYTES / 1_000_000:.0f} MB).")
            chunks.append(chunk)
        contents = b"".join(chunks)

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
                "-t", "120", # cap decoded length (seconds)
                "pipe:1" # Output to stdout
            ]

            # If the input is webm, specify the input format and codec
            if 'webm' in (file.content_type or ""):
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
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(input=input_audio_buffer.getvalue()), timeout=60)
            except asyncio.TimeoutError:
                process.kill()
                raise RuntimeError("FFmpeg conversion timed out")

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
            raise HTTPException(status_code=500, detail="Failed to process the uploaded audio. Use a clear WAV, MP3, M4A or WebM recording.") # type: ignore

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
    except HTTPException:
        # Don't let a deliberately-raised HTTPException (e.g. the 413 size-limit check above)
        # get swallowed and rewritten to a generic 500 by the except-Exception clause below.
        raise
    except Exception as e:
        logging.error(f"Backend: Error uploading voice '{voice_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload voice: {e}")

class ProcessSessionRequest(BaseModel):
    session: str
    speaker_lang: str
    items: List[Dict[str, str]]  # [{idx: "01", text: "..."}] in recording order


@router.post("/voices/process", summary="Segment an enrollment session into a training corpus")
async def process_session(
    body: ProcessSessionRequest,
    current_user: User = Depends(get_current_user),
):
    """Collects one enrollment session's clips, resamples to 22050 Hz mono,
    writes processed/enroll/<user>/<session>/ + speaker_voices.json, and
    reports READY FOR TRAINING APPROVAL. Nothing trains by itself (approved-first)."""
    import librosa

    metadata = _read_speaker_voices_metadata()
    session = "".join(c for c in body.session if c.isalnum() or c in ("-", "_")) or "take1"
    found, total_s = [], 0.0
    for it in body.items:
        idx = str(it.get("idx", "")).zfill(2)
        name = f"{session}_{idx}"
        entry = next((v for v in metadata
                      if v.get("user_id") == current_user.id and v.get("name") == name), None)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Clip not found: {name}")
        found.append((entry, it.get("text", "")))
    out_dir = os.path.join("processed", "enroll", str(current_user.id), session)
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for entry, text in found:
        wav, _ = librosa.load(entry["path"], sr=22050, mono=True)
        total_s += len(wav) / 22050
        base = os.path.basename(entry["path"])
        root, _ = os.path.splitext(base)
        out_wav = os.path.join(out_dir, root + "_22k.wav")
        sf.write(out_wav, wav, 22050)
        entries.append({"id": entry["name"], "language": body.speaker_lang,
                        "name": entry["name"], "path": os.path.basename(out_wav),
                        "transcript_source": "in-browser enrollment",
                        "transcribed_text": text})
    with open(os.path.join(out_dir, "speaker_voices.json"), "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    return {"status": "ready_for_approval", "clips": len(entries),
            "seconds": round(total_s, 1), "corpus_dir": out_dir}


@router.put("/voices/rename", summary="Rename an existing speaker voice", response_model=Dict[str, str])
async def rename_voice(
    request: RenameVoiceRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Renames an existing speaker voice file and updates its metadata.
    """
    _deny_guest(current_user)
    metadata = _read_speaker_voices_metadata()
    
    # 1. Find all voices with the requested name
    matching_voices = [v for v in metadata if v.get("name") == request.old_name]
    
    if not matching_voices:
        raise HTTPException(status_code=404, detail=f"Voice '{request.old_name}' not found.")

    # 2. Check if the user owns any of these voices
    user_voice = next((v for v in matching_voices if v.get("user_id") == current_user.id), None)

    if user_voice:
        # User owns the voice, proceed with rename
        # new_name becomes part of a file path: apply the same policy as /voices/upload (no separators, no ..).
        new_name = _safe_voice_name(request.new_name)
        if any(v.get("user_id") == current_user.id and v.get("name") == new_name for v in metadata):
            raise HTTPException(status_code=409, detail=f"Voice with name '{new_name}' already exists for this user.")

        old_filename = user_voice["filename"]
        file_extension = os.path.splitext(old_filename)[1]
        new_filename = f"{current_user.id}_{new_name}{file_extension}"

        old_file_path = _voice_path(old_filename)
        new_file_path = _voice_path(new_filename)

        try:
            os.rename(old_file_path, new_file_path)
            user_voice["name"] = new_name
            user_voice["filename"] = new_filename
            user_voice["path"] = new_file_path
            _write_speaker_voices_metadata(metadata)
            logging.info(f"Backend: Renamed voice from '{request.old_name}' to '{request.new_name}' for user {current_user.id}")
            return {"status": "success", "message": f"Voice '{request.old_name}' renamed to '{request.new_name}' successfully."}
        except FileNotFoundError:
            logging.error(f"Backend: File not found for voice '{request.old_name}' at {old_file_path}")
            raise HTTPException(status_code=404, detail=f"Voice file '{request.old_name}' not found.")
        except Exception as e:
            logging.error(f"Backend: Error renaming voice '{request.old_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to rename voice: {e}")
    else:
        # Voice exists but belongs to someone else (e.g., system default)
        raise HTTPException(status_code=403, detail="Sorry, but you don't have permission to edit this file, deal with it")

@router.delete("/voices/delete", summary="Delete a speaker voice", response_model=Dict[str, str])
async def delete_voice(
    request: DeleteVoiceRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Deletes a speaker voice file and removes its metadata.
    """
    _deny_guest(current_user)
    metadata = _read_speaker_voices_metadata()
    updated_metadata = []
    voice_deleted = False
    for voice in metadata:
        if voice.get("user_id") == current_user.id and voice.get("filename") == request.filename:
            file_path = _voice_path(request.filename)
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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db_dependency)
):
    """
    Retrieves a list of speaker voices.
    - If authenticated: returns default voices (user_id=null) + user's own voices
    - If not authenticated: returns only default voices (user_id=null)
    """
    metadata = _read_speaker_voices_metadata()
    
    # Try to get current user if credentials provided
    current_user = None
    if credentials:
        try:
            from backend.utils.auth import decode_access_token
            email = decode_access_token(credentials.credentials)
            current_user = db.query(User).filter(User.email == email).first()
        except Exception as e:
            logging.warning(f"Backend: Failed to authenticate user for voices endpoint: {e}")
    
    # Return voices based on authentication status
    if current_user:
        # Authenticated: return default voices + user's voices
        user_voices = [
            {"id": v.get("id", str(uuid.uuid4())), "name": v["name"], "filename": v.get("filename") or os.path.basename(v.get("path", "")) or "unknown_filename.wav", "path": v.get("path", "unknown_path"), "language": v.get("language", "unknown"), "upload_time": v.get("upload_time", 0)}
            for v in metadata if v.get("public_demo") is True or v.get("user_id") == current_user.id
        ]
        logging.info(f"Backend: Retrieved {len(user_voices)} voices for authenticated user {current_user.id}")
    else:
        # Unauthenticated: return only default voices
        user_voices = [
            {"id": v.get("id", str(uuid.uuid4())), "name": v["name"], "filename": v.get("filename") or os.path.basename(v.get("path", "")) or "unknown_filename.wav", "path": v.get("path", "unknown_path"), "language": v.get("language", "unknown"), "upload_time": v.get("upload_time", 0)}
            for v in metadata if v.get("public_demo") is True
        ]
        logging.info(f"Backend: Retrieved {len(user_voices)} default voices for unauthenticated user")
    
    return user_voices

@router.get("/voices/file", summary="Download a speaker voice file (preview/listen)")
async def get_voice_file(
    filename: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db_dependency)
):
    """Serve one voice file for in-browser preview. Same visibility as GET /voices
    (default voices + the caller's own); path traversal blocked by _voice_path."""
    metadata = _read_speaker_voices_metadata()
    current_user = None
    if credentials:
        try:
            from backend.utils.auth import decode_access_token
            email = decode_access_token(credentials.credentials)
            current_user = db.query(User).filter(User.email == email).first()
        except Exception:
            current_user = None
    entry = next((v for v in metadata if v.get("filename") == filename), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Voice not found.")
    if entry.get("public_demo") is not True and (current_user is None or entry.get("user_id") != current_user.id):
        raise HTTPException(status_code=403, detail="Not your voice.")
    path = _voice_path(filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Voice file missing.")
    ext = os.path.splitext(filename)[1].lower()
    media = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
             "ogg": "audio/ogg", "flac": "audio/flac"}.get(ext.lstrip("."), "audio/wav")
    return FileResponse(path, media_type=media, filename=filename)

EAR_GRADES_PATH = os.path.join("processed", "ear_grades.json")


class RatingIn(BaseModel):
    name: str
    ear: Dict[str, Any] = {}


def _read_ear_grades() -> Dict[str, Any]:
    try:
        with open(EAR_GRADES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"version": 1, "grades": []}


def _write_ear_grades(store: Dict[str, Any]) -> None:
    tmp = EAR_GRADES_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    os.replace(tmp, EAR_GRADES_PATH)


@router.get("/ratings", summary="Read ear-grade store (Voice Lab sync)")
async def get_ratings(current_user: User = Depends(get_current_user)):
    """Return processed/ear_grades.json so the Lab can merge server-side
    verdicts over browser localStorage. Authenticated like /voices/upload."""
    return _read_ear_grades()


@router.post("/ratings", summary="Save one ear-grade verdict (Voice Lab sync)")
async def post_rating(body: RatingIn, current_user: User = Depends(get_current_user)):
    """Merge one Lab verdict into processed/ear_grades.json. The item must
    already exist in the store (run scripts/grade_library.py first); the note
    is capped, defect values are validated. Returns the stored ear slot."""
    _deny_guest(current_user)
    store = _read_ear_grades()
    grades = store.get("grades", [])
    entry = next((g for g in grades if g.get("name") == body.name), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No graded item named '{body.name}'.")
    ear = body.ear if isinstance(body.ear, dict) else {}
    grade = ear.get("grade")
    if grade is not None and (not isinstance(grade, int) or not 0 <= grade <= 5):
        raise HTTPException(status_code=400, detail="grade must be an int 0-5.")
    keep = ear.get("keep")
    if keep is not None and keep not in (0, 1, 2, "keep", "kill"):
        raise HTTPException(status_code=400, detail="keep must be 0-2 or keep/kill.")
    defects = ear.get("defects") or {}
    if not isinstance(defects, dict) or any(
            k not in ("steadiness", "hiss", "muffled") or not isinstance(v, int)
            or not 0 <= v <= 4 for k, v in defects.items()):
        raise HTTPException(status_code=400, detail="defects must map steadiness/hiss/muffled to ints 0-4.")
    note = str(ear.get("note", ""))[:500]
    sim = ear.get("sim")
    if sim is not None and (not isinstance(sim, (int, float)) or not 0 <= sim <= 100):
        raise HTTPException(status_code=400, detail="sim must be 0-100.")
    slot = {"grade": grade if grade else None,
            "keep": ({1: "keep", 2: "kill"}.get(keep, keep) if isinstance(keep, int) else keep),
            "sim": sim, "defects": defects, "note": note, "updated": round(time.time(), 0)}
    old_updated = (entry.get("ear") or {}).get("updated", 0)
    if slot["updated"] >= old_updated:
        entry["ear"] = slot
        _write_ear_grades(store)
        logging.info(f"Backend: rating saved for '{body.name}' by user {current_user.id}")
    return {"name": body.name, "ear": entry["ear"]}

@router.get("/voice-lab/status", summary="Voice-lab system status (no auth, local eval page)")
async def voice_lab_status():
    """Read-only status for the local voice-lab eval page (ui/voice-lab/).

    Instantiates nothing — engine flags come from classes, hardware from
    backend/hardware.py detection, QC state from processed/voice_qc/ manifests.
    """
    from backend import hardware

    engine_classes = {
        "piper": PiperTTS,
        "piper_personal": PiperTTS,
        "piper_personal_v2": PiperTTS,
        "xtts": CoquiTTS,  # None when Coqui is uninstallable (no Windows wheels)
        "hybrid": HybridTTS,
        "omnivoice": OmniVoiceTTS,
    }
    # SANDBOX-FIX (2026-09-28): don't advertise engines whose class is None --
    # otherwise the UI offers XTTS on machines where selecting it can only fail.
    engine_classes = {k: v for k, v in engine_classes.items() if v is not None}
    engines = {
        key: {
            "cloning": bool(getattr(cls, "SUPPORTS_CLONING", False)),
            "streaming": bool(getattr(cls, "SUPPORTS_STREAMING", False)),
            "requires_speaker_wav": bool(getattr(cls, "REQUIRES_SPEAKER_WAV", False)),
        }
        for key, cls in engine_classes.items()
    }

    backends = {}
    for stage in ("stt", "mt", "tts_baseline", "tts_clone"):
        try:
            backends[stage] = hardware.detect_backend(stage)
        except Exception as e:
            backends[stage] = f"error: {e}"

    piper_dir = os.path.join("backend", "tts", "piper_models")
    try:
        piper_models = sorted(f[:-5] for f in os.listdir(piper_dir) if f.endswith(".onnx"))
    except OSError:
        piper_models = []

    qc_dir = os.path.join("processed", "voice_qc")
    qc_manifest, qc_scores = None, None
    try:
        with open(os.path.join(qc_dir, "candidates.json")) as f:
            qc_manifest = json.load(f)
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(qc_dir, "scores.json")) as f:
            qc_scores = json.load(f)
    except (OSError, ValueError):
        pass

    return {
        "engines": engines,
        "hardware_backends": backends,
        "piper_models": piper_models,
        "qc_candidates": (qc_manifest or {}).get("candidates", []),
        "qc_scores": (qc_scores or {}).get("results", []),
        "qc_scored": qc_scores is not None,
    }

@router.get("/health", summary="Health check — per-stage readiness, engine in use")
async def health_check():
    """Lightweight health check for load balancers / monitoring.
    Returns per-stage readiness, engine name + version, active sessions.
    No auth required — this is a public health endpoint."""
    from backend.registry import engines as reg_engines
    from backend.main import active_sessions
    
    # Check which engines are actually loaded in sessions
    loaded_engines = {}
    for session_data in active_sessions.values():
        if session_data.get("stt_model"):
            loaded_engines.setdefault("stt", set()).add("faster_whisper")
        for mt_key, mt_model in session_data.get("mt_models", {}).items():
            if mt_model:
                loaded_engines.setdefault("mt", set()).add("opus-mt")
        if session_data.get("tts_engine"):
            tts_name = session_data.get("tts_engine_name", "piper")
            loaded_engines.setdefault("tts", set()).add(tts_name)
        if session_data.get("vad_instance"):
            loaded_engines.setdefault("vad", set()).add("webrtcvad")
    
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "stages": {
            "stt": {"ready": "stt" in loaded_engines, "engines": list(loaded_engines.get("stt", []))},
            "mt": {"ready": "mt" in loaded_engines, "engines": list(loaded_engines.get("mt", []))},
            "tts": {"ready": "tts" in loaded_engines, "engines": list(loaded_engines.get("tts", []))},
            "vad": {"ready": "vad" in loaded_engines, "engines": list(loaded_engines.get("vad", []))},
        },
        "model_pool": get_model_pool_stats(),
        "timestamp": time.time(),
    }

@router.get("/metrics", summary="Detailed metrics — pool stats, session details, latency")
async def metrics(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Detailed metrics endpoint — requires auth.
    Returns model pool stats, per-session details, capacity."""
    from backend.utils.auth import decode_access_token
    from backend.utils.db_manager import get_db
    from sqlalchemy.orm import Session
    
    email = decode_access_token(credentials.credentials)
    db = next(get_db(SessionLocal))
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    pool_stats = get_model_pool_stats()
    
    # Per-session detail
    sessions_detail = []
    for client_info, session_data in active_sessions.items():
        config = session_data.get("session_config", {})
        sessions_detail.append({
            "client": client_info,
            "source_lang": config.get("source_lang"),
            "target_lang": config.get("target_lang"),
            "tts_engine": config.get("tts_model_choice"),
            "stt_model": config.get("stt_model_size"),
            "has_stt": session_data.get("stt_model") is not None,
            "has_mt": bool(session_data.get("mt_models")),
            "has_tts": session_data.get("tts_engine") is not None,
            "has_vad": session_data.get("vad_instance") is not None,
        })
    
    return {
        "model_pool": pool_stats,
        "active_sessions": len(active_sessions),
        "sessions": sessions_detail,
        "capacity": {
            "stt_max": MODEL_POOL_CONFIG["stt"]["max_instances"],
            "mt_max": MODEL_POOL_CONFIG["mt"]["max_instances"],
            "tts_max": MODEL_POOL_CONFIG["tts"]["max_instances"],
        },
        "timestamp": time.time(),
    }

def get_initialized_models(client_info: str, session_config: Dict[str, Any]) -> Tuple[Optional[FasterWhisperSTT], Optional[CTranslate2MT], Optional[Any], Optional[Any], str]:
    """Returns the currently initialized model instances and current TTS choice based on session config."""
    session_data = active_sessions.get(client_info)
    if not session_data:
        logging.warning(f"Backend: Session {client_info} not found in active_sessions. Cannot retrieve models.")
        return None, None, None, None, session_config["tts_model_choice"] # Return None for models if session not found

    stt_model_instance = session_data.get("stt_model")
    main_mt_model = session_data["mt_models"].get(f"{session_config['source_lang']}-{session_config['target_lang']}")
    vad_instance = session_data.get("vad_instance")
    
    tts_model_instance = session_data.get("tts_engine") if session_data.get("tts_engine_name") == session_config["tts_model_choice"] else None

    return stt_model_instance, main_mt_model, tts_model_instance, vad_instance, session_config["tts_model_choice"]

# --- Core Speech Processing Pipeline ---
_session_log_files: Dict[str, Any] = {}

def _log_session_event(client_info: str, event: Dict[str, Any]):
    """Append-only JSONL log per WebSocket session (transcripts, translations,
    captions, latencies) for later accuracy review. Never raises — logging must
    not break the pipeline. Files: processed/sessions/<stamp>_<client>.jsonl
    Opt-in (BP_LOG_SESSIONS=1): transcripts of meetings are personal data, so nothing is written by default."""
    if os.environ.get("BP_LOG_SESSIONS") != "1":
        return
    try:
        f = _session_log_files.get(client_info)
        if f is None or f.closed:
            safe = "".join(c if c.isalnum() else "_" for c in (client_info or "unknown"))
            os.makedirs(os.path.join("processed", "sessions"), exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            f = open(os.path.join("processed", "sessions", f"{stamp}_{safe}.jsonl"), "a", encoding="utf-8")  # SANDBOX-FIX (2026-09-28): explicit UTF-8 -- stock Windows consoles default to cp1252 and any Slovak diacritic (č/š/ž) killed the write
            _session_log_files[client_info] = f
        f.write(json.dumps({"ts": time.time(), **event}, ensure_ascii=False) + "\n")
        f.flush()
    except Exception as e:
        logging.warning(f"Backend: session log write failed (non-fatal): {e}")

def _queue_pre_vad_audio(audio, queue, pending, threshold_met):
    """Admit each sample once; a disabled RMS gate must not add latency."""
    if threshold_met or SILENCE_RMS_THRESHOLD <= 0:
        queue.extend(audio)
        return True
    pending.extend(audio)
    if len(pending) < int(AUDIO_SAMPLE_RATE * PRE_VAD_BUFFER_DURATION):
        return False
    buffered = np.nan_to_num(np.asarray(pending, dtype=np.float32),
                             nan=0.0, posinf=0.0, neginf=0.0)
    admitted = bool(np.sqrt(np.mean(buffered ** 2)) > SILENCE_RMS_THRESHOLD)
    if admitted:
        queue.extend(buffered)
    pending.clear()
    return admitted


def _stt_prompt(session_data) -> Optional[str]:
    """Pack vocabulary prompt + the tail of the previous transcription: fragments cut at pauses lose their left context,
    which is where "live demonstration" became "live to Messiah". OFF by default (BP_STT_CONTEXT=1 enables): the first live test with it on produced a repetition loop and a credits hallucination; unproven until A/B-tested."""
    base = session_data.get("language_pack", {}).get("stt", {}).get("initial_prompt") or ""
    tail = (session_data.get("last_transcription_text") or "")[-160:].strip() if os.environ.get("BP_STT_CONTEXT", "0") == "1" else ""
    return (f"{base} {tail}".strip()) or None


async def _process_speech_segment_pipeline(
    websocket: WebSocket, audio_segment_np: np.ndarray, segment_start_time: float, is_final: bool = True,
    session_config: Dict[str, Any] = None, # Add session_config parameter
    tts_seq: Optional[int] = None,  # emission order: TTS of segment N may not start before segment N-1 finished
    pretranscribed: Optional[str] = None,  # incremental mode: text already agreed by the streaming transcriber, skip STT
    source_span: Optional[tuple] = None,  # stream-clock (start_s, end_s) of the source words, reported to clients
    source_onset_time: Optional[float] = None  # server-clock first source sample of this utterance
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
    _log_session_event(client_info, {"type": "session_config", "dir": "in",
        "source_lang": source_lang, "target_lang": target_lang,
        "tts_model_choice": tts_model_choice, "speaker_wav_path": speaker_wav_path})
    session_data = active_sessions.get(client_info)
    if not session_data:
        logging.error(f"Backend: Session {client_info} not found in active_sessions. Cannot process speech segment.")
        # Do not send message if session is not found, as websocket might be closed
        return

    capture_generation = session_data.get("capture_generation", 0)

    first_audio_sent_time = None

    # Helper to safely send messages
    async def safe_send(message_type: str, payload: Dict[str, Any], is_bytes: bool = False):
        nonlocal first_audio_sent_time
        if (not session_data.get("capture_active", True)
                or session_data.get("capture_generation", 0) != capture_generation):
            return
        # Check if the WebSocket is still connected before sending
        if websocket.client_state == WebSocketState.DISCONNECTED:
            logging.debug(f"Backend: WebSocket for {websocket.scope['client'][0]}:{websocket.scope['client'][1]} is disconnected. Skipping send of {message_type}.")
            return
        try:
            if is_bytes:
                await websocket.send_bytes(payload)
                if first_audio_sent_time is None:
                    first_audio_sent_time = time.perf_counter()
                _log_session_event(client_info, {"type": message_type, "dir": "out",
                    "bytes": len(payload) if hasattr(payload, "__len__") else -1})
            else:
                await websocket.send_text(json.dumps(payload))
                if message_type != "tts_audio":
                    _log_session_event(client_info, {"dir": "out", **payload})
        except websockets.exceptions.ConnectionClosedOK:
            logging.info(f"Backend: WebSocket connection already closed for {websocket.scope['client'][0]}:{websocket.scope['client'][1]}. Skipping send of {message_type}.")
        except Exception as e:
            logging.error(f"Backend: Error sending {message_type} to client {websocket.scope['client'][0]}:{websocket.scope['client'][1]}: {e}")

    stt_model_instance = session_data.get("stt_model")
    mt_models_for_session = session_data.get("mt_models", {})
    tts_engine = session_data.get("tts_engine") if session_data.get("tts_engine_name") == tts_model_choice else None

    # T046: STT+MT run under the session's model-call lock, NOT cancelled by a new speech-start
    # (see barge-in trigger below) — these calls are cheap (sub-2s) and produce real text output
    # that a naive "cancel on any new speech" rule was found (live-tested) to silently drop for
    # ordinary back-to-back sentences in continuous speech, not just deliberate interruptions.
    # The lock also prevents two tasks from concurrently calling the same per-session model
    # instance from different executor threads, which create_task (T043) makes possible for the
    # first time in this codebase — untested for thread-safety, so serialize rather than assume.
    # Fragments from the incremental transcriber only need MT: a separate lock keeps them from queueing
    # behind the next incremental STT step (which shares the STT model and runs every ~0.7 s).
    pipeline_lock = session_data.setdefault("mt_call_lock", asyncio.Lock()) if pretranscribed is not None else session_data["model_call_lock"]
    async with pipeline_lock:
        # --- STT ---
        websocket.timestamps["stt_start"].append(time.perf_counter())
        stt_start_time = time.perf_counter()
        # Use the session-specific STT model
        if pretranscribed is not None:
            transcribed_segments, stt_time, detected_source_lang = [], 0.0, None
        else:
            # Short segments (< 1.5s): Whisper's VAD tends to kill them as silence.
            # Use vad_filter=False for short segments, True for long ones.
            segment_dur_s = audio_segment_np.size / AUDIO_SAMPLE_RATE
            use_vad_filter = segment_dur_s >= 1.5
            transcribed_segments, stt_time, detected_source_lang = await loop.run_in_executor(
                None,
                lambda: stt_model_instance.transcribe_audio(
                    audio_segment_np,
                    AUDIO_SAMPLE_RATE,
                    language=None if source_lang == "auto" else source_lang,
                    vad_filter=use_vad_filter,
                    initial_prompt=_stt_prompt(session_data),
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
        if pretranscribed is not None:
            transcribed_text = pretranscribed
        logging.info(f"Backend: Session {client_info}: STT completed. Transcribed: {_log_text(transcribed_text)} (Detected Lang: {detected_source_lang}). Time: {stt_total_time:.2f}s. Input->STT Latency: {input_to_stt_latency:.2f}s")

        # Filter out transcriptions that are just periods/punctuation (background noise artifacts)
        cleaned_text = transcribed_text.strip().replace('.', '').replace(',', '').replace('!', '').replace('?', '').strip()
        if not cleaned_text:
            logging.info(f"Backend: Session {client_info}: Transcription contains only punctuation (likely background noise). Skipping translation and TTS.")
            return

        # Track last non-empty transcription for promoting to final if the tail segment is too short
        session_data["last_transcription_text"] = transcribed_text

        if not transcribed_text.strip():
            logging.warning(f"Backend: Session {client_info}: STT produced no meaningful text for segment (Final: {is_final}). Skipping translation and TTS.")
            await safe_send("transcription_result", {"type": "transcription_result", "transcribed": "", "is_final": is_final, "metrics": {"stt_time": stt_total_time}})
            return
        await safe_send("transcription_result", {"type": "transcription_result", "transcribed": transcribed_text, "is_final": is_final, "metrics": {"stt_time": stt_total_time}, "source_span": list(source_span) if source_span else None})

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
            if pretranscribed is not None:   # already inside mt_call_lock
                translated_text, _ = await loop.run_in_executor(None, lambda: main_mt_model.translate(transcribed_text, source_lang, target_lang))
            else:   # one translator instance: never two executor threads at once
                async with session_data.setdefault("mt_call_lock", asyncio.Lock()):
                    translated_text, _ = await loop.run_in_executor(None, lambda: main_mt_model.translate(transcribed_text, source_lang, target_lang))
            mt_total_time = time.perf_counter() - mt_start_time
            websocket.timestamps["mt_end"].append(time.perf_counter())
        else:
            logging.info(f"Backend: Session {client_info}: Source and target languages are the same ({source_lang}). Skipping machine translation.")
        logging.info(f"Backend: Session {client_info}: MT completed for {source_lang}-{target_lang}. Translated: {_log_text(translated_text)}. Time: {mt_total_time:.2f}s")

        if not translated_text:
            logging.warning(f"Backend: Session {client_info}: MT produced no translated text for {_log_text(transcribed_text)} (Final: {is_final}). Skipping TTS.")
            await safe_send("translation_result", {"type": "translation_result", "translated": "", "is_final": is_final, "metrics": {"mt_time": mt_total_time}})
            return

        # Runaway guard: one VAD segment cannot carry more speech than ~22 chars per source second. A decoding loop upstream
        # ("1-1-1-...") once became 35 s of TTS and stalled the whole queue (2026-11-07 live replay).
        _src_s = audio_segment_np.size / AUDIO_SAMPLE_RATE
        if len(translated_text) > max(120, 22 * _src_s + 40):
            logging.warning(f"Backend: Session {client_info}: translation of {len(translated_text)} chars for {_src_s:.2f}s of audio looks like a decoding loop; dropping it before TTS.")
            return

        # Send translation result to frontend
        await safe_send("translation_result", {"type": "translation_result", "translated": translated_text, "is_final": is_final, "metrics": {"mt_time": mt_total_time}})
    # model_call_lock released here — TTS below runs unlocked, is the only phase barge-in cancels.

    # Emit in segment order: STT/MT are FIFO under model_call_lock, but TTS runs unlocked, so a short
    # later segment could otherwise interleave its audio ahead of a longer earlier one.
    if tts_seq:
        events = session_data.setdefault("tts_events", {})
        previous = events.setdefault(tts_seq - 1, asyncio.Event())
        try:
            await asyncio.wait_for(previous.wait(), timeout=20.0)   # a hung earlier segment must not mute the session
        except asyncio.TimeoutError:
            logging.warning(f"Backend: Session {client_info}: segment {tts_seq - 1} did not finish in 20 s; not waiting further.")
        events.pop(tts_seq - 1, None)

    # --- TTS ---
    # Standard non-streaming synthesis (Piper or fallback)
    logging.info(f"Backend: Session {client_info}: Starting TTS synthesis...")

    # Check for speaker_wav_path requirement for engines that have no generic-voice fallback
    # (declared per-engine via REQUIRES_SPEAKER_WAV in backend/tts/*.py, not hardcoded per name)
    if tts_engine is not None and getattr(tts_engine, "REQUIRES_SPEAKER_WAV", False) and not speaker_wav_path:
        logging.warning(f"Backend: Session {client_info}: '{tts_model_choice}' requires a speaker voice but none was provided. Sending error to client.")
        await safe_send("error", {"type": "error", "message": f"'{tts_model_choice}' requires a selected voice. Please choose or record one."})
        return

    # --- TTS ---
    websocket.timestamps["tts_start"].append(time.perf_counter())
    tts_start_time = time.perf_counter()

    audio_wav = None
    sample_rate = None
    tts_total_time = 0.0

    if tts_engine is None:
        logging.warning(f"Backend: Session {client_info}: Selected TTS model '{tts_model_choice}' not initialized or invalid. No TTS will be performed.")
        await safe_send("error", {"type": "error", "message": f"TTS model '{tts_model_choice}' not ready or invalid. No TTS output."})
        return

    # Some engines can't synthesize a requested language directly and proxy through a
    # supported one instead (e.g. XTTS has no native Slovak -> uses Czech). Declared per-engine
    # via LANGUAGE_OVERRIDES, not hardcoded here.
    tts_lang = getattr(tts_engine, "LANGUAGE_OVERRIDES", {}).get(target_lang, target_lang)
    if tts_lang != target_lang:
        logging.info(f"Backend: Session {client_info}: '{tts_model_choice}' using '{tts_lang}' as a proxy for '{target_lang}'.")

    if getattr(tts_engine, "SUPPORTS_STREAMING", False):
        logging.info(f"Backend: Session {client_info}: Starting streaming TTS synthesis with '{tts_model_choice}'...")
        queue = asyncio.Queue()

        prosody_on = _prosody.ENABLED and getattr(tts_engine, "SUPPORTS_PROSODY", False)
        sched = session_data.setdefault("prosody_sched", _prosody.PlayoutScheduler()) if prosody_on else None
        follower = session_data.setdefault("prosody_rate", _prosody.RateFollower()) if prosody_on else None
        prosody_kwargs = {"length_scale": follower.scale, "pauses": True} if prosody_on else {}
        stream_audio_s = 0.0

        def run_streaming_tts():
            try:
                # synthesize_stream yields numpy arrays
                for chunk in tts_engine.synthesize_stream(
                    text=translated_text,
                    language=tts_lang,
                    speaker_wav_path=speaker_wav_path,
                    **prosody_kwargs
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None) # Sentinel
            except Exception as e:
                logging.error(f"Backend: Error in '{tts_model_choice}' streaming thread: {e}")
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Run the blocking generator in a separate thread
        threading.Thread(target=run_streaming_tts, daemon=True).start()

        chunk_count = 0
        stream_sample_rate = getattr(tts_engine, "sample_rate", 24000)
        while True:
            chunk = await queue.get()
            if chunk is None:
                break

            chunk_count += 1
            stream_audio_s += len(chunk) / stream_sample_rate
            if prosody_on and chunk_count == 1:
                # Mirror the speaker's pause before this segment: pad silence so the onset-to-onset spacing is kept.
                # Reported onsets are None for most segments and span a whole utterance for finals, so they cannot drive the
                # scheduler (found 2026-11-07: 4 of 22 segments had one). Use when THIS audio started: dispatch time minus its length.
                chunk_onset = segment_start_time - audio_segment_np.size / AUDIO_SAMPLE_RATE
                lead_s = sched.lead_silence(chunk_onset, time.perf_counter())
                if lead_s > 0:
                    chunk = np.concatenate([np.zeros(int(stream_sample_rate * lead_s), dtype=np.asarray(chunk).dtype), np.asarray(chunk)])
                    logging.info(f"Backend: Session {client_info}: prosody lead silence {lead_s:.2f}s")
            # Convert numpy chunk to WAV bytes
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, chunk, stream_sample_rate, format="WAV")
            await safe_send("tts_audio", audio_buffer.getvalue(), is_bytes=True)

        if prosody_on and chunk_count:
            sched.committed(stream_audio_s)
            scale = follower.update(audio_segment_np.size / AUDIO_SAMPLE_RATE, stream_audio_s)
            logging.info(f"Backend: Session {client_info}: prosody rate: source {audio_segment_np.size / AUDIO_SAMPLE_RATE:.2f}s vs tts {stream_audio_s:.2f}s -> length_scale {scale:.2f}")

        tts_total_time = time.perf_counter() - tts_start_time
        logging.info(f"Backend: Session {client_info}: Streaming TTS completed. Time: {tts_total_time:.2f}s. Chunks: {chunk_count}")

    else:
        logging.info(f"Backend: Session {client_info}: Starting TTS synthesis with '{tts_model_choice}'...")
        audio_wav, sample_rate, tts_latency = await loop.run_in_executor(
            None,
            lambda: tts_engine.synthesize(translated_text, language=tts_lang, speaker_wav_path=speaker_wav_path)
        )
        tts_total_time = time.perf_counter() - tts_start_time

    if audio_wav is not None and sample_rate is not None:
        # Ensure audio_wav is a numpy array
        if torch is not None and isinstance(audio_wav, torch.Tensor):
            audio_wav = audio_wav.cpu().numpy()
        
        audio_buffer = io.BytesIO()
        # Use soundfile to write the audio to a WAV buffer
        await loop.run_in_executor(None, lambda: sf.write(audio_buffer, audio_wav, sample_rate, format="WAV"))
        # Send the WAV data as bytes over the WebSocket
        await safe_send("tts_audio", audio_buffer.getvalue(), is_bytes=True)
    elif not getattr(tts_engine, "SUPPORTS_STREAMING", False):
        # T045: this used to hardcode `!= "xtts"` (the only streaming engine at the time) — now
        # piper/hybrid also stream (audio already sent chunk-by-chunk above, audio_wav staying
        # None is expected for them, not an error). Generalized to the same SUPPORTS_STREAMING
        # flag the dispatch above already uses, instead of a second hardcoded engine-name check
        # that silently drifts out of sync with it (confirmed live: the old check fired a false
        # "TTS model not ready" error on every successful piper/hybrid streaming synthesis).
        logging.warning(f"Backend: Session {client_info}: {tts_model_choice} TTS model not initialized or produced no audio. No TTS will be performed for {_log_text(translated_text)}.")
        await safe_send("error", {"type": "error", "message": f"{tts_model_choice} TTS model not ready or invalid. No TTS output."})
        return
    websocket.timestamps["tts_end"].append(time.perf_counter()) # Corrected to log end time
    logging.info(f"Backend: Session {client_info}: TTS completed. Time: {tts_total_time:.2f}s")

    metrics_payload = {"stt_time": stt_total_time, "mt_time": mt_total_time, "tts_time": tts_total_time, "input_to_stt_latency": input_to_stt_latency}
    metrics_payload["onset_to_first_audio_s"] = (
        first_audio_sent_time - source_onset_time
        if first_audio_sent_time is not None and source_onset_time is not None else None)
    metrics_payload["source_onset_time"] = source_onset_time
    metrics_payload["first_audio_sent_time"] = first_audio_sent_time
    if is_final:
        metrics_payload["total_latency"] = stt_total_time + mt_total_time + tts_total_time
        logging.info(f"Backend: Sending final_metrics: {metrics_payload}")
    else:
        logging.debug(f"Backend: Sending streaming metrics: {metrics_payload}")
    await safe_send("final_metrics", {"type": "final_metrics", "metrics": metrics_payload, "is_final": is_final})

# T043: fire-and-forget wrapper so audio capture (the main receive loop) never blocks on
# translation. Cancellation (T046 barge-in) unwinds cleanly here instead of crashing or leaving
# an unretrieved-exception warning. NOTE (verified live, see documentation/realtime_pipeline_rearchitecture_2026-08-22.md):
# cancelling a task that's currently awaiting a run_in_executor() call (STT/MT/non-streaming TTS)
# does not preempt that specific blocking call — asyncio defers the CancelledError until that
# call returns, then stops before the next stage/send. Worst-case barge-in lag is bounded by
# whichever single stage is in flight, not the whole pipeline. Real numbers in the doc above.
async def _run_pipeline_task(websocket, audio_segment_np, segment_start_time, is_final, session_config, client_info, pretranscribed=None, source_span=None, source_onset_time=None):
    # T044: every real (SILENCE_TIMEOUT-triggered) segment resets the captioner's buffer, so its
    # partials don't bleed across utterance boundaries. All call sites now funnel through here,
    # so one reset call covers every trigger point instead of duplicating it at each of them.
    session_data = active_sessions.get(client_info)
    if session_data is None or not session_data.get("capture_active", True):
        return
    generation = session_data.get("capture_generation", 0)
    task = asyncio.current_task()
    tasks = session_data.setdefault("pipeline_tasks", set())
    tasks.add(task)
    seq = session_data.get("tts_next_seq", 0)
    session_data["tts_next_seq"] = seq + 1
    if session_data.get("captioner") is not None:
        try:
            session_data["captioner"].reset()
        except Exception as e:
            logging.error(f"Backend: Session {client_info}: captioner reset error (non-fatal): {e}")
    try:
        if session_data.get("capture_generation", 0) != generation:
            return
        await _process_speech_segment_pipeline(websocket, audio_segment_np, segment_start_time, is_final=is_final, session_config=session_config, tts_seq=seq, pretranscribed=pretranscribed, source_span=source_span, source_onset_time=source_onset_time)
    except asyncio.CancelledError:
        logging.info(f"Backend: Session {client_info}: Pipeline task cancelled (superseded by new speech).")
    except Exception as e:
        logging.error(f"Backend: Session {client_info}: Unhandled error in pipeline task: {e}")
    finally:
        tasks.discard(task)
        session_data.setdefault("tts_events", {}).setdefault(seq, asyncio.Event()).set()

# T044: fire-and-forget per audio chunk. Bounded to one in-flight feed() at a time via
# captioner_lock — a chunk is dropped, never queued, if the previous call hasn't returned yet, so
# this can never build backlog the way the old always-full-pipeline streaming path did.
# ---- Incremental (simultaneous) mode ------------------------------------------------------------
from backend.stt.incremental import Agreement, Fragmenter

INCREMENTAL_STEP_S = float(os.environ.get("BP_INCR_STEP_S", "0.7"))      # min new audio between hypotheses
INCREMENTAL_MIN_AUDIO_S = float(os.environ.get("BP_INCR_MIN_AUDIO_S", "1.0"))


class _Frames(list):
    """VAD speech frames plus the stream-clock time (s) at which each frame ended."""
    def __init__(self):
        super().__init__()
        self.t = []

    def add(self, frame, t):
        self.append(frame)
        self.t.append(t)

    def clear(self):
        super().clear()
        self.t.clear()

    def trim(self, k):
        del self[:k]
        del self.t[:k]


class _IncState:
    def __init__(self, source_onset_time=None):
        self.source_onset_time = source_onset_time
        self.agree = Agreement()
        self.frag = Fragmenter(min_words=int(os.environ.get("BP_INCR_MIN_WORDS", "4")),
                               max_words=int(os.environ.get("BP_INCR_MAX_WORDS", "9")),
                               max_wait_s=float(os.environ.get("BP_INCR_MAX_WAIT_S", "1.6")))
        self.busy = False
        self.closed = False
        self.last_step = 0.0
        self.committed = 0
        self.context = ""


async def _emit_fragment(websocket, session_config, client_info, session_data, text, span, source_onset_time=None):
    """Translate + voice one agreed fragment; ordered with all other segments by _run_pipeline_task."""
    task = asyncio.create_task(_run_pipeline_task(
        websocket, np.zeros(1, dtype=np.float32), time.perf_counter(), True, session_config, client_info,
        pretranscribed=text, source_span=span, source_onset_time=source_onset_time))
    session_data["active_pipeline_task"] = task


async def _incremental_step(websocket, session_data, session_config, client_info, frames, inc):
    loop = asyncio.get_event_loop()
    try:
        n = len(frames)
        if n == 0 or inc.closed:
            return
        audio = np.concatenate(frames[:n])
        times = list(frames.t[:n])
        stt = session_data.get("stt_model")
        src = session_config.get("source_lang")
        async with session_data["model_call_lock"]:
            if inc.closed:
                return
            segs, _, _ = await loop.run_in_executor(None, lambda: stt.transcribe_audio(
                audio, AUDIO_SAMPLE_RATE, language=None if src == "auto" else src, vad_filter=False,
                initial_prompt=(inc.context[-200:] or None), word_timestamps=True))
        if inc.closed:
            return
        words = [(w.word.strip(), w.start, w.end) for sgm in segs for w in (getattr(sgm, "words", None) or []) if w.word.strip()]
        committed = inc.agree.update(words)
        logging.info(f"INCR step: audio={n*VAD_FRAME_DURATION/1000:.1f}s words={len(words)} committed={len(committed)} pending={len(inc.frag.pending)} hyp='{' '.join(w[0] for w in words)[:90]}'")
        if committed:
            frame_s = VAD_FRAME_DURATION / 1000.0
            def abs_t(rel):   # word time inside the snapshot -> stream clock
                i = min(max(int(rel / frame_s), 0), n - 1)
                return times[i]
            inc.frag.push([(w[0], abs_t(w[1]), abs_t(w[2])) for w in committed], time.perf_counter())
            inc.committed += len(committed)
            inc.context += " " + " ".join(w[0] for w in committed)
            frames.trim(min(n, int(committed[-1][2] / frame_s) + 1))   # audio before the last agreed word is done
        text = inc.frag.pop(time.perf_counter())
        if text:
            await _emit_fragment(websocket, session_config, client_info, session_data, text, inc.frag.span, inc.source_onset_time)
    except Exception as e:
        logging.error(f"Backend: Session {client_info}: incremental step error (non-fatal): {e}")
    finally:
        inc.busy = False
        inc.last_step = time.perf_counter()


async def _incremental_close(websocket, session_config, client_info, session_data, frames, inc):
    """Segment ended: voice any agreed-but-unflushed words first; return True if the remainder is too short to transcribe."""
    inc.closed = True
    text = inc.frag.pop(time.perf_counter(), final=True)
    if text:
        await _emit_fragment(websocket, session_config, client_info, session_data, text, inc.frag.span, inc.source_onset_time)
    return inc.committed > 0 and len(frames) * (VAD_FRAME_DURATION / 1000.0) < 0.3


async def _feed_captioner_task(websocket, session_data, client_info, audio_chunk):
    captioner = session_data.get("captioner")
    lock = session_data.get("captioner_lock")
    if captioner is None or lock is None or lock.locked():
        return
    async with lock:
        try:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, captioner.feed, audio_chunk)
            if text:
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_text(json.dumps({"type": "caption_partial", "text": text}))
                    _log_session_event(client_info, {"type": "caption_partial", "dir": "out", "text": text})
        except Exception as e:
            logging.error(f"Backend: Session {client_info}: captioner feed error (non-fatal): {e}")

# --- WebSocket Handler ---
async def handle_audio_stream(websocket: WebSocket, token: Optional[str] = None):
    # JWT auth: require a valid bearer token passed as ?token=<jwt> query param.
    # Without this, anyone on the network could open the WebSocket and use the
    # translation pipeline (the pre-hardening state).
    if not token:
        await websocket.close(code=4003, reason="Authentication required: pass ?token=<jwt>")
        return
    try:
        from backend.utils.auth import decode_access_token
        email = decode_access_token(token)
    except Exception:
        await websocket.close(code=4003, reason="Invalid or expired token")
        return
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
    finally:
        db.close()
    if user is None:
        await websocket.close(code=4003, reason="User not found")
        return

    # Use websocket.scope["client"] to get client host and port.
    # NOTE: app.py's websocket_endpoint (app.py:331) initializes the session under the
    # plain "host:port" key and all active_sessions lookups use client_info as that key,
    # so client_info MUST stay "host:port" (email is only for display/logging).
    client_host, client_port = websocket.scope["client"]
    client_info = f"{client_host}:{client_port}"
    logging.info(f"Client connected from {client_info} (user={email})")
    
    # Initialize timestamps attribute for this WebSocket connection
    websocket.timestamps = {
        "audio_input_start": [], "audio_input_end": [],
        "stt_start": [], "stt_end": [],
        "mt_start": [], "mt_end": [],
        "tts_start": [], "tts_end": [],
        "tts_audio_sent": []
    }

    # Retrieve or initialize session data. client_info is "host:port", matching the key
    # app.py's websocket_endpoint initialized the session under (app.py:331).
    session_data = active_sessions.setdefault(client_info, {
        "client_info": client_info,
        "stt_model": None,
        "mt_models": {},
        "tts_engine": None,
        "tts_engine_name": None,
        "vad_instance": None,
        "active_pipeline_task": None,  # T043: tracks the in-flight STT->MT->TTS asyncio.Task for this session, so a new speech-start can cancel a stale one (barge-in, T046)
        "model_call_lock": asyncio.Lock(),  # T046: serializes STT+MT calls on this session's shared model instances; NOT held during TTS, which is the only phase barge-in is allowed to cancel
        "captioner": None,  # T044
        "captioner_lock": asyncio.Lock(),  # T044: bounds captioner.feed() to one in-flight call at a time; a chunk is dropped (not queued) if the previous feed hasn't finished — best-effort, never adds backlog
        "last_transcription_text": None,  # Last non-empty streaming transcription, promoted to final if the tail segment is too short
        "session_config": { # Default config, will be updated by client
            "source_lang": "en",
            "target_lang": "sk",
            "tts_model_choice": DEFAULT_TTS_MODEL,
            "stt_model_size": DEFAULT_STT_MODEL_SIZE,
            "speaker_wav_path": None,
            "speaker_text": None,
            "speaker_lang": None,
            "vad_enabled": True,
            "vad_silence_timeout_s": SILENCE_TIMEOUT,
            "vad_max_segment_s": MAX_SEGMENT_S,
            "vad_end_of_sentence_gating": True,
            "barge_in": False,  # one-way interpretation: the speaker talking on must not cancel the translation still being voiced
            "language_pack": {},
        }
    })
    session_data["capture_active"] = False
    session_data["capture_generation"] = 0
    session_config = session_data["session_config"] # Reference to the session's config

    audio_queue = deque()
    speech_frames = _Frames() # VAD-detected speech frames of the open utterance (+ stream-clock end time per frame)
    received_samples = 0
    source_clock_origin = None
    utterance_onset_time = None
    inc = None  # _IncState while an utterance is open in incremental mode
    last_speech_time = time.perf_counter()
    in_speech_segment = False
    last_streaming_process_time = time.perf_counter() # Track last time a streaming chunk was processed
    
    # New variables for pre-VAD silence detection
    frame_size_samples = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
    streaming_chunk_samples = int(AUDIO_SAMPLE_RATE * STREAMING_CHUNK_LENGTH)

    # New variables for pre-VAD silence detection
    pre_vad_buffer = deque()
    pre_vad_threshold_met = False

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                # Starlette emits exactly one of these when the peer goes away and raises
                # RuntimeError if receive() is called again after it. This loop only checked
                # for "websocket.receive", so the disconnect message fell through, receive()
                # was called a second time, and the RuntimeError surfaced as two ERROR lines
                # on every client teardown (reproduced 2026-09-28 in the live WS rehearsal).
                # Clients flush with an explicit 'stop', so exiting here keeps the old
                # behaviour: cleanup in the finally block, no extra pipeline work.
                logging.info(f"Client {client_info} disconnected (code {message.get('code')}).")
                break
            if message["type"] == "websocket.receive": # Check for the correct message type
                if "text" in message:
                    data = json.loads(message["text"])
                    if data["type"] == "start":
                        session_data["capture_active"] = True
                        session_data["capture_generation"] += 1
                        logging.info("Received 'start' command. Beginning transcription.")
                        audio_queue.clear()
                        speech_frames.clear()
                        received_samples = 0
                        source_clock_origin = None
                        utterance_onset_time = None
                        pre_vad_buffer.clear() # Clear pre-VAD buffer on start
                        pre_vad_threshold_met = False # Reset threshold flag
                        in_speech_segment = False
                        if inc is not None:
                            inc.closed = True
                            inc = None
                        last_speech_time = time.time()
                        last_streaming_process_time = time.perf_counter()
                    elif data["type"] == "stop":
                        session_data["capture_active"] = False
                        session_data["capture_generation"] += 1
                        tasks = set(session_data.get("pipeline_tasks", set()))
                        if session_data.get("active_pipeline_task"):
                            tasks.add(session_data["active_pipeline_task"])
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        audio_queue.clear()
                        speech_frames.clear()
                        in_speech_segment = False
                        if inc is not None:
                            inc.closed = True   # an in-flight incremental step must not voice anything after Stop
                            inc = None
                        pre_vad_buffer.clear()
                        pre_vad_threshold_met = False
                        await websocket.send_text(json.dumps({"type": "session_stopped"}))
                    elif data["type"] == "config_update":
                        # THESIS NOTE: Use session lock to prevent race conditions when
                        # multiple config updates arrive concurrently for the same session
                        session_lock = get_session_lock(client_info)
                        async with session_lock:
                            new_source_lang = data.get("source_lang", session_config["source_lang"])
                            new_target_lang = data.get("target_lang", session_config["target_lang"])
                            new_tts_model_choice = data.get("tts_model_choice", session_config["tts_model_choice"])
                            new_speaker_wav_path = authorize_speaker_wav(data.get("speaker_wav_path", session_config["speaker_wav_path"]), user.id)
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

                            if config_changed or session_data.get("stt_model") is None:
                                logging.info(f"Backend: Configuration changed. Re-initializing models with: Source={new_source_lang}, Target={new_target_lang}, TTS={new_tts_model_choice}, SpeakerWav={new_speaker_wav_path}, VAD_Enabled={new_vad_enabled}")
                                # Direction must land in session_config BEFORE init: the
                                # piper->piper_sk_personal remap reads target_lang from here,
                                # and with a stale "sk" an EN target keeps the Slovak voice
                                # (measured live 2026-09-29: English read by piper_sk_personal).
                                old_source_lang = session_config["source_lang"]
                                session_config["source_lang"] = new_source_lang
                                session_config["target_lang"] = new_target_lang
                                await initialize_all_models(                                    client_info,
                                    new_source_lang, new_target_lang, new_tts_model_choice,
                                    stt_model_size=data.get("stt_model_size", DEFAULT_STT_MODEL_SIZE if new_source_lang != old_source_lang else session_config["stt_model_size"]),
                                    speaker_wav_path=new_speaker_wav_path, speaker_text=new_speaker_text,
                                    speaker_lang=new_speaker_lang, vad_enabled_param=new_vad_enabled, websocket=websocket
                                )
                                
                                # Check if source language changed and send a translated notification
                                if new_source_lang != old_source_lang:
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
                                # SANDBOX-FIX (2026-09-28): do NOT write back the requested
                                # tts_model_choice here. initialize_all_models may have remapped
                                # it to a measured default (piper->piper_sk_personal) and stored
                                # the effective name; restoring 'piper' orphans the initialized
                                # engine so every later lookup reports "TTS not initialized"
                                # and no synthesis ever runs (seen live over WS).
                                session_config["speaker_wav_path"] = new_speaker_wav_path
                                session_config["speaker_text"] = new_speaker_text
                                session_config["speaker_lang"] = new_speaker_lang
                                session_config["vad_enabled"] = new_vad_enabled # Update session VAD state
                                
                                await websocket.send_text(json.dumps({"type": "status", "message": "Configuration updated and models re-initialized."}))
                                await websocket.send_text(json.dumps({"type": "models_loading_status", "fully_loaded": True, "route_status": session_data.get("route_status")}))
                            else:
                                logging.info(f"Backend: Session {client_info}: Configuration updated, but no model re-initialization needed.")
                                await websocket.send_text(json.dumps({"type": "status", "message": "Configuration updated."}))
                elif "bytes" in message:
                    if not session_data.get("capture_active"):
                        continue
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
                    if audio_np.size > 0:
                        logging.debug(f"Backend: audio_np min: {np.min(audio_np)}, max: {np.max(audio_np)}")
                        logging.debug(f"Backend: audio_np contains NaN: {np.isnan(audio_np).any()}, Inf: {np.isinf(audio_np).any()}")
                    logging.debug(f"Backend: audio_np shape: {audio_np.shape}, dtype: {audio_np.dtype}")
                    # All-zero chunks are digital silence (mic muted / tail gap), NOT speech.
                    # They must still flow into the VAD frame loop below so the
                    # SILENCE_TIMEOUT close branch sees the gap. Skipping them here
                    # with `continue` starves close-eval: trailing silence never
                    # closes (only teardown flushes). WebRTC VAD scores pure zeros
                    # as non-speech, so there is no false-trigger risk.
                    _is_digital_silence = not audio_np.any()
                    if _is_digital_silence:
                        logging.debug("Backend: all-zero audio chunk — feeding as silence for close-eval.")

                    # T044: feed the streaming captioner, fire-and-forget. Never awaited inline —
                    # a slow feed() call must not reintroduce the capture-blocking bug T043 just
                    # fixed. Best-effort: errors are logged and swallowed, they never affect the
                    # real STT/MT/TTS path below, which doesn't read anything this produces.
                    if session_data.get("captioner") is not None:
                        asyncio.create_task(_feed_captioner_task(websocket, session_data, client_info, audio_np.copy()))

                    # Calculate RMS on raw audio (not normalized) to get accurate level readings
                    # Clip to ensure values are within valid range before RMS calculation
                    audio_np_clipped = np.clip(audio_np, -1.0, 1.0)
                    rms = np.sqrt(np.mean(audio_np_clipped**2))
                    
                    # Ensure RMS is a valid number before sending
                    if np.isnan(rms) or np.isinf(rms):
                        logging.warning(f"Backend: Calculated RMS is NaN or Inf ({rms}). Sending 0.0 to frontend.")
                        rms = 0.0
                    
                    await websocket.send_text(json.dumps({"type": "audio_level", "level": float(rms)}))

                    # Anchor the sample clock at packet arrival minus packet duration.
                    # Browser capture/network delay is outside this server-side metric.
                    if source_clock_origin is None:
                        source_clock_origin = time.perf_counter() - len(audio_np) / AUDIO_SAMPLE_RATE
                    received_samples += len(audio_np)
                    pre_vad_threshold_met = _queue_pre_vad_audio(
                        audio_np, audio_queue, pre_vad_buffer, pre_vad_threshold_met)

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
                                task = asyncio.create_task(_run_pipeline_task(websocket, chunk_to_process, time.perf_counter(), False, session_config, client_info, source_onset_time=source_clock_origin + (received_samples - len(chunk_to_process)) / AUDIO_SAMPLE_RATE))
                                session_data["active_pipeline_task"] = task
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
                                logging.debug("Backend: VAD received an all-zero audio frame.")

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
                                speech_frames.add(frame_float33, (received_samples - len(audio_queue)) / AUDIO_SAMPLE_RATE)
                                if not in_speech_segment:
                                    logging.info(f"Backend: Speech segment STARTED at {time.perf_counter():.3f}s")
                                    in_speech_segment = True
                                    utterance_onset_time = source_clock_origin + (
                                        received_samples - len(audio_queue) - frame_size_samples) / AUDIO_SAMPLE_RATE
                                    inc = _IncState(utterance_onset_time) if session_config.get("incremental") else None
                                    # T046 barge-in: only cancel a previous task if it's past STT+MT (lock released) —
                                    # i.e. it's in/about to enter TTS (the only phase where "two voices at once" is a
                                    # real problem) or already finished. If it's still inside STT/MT, let it finish:
                                    # live-tested (test/interrupt_smoke_test.py) confirmed unconditional cancellation
                                    # silently drops most sentences in ordinary back-to-back continuous speech, since
                                    # natural inter-sentence pauses are shorter than one STT+MT round trip.
                                    prev_task = session_data.get("active_pipeline_task")
                                    if (session_config.get("barge_in", False) and prev_task is not None and not prev_task.done()
                                            and not session_data["model_call_lock"].locked()):
                                        logging.info(f"Backend: Session {client_info}: New speech started during TTS/idle — cancelling in-flight pipeline task (barge-in).")
                                        prev_task.cancel()
                                last_speech_time = time.perf_counter()

                                # Safety Valve (vad_max_segment_s bound): force-close long open segments
                                # even without silence — dense speech otherwise waits for teardown.
                                # Next voiced frame opens a chained segment; nothing is dropped.
                                current_speech_duration = len(speech_frames) * (VAD_FRAME_DURATION / 1000.0)
                                max_segment_s = session_config.get("vad_max_segment_s", MAX_SEGMENT_S)
                                if inc is not None:
                                    max_segment_s = max(max_segment_s, 20.0)   # fragments flush as words agree; the valve is a last resort
                                    if (not inc.busy and time.perf_counter() - inc.last_step >= INCREMENTAL_STEP_S
                                            and len(speech_frames) * (VAD_FRAME_DURATION / 1000.0) >= INCREMENTAL_MIN_AUDIO_S
                                            and session_data.get("stt_model") is not None):
                                        inc.busy = True
                                        asyncio.create_task(_incremental_step(websocket, session_data, session_config, client_info, speech_frames, inc))
                                if current_speech_duration > max_segment_s:
                                    logging.warning(f"Backend: Speech segment exceeded {max_segment_s:.0f}s ({current_speech_duration:.2f}s). Forcing processing.")
                                    skip_final = inc is not None and await _incremental_close(websocket, session_config, client_info, session_data, speech_frames, inc)
                                    inc = None
                                    if not skip_final and len(speech_frames):
                                        final_speech_segment_np = np.concatenate(speech_frames)
                                        task = asyncio.create_task(_run_pipeline_task(websocket, final_speech_segment_np, last_speech_time, True, session_config, client_info, source_onset_time=utterance_onset_time))
                                        session_data["active_pipeline_task"] = task
                                    speech_frames.clear()
                                    in_speech_segment = False
                                    # Reset buffers
                                    pre_vad_buffer.clear()
                                    pre_vad_threshold_met = False
                                    last_speech_time = 0 # Reset

                                # Streaming chunk processing: process STREAMING_CHUNK_LENGTH (0.5s)
                                # audio chunks during speech so STT starts mid-utterance instead of
                                # waiting for segment close. RMS pre-filter blocks background noise.
                                now = time.perf_counter()
                                if (now - last_streaming_process_time >= STREAMING_CHUNK_LENGTH
                                        and len(speech_frames) > 0):
                                    # Collect streaming_chunk_samples worth of audio from speech_frames
                                    needed = streaming_chunk_samples
                                    collected = 0
                                    chunk_parts = []
                                    while collected < needed and speech_frames:
                                        frame = speech_frames[0]
                                        frame_len = len(frame)
                                        if collected + frame_len <= needed:
                                            chunk_parts.append(speech_frames.pop(0))
                                            collected += frame_len
                                        else:
                                            chunk_parts.append(frame[needed - collected:])
                                            speech_frames[0] = frame[:needed - collected]
                                            collected = needed
                                    if chunk_parts and collected > 0:
                                        streaming_segment_np = np.concatenate(chunk_parts)
                                        # RMS pre-filter: skip noise-only chunks (prevents "I'm sorry" loop)
                                        chunk_rms = float(np.sqrt(np.mean(np.clip(streaming_segment_np, -1.0, 1.0) ** 2)))
                                        if chunk_rms >= SILENCE_RMS_THRESHOLD:
                                            chunk_onset = source_clock_origin + (
                                                received_samples - len(audio_queue) - collected) / AUDIO_SAMPLE_RATE
                                            asyncio.create_task(_process_speech_segment_pipeline(
                                                websocket, streaming_segment_np, chunk_onset,
                                                is_final=False, session_config=session_config))
                                        last_streaming_process_time = now

                            elif in_speech_segment:
                                # Silence detected after speech, or speech ended
                                if time.perf_counter() - last_speech_time > session_config.get("vad_silence_timeout_s", SILENCE_TIMEOUT):
                                    skip_final = inc is not None and await _incremental_close(websocket, session_config, client_info, session_data, speech_frames, inc)
                                    inc = None
                                    if speech_frames and not skip_final:
                                        final_dur_s = len(speech_frames) * (VAD_FRAME_DURATION / 1000.0)
                                        # Final segment too short (< 1.0s): Whisper can't transcribe it reliably.
                                        # The last streaming transcription serves as the final result.
                                        if final_dur_s < 1.0:
                                            logging.info(f"Backend: Session {client_info}: Final segment {final_dur_s:.2f}s too short, skipping — last streaming transcription stands as final.")
                                            # Promote last streaming transcription to final so the frontend knows the utterance is done
                                            last_text = session_data.get("last_transcription_text")
                                            if last_text:
                                                await websocket.send_text(json.dumps({"type": "transcription_result", "transcribed": last_text, "is_final": True, "metrics": {"stt_time": 0.0}}))
                                            speech_frames.clear()
                                        else:
                                            final_speech_segment_np = np.concatenate(speech_frames)
                                            # Sanitize final_speech_segment_np before pipeline
                                            if np.isnan(final_speech_segment_np).any() or np.isinf(final_speech_segment_np).any():
                                                logging.warning("Backend: Detected NaN or Inf in final_speech_segment_np. Sanitizing to 0.")
                                                final_speech_segment_np = np.nan_to_num(final_speech_segment_np, nan=0.0, posinf=0.0, neginf=0.0)
                                            logging.info(f"Backend: Final speech segment ENDED at {time.perf_counter():.3f}s. Duration: {final_dur_s:.3f}s")
                                            task = asyncio.create_task(_run_pipeline_task(websocket, final_speech_segment_np, last_speech_time, True, session_config, client_info, source_onset_time=utterance_onset_time))
                                            session_data["active_pipeline_task"] = task
                                            speech_frames.clear()
                                    in_speech_segment = False
                                    # Keep queued source frames for the next segment; reset speech state.
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
                            task = asyncio.create_task(_run_pipeline_task(websocket, chunk_to_process, time.perf_counter(), False, session_config, client_info, source_onset_time=source_clock_origin + (received_samples - len(chunk_to_process)) / AUDIO_SAMPLE_RATE))
                            session_data["active_pipeline_task"] = task
                            audio_queue.clear()

    except WebSocketDisconnect:
        logging.info(f"Client {client_info} disconnected normally. Final speech_frames size: {len(speech_frames)}")
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
            cleanup_session_models(client_info)
            session_data = active_sessions.pop(client_info)

            # T043: cancel any pipeline task orphaned by an error/disconnect path that didn't
            # already await it, so it doesn't keep running against a torn-down session.
            leftover_task = session_data.get("active_pipeline_task")
            if leftover_task is not None and not leftover_task.done():
                logging.info(f"Backend: Session {client_info}: Cancelling leftover pipeline task during cleanup.")
                leftover_task.cancel()

            # Explicitly delete model references to trigger garbage collection
            if session_data.get("stt_model"):
                logging.debug(f"Backend: Deleting STT model for session {client_info}")
                del session_data["stt_model"]
            
            if session_data.get("tts_engine"):
                logging.debug(f"Backend: Deleting TTS engine ('{session_data.get('tts_engine_name')}') for session {client_info}")
                del session_data["tts_engine"]
            
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
        log_handle = _session_log_files.pop(client_info, None)
        if log_handle and not log_handle.closed:
            log_handle.close()
        
        logging.info(f"Backend: Cleanup complete for client {client_info}.")
