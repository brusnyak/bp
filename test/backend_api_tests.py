import pytest
import httpx
import asyncio
import os
import json
import soundfile as sf
import numpy as np
from unittest.mock import patch, AsyncMock
import io

# Assuming the FastAPI app is defined in backend.main and can be imported
# For testing, we'll use a TestClient
from fastapi.testclient import TestClient
from app import app # Import the main FastAPI app instance
from backend.main import SPEAKER_VOICES_DIR, SPEAKER_VOICES_METADATA_FILE, AUDIO_SAMPLE_RATE, initialize_all_models, _read_speaker_voices_metadata, _write_speaker_voices_metadata, get_current_user
from backend.utils.db_manager import User
from fastapi import Depends

# Create a TestClient for the FastAPI application
client = TestClient(app)

# --- Fixtures for Test Setup and Teardown ---
@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    # Ensure speaker_voices directory exists for tests
    os.makedirs(SPEAKER_VOICES_DIR, exist_ok=True)
    # Snapshot the real registry + directory contents BEFORE the test writes its empty
    # registry. The old teardown wrote [] and unlinked every *.wav in place, which on a
    # machine with real recordings destroyed the user's speaker metadata for good
    # (nothing in git to restore it from on Windows; on macOS only the committed copy
    # saves you). Restore-instead-of-destroy, 2026-09-28.
    meta_backup = None
    if os.path.exists(SPEAKER_VOICES_METADATA_FILE):
        with open(SPEAKER_VOICES_METADATA_FILE, encoding="utf-8") as f:
            meta_backup = f.read()
    files_before = set(os.listdir(SPEAKER_VOICES_DIR))
    # Clear metadata file before tests
    _write_speaker_voices_metadata([]) # Initialize as an empty list
    yield
    # Clean up after tests: restore the registry, remove only files the test created
    if meta_backup is not None:
        with open(SPEAKER_VOICES_METADATA_FILE, "w", encoding="utf-8") as f:
            f.write(meta_backup)
    elif os.path.exists(SPEAKER_VOICES_METADATA_FILE):
        os.remove(SPEAKER_VOICES_METADATA_FILE)
    for f in os.listdir(SPEAKER_VOICES_DIR):
        if f not in files_before and (f.endswith(".wav") or f.startswith("temp_upload_")):
            os.remove(os.path.join(SPEAKER_VOICES_DIR, f))

# --- Mocking external dependencies ---
# SANDBOX-NOTE (2026-09-28): the old mock_models fixture patched backend.main.F5_TTS,
# an engine that no longer exists, and the translate_phrase tests below targeted a
# /api/translate_phrase route that no longer exists either (404 for all six tests).
# This file now smoke-tests the routes that DO exist. The deleted machinery is
# documented here so a future F5/translate_phrase revival knows what was here.
# (No autouse mocks: the live-simulation tests below intentionally use real models.)

@pytest.fixture(scope="module")
def mock_user():
    """Provides a mock User object for authentication."""
    class MockUser:
        def __init__(self, id: int, username: str, email: str):
            self.id = id
            self.username = username
            self.email = email
    return MockUser(id=1, username="testuser", email="testuser@example.com")

@pytest.fixture(scope="module", autouse=True)
def override_get_current_user_dependency(mock_user):
    """Overrides the get_current_user dependency for tests."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield
    app.dependency_overrides.clear()

# --- Live API smoke tests (real routes, real models where cheap) ---
def test_voice_lab_status_no_auth():
    """Capabilities endpoint: no auth, lists engines + per-stage CPU backends."""
    response = client.get("/api/voice-lab/status")
    assert response.status_code == 200
    body = response.json()
    assert "piper" in body["engines"]
    # Coqui has no Windows wheels -> guarded import (backend/tts/base.py, backend/main.py).
    # So "xtts" is advertised exactly when Coqui actually imported: absent on stock
    # Windows, present on macOS/Linux. Asserting it must be absent everywhere was a
    # Windows-only expectation and failed on macOS (measured 2026-09-28).
    from backend.main import CoquiTTS
    assert ("xtts" in body["engines"]) is (CoquiTTS is not None)
    assert body["hardware_backends"]["stt"] == "cpu"
    assert body["hardware_backends"]["mt"] == "cpu"


def test_root_serves_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_initialize_full_pipeline_live():
    """POST /initialize with real STT/MT/Piper: proves the endpoint, the model
    stack, and the piper->piper_sk_personal remap write-back (engine must be
    reachable under its stored name, otherwise TTS silently never runs)."""
    response = client.post(
        "/initialize",  # app-level route (not under /api -- see app.py)
        params={"source_lang": "en", "target_lang": "sk",
                "tts_model_choice": "piper", "stt_model_size": "base",
                "vad_enabled_param": True},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    from backend.main import active_sessions
    session = active_sessions.get("http_init_client")
    assert session is not None
    assert session["stt_model"] is not None
    assert session["tts_engine"] is not None
    assert session["tts_engine_name"] == session["session_config"]["tts_model_choice"] == "piper_sk_personal"
