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
    # Clear metadata file before tests
    _write_speaker_voices_metadata([]) # Initialize as an empty list
    yield
    # Clean up after tests
    if os.path.exists(SPEAKER_VOICES_METADATA_FILE):
        os.remove(SPEAKER_VOICES_METADATA_FILE)
    # Clean up any test WAV files
    for f in os.listdir(SPEAKER_VOICES_DIR):
        if f.endswith(".wav") or f.startswith("temp_upload_"):
            os.remove(os.path.join(SPEAKER_VOICES_DIR, f))

# --- Mocking external dependencies ---
@pytest.fixture(autouse=True)
def mock_models():
    """Mocks STT, MT, and TTS models for tests."""
    with patch('backend.main.FasterWhisperSTT') as MockSTT, \
         patch('backend.main.CTranslate2MT') as MockMT, \
         patch('backend.main.F5_TTS') as MockF5TTS, \
         patch('backend.main.PiperTTS') as MockPiperTTS, \
         patch('backend.main.webrtcvad.Vad') as MockVAD:
        
        # Configure STT mock
        mock_stt_instance = MockSTT.return_value
        mock_stt_instance.transcribe_audio.return_value = ([AsyncMock(text="mocked transcription")], 0.1, "en")
        
        # Configure MT mock
        mock_mt_instance = MockMT.return_value
        mock_mt_instance.translate.return_value = ("mocked translation", 0.1)

        # Configure TTS mocks
        mock_f5_tts_instance = MockF5TTS.return_value
        mock_f5_tts_instance.synthesize.return_value = (np.zeros(16000), 16000, 0.1)
        mock_piper_tts_instance = MockPiperTTS.return_value
        mock_piper_tts_instance.synthesize.return_value = (np.zeros(16000), 16000, 0.1)

        yield MockSTT, MockMT, MockF5TTS, MockPiperTTS, MockVAD

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

# --- Tests for /translate_phrase endpoint ---
@pytest.mark.asyncio
async def test_translate_phrase_success(mock_models):
    # Initialize models first, as translate_phrase depends on STT and MT
    await initialize_all_models(client_info="test_client", source_lang="en", target_lang="sk", tts_model_choice="piper")
    
    response = client.post(
        "/api/translate_phrase", # Updated endpoint path
        json={"phrase": "Hello world", "target_lang": "sk"}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["translated_text"] == "mocked translation"
    
    # Verify MT model was called
    mock_models[1].return_value.translate.assert_called_once_with("Hello world", "en", "sk")

@pytest.mark.asyncio
async def test_translate_phrase_same_language(mock_models):
    # Initialize models
    await initialize_all_models(client_info="test_client", source_lang="en", target_lang="en", tts_model_choice="piper")

    response = client.post(
        "/api/translate_phrase", # Updated endpoint path
        json={"phrase": "Hello world", "target_lang": "en"}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["translated_text"] == "Hello world" # Should return original phrase
    
    # Verify MT model was NOT called
    mock_models[1].return_value.translate.assert_not_called()

@pytest.mark.asyncio
async def test_translate_phrase_stt_not_initialized():
    # Initialize models with a dummy client_info, then patch STT model
    await initialize_all_models(client_info="test_client", source_lang="en", target_lang="sk", tts_model_choice="piper")
    with patch('backend.main.active_sessions.get("test_client")["stt_model"]', None):
        response = client.post(
            "/api/translate_phrase", # Updated endpoint path
            json={"phrase": "Hello world", "target_lang": "sk"}
        )
        assert response.status_code == 500
        assert "STT model not initialized" in response.json()["detail"]

@pytest.mark.asyncio
async def test_translate_phrase_mt_not_initialized(mock_models):
    # Initialize STT but not MT for the specific language pair
    await initialize_all_models(client_info="test_client", source_lang="en", target_lang="en", tts_model_choice="piper") # Init with same lang to avoid MT init
    
    # Manually clear the specific MT model for the test client
    from backend.main import active_sessions
    if "test_client" in active_sessions and "en-sk" in active_sessions["test_client"]["mt_models"]:
        del active_sessions["test_client"]["mt_models"]["en-sk"]

    response = client.post(
        "/api/translate_phrase", # Updated endpoint path
        json={"phrase": "Hello world", "target_lang": "sk"}
    )
    assert response.status_code == 200 # Expect 200 because dynamic initialization is now handled
    assert response.json()["status"] == "success"
    assert response.json()["translated_text"] == "mocked translation"

@pytest.mark.asyncio
async def test_translate_phrase_missing_phrase():
    response = client.post(
        "/api/translate_phrase", # Updated endpoint path
        json={"target_lang": "sk"}
    )
    assert response.status_code == 422 # FastAPI validation error
    assert "Field required" in response.json()["detail"][0]["msg"]

@pytest.mark.asyncio
async def test_translate_phrase_missing_target_lang():
    response = client.post(
        "/api/translate_phrase", # Updated endpoint path
        json={"phrase": "Hello world"}
    )
    assert response.status_code == 422 # FastAPI validation error
    assert "Field required" in response.json()["detail"][0]["msg"]
