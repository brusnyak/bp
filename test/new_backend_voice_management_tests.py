import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from backend.utils.db_manager import User
from backend.utils.auth import get_password_hash
import os
import json
import httpx
import asyncio
import wave # For creating dummy WAV file
import numpy as np # For creating dummy WAV file

# Mock JWT token for testing purposes
MOCK_JWT_TOKEN_PREFIX = "mock-jwt-token-for-"
TEST_USER_EMAIL = "test@example.com"
TEST_USER_USERNAME = "testuser"
TEST_USER_PASSWORD = "password"
TEST_USER_TOKEN = f"{MOCK_JWT_TOKEN_PREFIX}{TEST_USER_EMAIL}"

# Dummy audio file for testing uploads
DUMMY_AUDIO_PATH = "test/dummy_audio.wav"
DUMMY_AUDIO_SAMPLE_RATE = 16000
DUMMY_AUDIO_DURATION = 1 # second

@pytest.fixture(scope="session", autouse=True)
def create_dummy_audio_file():
    """Creates a dummy WAV file for testing voice uploads."""
    if not os.path.exists(DUMMY_AUDIO_PATH):
        # Generate a simple sine wave
        frequency = 440  # Hz
        t = np.linspace(0, DUMMY_AUDIO_DURATION, int(DUMMY_AUDIO_SAMPLE_RATE * DUMMY_AUDIO_DURATION), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5 # Half of max int16 value
        data = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

        with wave.open(DUMMY_AUDIO_PATH, 'w') as wf:
            wf.setnchannels(1) # Mono
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(DUMMY_AUDIO_SAMPLE_RATE)
            wf.writeframes(data.tobytes())
        print(f"\nCreated dummy audio file: {DUMMY_AUDIO_PATH}")
    yield
    if os.path.exists(DUMMY_AUDIO_PATH):
        os.remove(DUMMY_AUDIO_PATH)
        print(f"Removed dummy audio file: {DUMMY_AUDIO_PATH}")

@pytest.fixture(scope="function")
def authenticated_test_client(test_client: TestClient, app_with_test_db):
    """
    Provides a TestClient instance with an authenticated user.
    This fixture ensures a user exists in the database and provides a client
    that sends a mock JWT token for that user.
    """
    # Ensure the default user exists (created by app's lifespan event)
    # We don't need to explicitly create it here, as app_with_test_db handles it.
    
    # The test_client fixture already provides a client for the app_with_test_db
    # We just need to ensure the headers are set for authentication.
    test_client.headers["Authorization"] = f"Bearer {TEST_USER_TOKEN}"
    yield test_client
    # Clean up any voices uploaded by the user during the test
    # This requires accessing the backend's SPEAKER_VOICES_DIR and metadata
    from backend.main import SPEAKER_VOICES_DIR, _read_speaker_voices_metadata, _write_speaker_voices_metadata
    
    metadata = _read_speaker_voices_metadata()
    updated_metadata = []
    for voice in metadata:
        # Assuming current_user.id is 1 for the default test user
        if voice.get("user_id") == 1: 
            file_path = os.path.join(SPEAKER_VOICES_DIR, voice["filename"])
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Cleaned up uploaded voice file: {file_path}")
            else:
                print(f"Voice file not found during cleanup (already deleted?): {file_path}")
        else:
            updated_metadata.append(voice)
    _write_speaker_voices_metadata(updated_metadata)
    print("Cleaned up user-specific voice metadata.")


# --- Authentication Tests ---

def test_register_user_success(test_client: TestClient):
    """Test successful user registration."""
    response = test_client.post(
        "/api/register",
        json={"username": "newuser", "email": "new@example.com", "password": "newpassword"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "User registered successfully"

def test_register_user_duplicate_email(test_client: TestClient):
    """Test registration with a duplicate email."""
    # First registration (default user from app.py lifespan)
    response = test_client.post(
        "/api/register",
        json={"username": "anotheruser", "email": TEST_USER_EMAIL, "password": "somepassword"}
    )
    assert response.status_code == 400
    assert "Email already registered" in response.json()["detail"]

def test_register_user_duplicate_username(test_client: TestClient):
    """Test registration with a duplicate username."""
    # First registration (default user from app.py lifespan)
    response = test_client.post(
        "/api/register",
        json={"username": TEST_USER_USERNAME, "email": "another@example.com", "password": "somepassword"}
    )
    assert response.status_code == 400
    assert "Username already taken" in response.json()["detail"]

def test_login_user_success(test_client: TestClient):
    """Test successful user login."""
    response = test_client.post(
        "/api/token",
        data={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_login_user_invalid_credentials(test_client: TestClient):
    """Test login with invalid credentials."""
    response = test_client.post(
        "/api/token",
        data={"username": TEST_USER_EMAIL, "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]

def test_login_user_not_found(test_client: TestClient):
    """Test login for a non-existent user."""
    response = test_client.post(
        "/api/token",
        data={"username": "nonexistent@example.com", "password": "anypassword"}
    )
    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


# --- Voice Management Tests ---

def test_upload_voice_success(authenticated_test_client: TestClient, create_dummy_audio_file):
    """Test successful upload of a speaker voice."""
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        response = authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio.wav", f, "audio/wav")},
            data={"voice_name": "My Test Voice", "speaker_lang": "en"} # Added speaker_lang
        )
    assert response.status_code == 200
    assert response.json()["message"] == "Voice 'My Test Voice' uploaded successfully."

def test_upload_voice_empty_name(authenticated_test_client: TestClient, create_dummy_audio_file):
    """Test uploading a voice with an empty name."""
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        response = authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio.wav", f, "audio/wav")},
            data={"voice_name": "", "speaker_lang": "en"}
        )
    assert response.status_code == 400
    assert "Voice name cannot be empty." in response.json()["detail"]

def test_upload_voice_duplicate_name(authenticated_test_client: TestClient, create_dummy_audio_file):
    """Test uploading a voice with a duplicate name for the same user."""
    # First upload
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio.wav", f, "audio/wav")},
            data={"voice_name": "Duplicate Voice", "speaker_lang": "en"}
        )
    # Second upload with same name
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        response = authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio.wav", f, "audio/wav")},
            data={"voice_name": "Duplicate Voice", "speaker_lang": "en"}
        )
    assert response.status_code == 409
    assert "Voice with name 'Duplicate Voice' already exists for this user." in response.json()["detail"]

def test_upload_voice_unauthenticated(test_client: TestClient, create_dummy_audio_file):
    """Test uploading a voice without authentication."""
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        response = test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio.wav", f, "audio/wav")},
            data={"voice_name": "Unauthorized Voice", "speaker_lang": "en"} # Added speaker_lang
        )
    assert response.status_code == 403
    assert "Not authenticated" in response.json()["detail"]

def test_get_voices_success(authenticated_test_client: TestClient, create_dummy_audio_file):
    """Test retrieving uploaded voices for an authenticated user."""
    # Upload a voice first
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio.wav", f, "audio/wav")},
            data={"voice_name": "Voice to Get", "speaker_lang": "sk"} # Added speaker_lang
        )
    
    response = authenticated_test_client.get("/api/voices")
    assert response.status_code == 200
    voices = response.json()
    assert isinstance(voices, list)
    assert len(voices) > 0
    assert any(v["name"] == "Voice to Get" and v["language"] == "sk" for v in voices) # Verify speaker_lang

def test_get_voices_unauthenticated(test_client: TestClient):
    """Test retrieving voices without authentication."""
    response = test_client.get("/api/voices")
    assert response.status_code == 403
    assert "Not authenticated" in response.json()["detail"]

def test_rename_voice_success(authenticated_test_client: TestClient, create_dummy_audio_file):
    """Test successful renaming of a speaker voice."""
    # Upload a voice first
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio.wav", f, "audio/wav")},
            data={"voice_name": "Old Voice Name", "speaker_lang": "en"}
        )
    
    response = authenticated_test_client.put(
        "/api/voices/rename",
        json={"old_name": "Old Voice Name", "new_name": "New Voice Name"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Voice 'Old Voice Name' renamed to 'New Voice Name' successfully."

    # Verify the name change by getting voices
    voices_response = authenticated_test_client.get("/api/voices")
    voices = voices_response.json()
    assert any(v["name"] == "New Voice Name" for v in voices)
    assert not any(v["name"] == "Old Voice Name" for v in voices)

def test_rename_voice_not_found(authenticated_test_client: TestClient):
    """Test renaming a non-existent voice."""
    response = authenticated_test_client.put(
        "/api/voices/rename",
        json={"old_name": "Non Existent Voice", "new_name": "Some New Name"}
    )
    assert response.status_code == 404
    assert "Voice 'Non Existent Voice' not found for this user." in response.json()["detail"]

def test_rename_voice_to_existing_name(authenticated_test_client: TestClient, create_dummy_audio_file):
    """Test renaming a voice to a name that already exists for the user."""
    # Upload two voices
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio_1.wav", f, "audio/wav")},
            data={"voice_name": "Voice One", "speaker_lang": "en"}
        )
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio_2.wav", f, "audio/wav")},
            data={"voice_name": "Voice Two", "speaker_lang": "en"}
        )
    
    # Try to rename Voice One to Voice Two
    response = authenticated_test_client.put(
        "/api/voices/rename",
        json={"old_name": "Voice One", "new_name": "Voice Two"}
    )
    assert response.status_code == 409
    assert "Voice with name 'Voice Two' already exists for this user." in response.json()["detail"]

def test_rename_voice_unauthenticated(test_client: TestClient):
    """Test renaming a voice without authentication."""
    response = test_client.put(
        "/api/voices/rename",
        json={"old_name": "Any Voice", "new_name": "New Name"}
    )
    assert response.status_code == 403
    assert "Not authenticated" in response.json()["detail"]

def test_delete_voice_success(authenticated_test_client: TestClient, create_dummy_audio_file):
    """Test successful deletion of a speaker voice."""
    # Upload a voice first to get its filename
    with open(DUMMY_AUDIO_PATH, "rb") as f:
        upload_response = authenticated_test_client.post(
            "/api/voices/upload",
            files={"file": ("dummy_audio.wav", f, "audio/wav")},
            data={"voice_name": "Voice to Delete", "speaker_lang": "en"}
        )
    
    # Get the filename from the metadata (assuming default user ID is 1)
    from backend.main import _read_speaker_voices_metadata
    metadata = _read_speaker_voices_metadata()
    voice_entry = next((v for v in metadata if v.get("user_id") == 1 and v.get("name") == "Voice to Delete"), None)
    assert voice_entry is not None
    filename_to_delete = voice_entry["filename"]

    response = authenticated_test_client.request(
        "DELETE",
        "/api/voices/delete",
        json={"filename": filename_to_delete}
    )
    assert response.status_code == 200
    assert response.json()["message"] == f"Voice '{filename_to_delete}' deleted successfully."

    # Verify deletion by trying to get voices
    voices_response = authenticated_test_client.get("/api/voices")
    voices = voices_response.json()
    assert not any(v["filename"] == filename_to_delete for v in voices)

def test_delete_voice_not_found(authenticated_test_client: TestClient):
    """Test deleting a non-existent voice file."""
    response = authenticated_test_client.request(
        "DELETE",
        "/api/voices/delete",
        json={"filename": "non_existent_voice.wav"}
    )
    assert response.status_code == 404
    assert "Voice file 'non_existent_voice.wav' not found for this user." in response.json()["detail"]

def test_delete_voice_unauthenticated(test_client: TestClient):
    """Test deleting a voice without authentication."""
    response = test_client.request(
        "DELETE",
        "/api/voices/delete",
        json={"filename": "any_voice.wav"}
    )
    assert response.status_code == 403 # Changed from 401 to 403
    assert "Not authenticated" in response.json()["detail"] # The detail message might also change, but 403 is the expected status
