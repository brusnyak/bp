import pytest
import os
import sys
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session # Import Session, create_engine, sessionmaker
from fastapi.testclient import TestClient # Import TestClient

# Add the backend directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

import backend.utils.db_manager # Import the db_manager module

from backend.utils.db_manager import Base, User # Import Base and User from db_manager
from backend.utils.auth import get_password_hash, verify_password, decode_access_token
from app import create_app
from backend.main import get_db # Import get_db from main.py

API = "/api"  # router is mounted under /api (app.py) — bare paths 404

def fresh(prefix: str) -> str:
    """Unique per-test identity: kills cross-test and cross-run DB pollution."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def test_register_user_success(test_client: TestClient):
    """Test successful user registration."""
    client = test_client
    response = client.post(f"{API}/register", json={"username": fresh("tu_reg"), "email": f"{fresh('tu_reg')}@x.com", "password": "passpass1"})
    assert response.status_code == 200
    assert response.json() == {"message": "User registered successfully"}

def test_register_user_already_exists(test_client: TestClient):
    """Test registration of an already existing user."""
    client = test_client
    uname, email = fresh("tu_dup"), f"{fresh('tu_dup')}@x.com"
    # Register user first
    client.post(f"{API}/register", json={"username": uname, "email": email, "password": "passpass1"})

    # Attempt to register again
    response = client.post(f"{API}/register", json={"username": uname, "email": email, "password": "passpass1"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Email already registered"} # email check runs first

def test_login_user_success(test_client: TestClient):
    """Test successful user login."""
    client = test_client
    uname, email = fresh("tu_login"), f"{fresh('tu_login')}@x.com"
    # Register user first
    client.post(f"{API}/register", json={"username": uname, "email": email, "password": "passpass1"})

    response = client.post(f"{API}/login", json={"email": email, "password": "passpass1"})
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["message"] == "Login successful"
    assert response_json["username"] == uname
    assert "token" in response_json
    # Real HS256 session JWT (no mock strings): must decode back to the email.
    assert decode_access_token(response_json["token"]) == email

def test_login_user_incorrect_password(test_client: TestClient):
    """Test login with incorrect password."""
    client = test_client
    uname, email = fresh("tu_bad"), f"{fresh('tu_bad')}@x.com"
    # Register user first
    client.post(f"{API}/register", json={"username": uname, "email": email, "password": "passpass1"})

    response = client.post(f"{API}/login", json={"email": email, "password": "wrong"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect email or password"}

def test_login_user_not_found(test_client: TestClient):
    """Test login with a non-existent user."""
    client = test_client
    response = client.post(f"{API}/login", json={"email": f"{fresh('tu_nobody')}@x.com", "password": "passpass1"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect email or password"}

# Test for the /initialize endpoint to ensure it starts without torio/ffmpeg errors
def test_initialize_pipeline_success(test_client: TestClient, monkeypatch):
    """Authenticated HTTP preparation uses an isolated user and a no-model initializer."""
    from unittest.mock import AsyncMock
    import app as app_module
    uname, email = fresh("tu_init"), f"{fresh('tu_init')}@x.com"
    test_client.post(f"{API}/register", json={"username": uname, "email": email, "password": "passpass1"})
    token = test_client.post(f"{API}/login", json={"email": email, "password": "passpass1"}).json()["token"]
    init = AsyncMock(return_value={"status": "success", "message": "Models initialized."})
    monkeypatch.setattr(app_module, "initialize_all_models", init)
    response = test_client.post("/initialize", params={"source_lang": "en", "target_lang": "sk", "tts_model_choice": "piper"},
                                headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    init.assert_awaited_once()


def test_guest_session_on_loopback_and_voice_denied(test_client: TestClient, monkeypatch):
    """Guest demo: token without an account on loopback; cannot password-login; cannot manage voices."""
    monkeypatch.delenv("BP_GUEST", raising=False)
    monkeypatch.delenv("BP_HOST", raising=False)
    assert test_client.get(f"{API}/auth/config").json()["guest_enabled"] is True
    r = test_client.post(f"{API}/guest")
    assert r.status_code == 200 and r.json()["guest"] is True
    token = r.json()["token"]
    assert decode_access_token(token) == "guest@guest.local"
    assert test_client.post(f"{API}/login", json={"email": "guest@guest.local", "password": ""}).status_code == 401
    up = test_client.post(f"{API}/voices/upload", headers={"Authorization": f"Bearer {token}"},
                          files={"file": ("a.wav", b"RIFF", "audio/wav")}, data={"voice_name": "x", "speaker_lang": "en"})
    assert up.status_code == 403


def test_guest_session_default_on_short_lived_and_switchable(test_client: TestClient, monkeypatch):
    """Anyone can try the demo without an account (even on an exposed host); tokens last 2 h; BP_GUEST=0 disables."""
    import jwt as _jwt, time as _time
    monkeypatch.setenv("BP_HOST", "0.0.0.0")
    monkeypatch.delenv("BP_GUEST", raising=False)
    r = test_client.post(f"{API}/guest")
    assert r.status_code == 200
    payload = _jwt.decode(r.json()["token"], options={"verify_signature": False})
    assert payload["exp"] - payload["iat"] <= 2 * 3600 + 1 and payload["exp"] > _time.time()
    monkeypatch.setenv("BP_GUEST", "0")
    assert test_client.post(f"{API}/guest").status_code == 403
    assert test_client.get(f"{API}/auth/config").json()["guest_enabled"] is False


def test_register_server_side_validation(test_client: TestClient):
    """Password length, email shape and the reserved guest address are enforced on the server, not only in the browser."""
    base = {"username": fresh("tu_val"), "email": f"{fresh('tu_val')}@x.com", "password": "passpass1"}
    assert test_client.post(f"{API}/register", json={**base, "password": "short"}).status_code == 422
    assert test_client.post(f"{API}/register", json={**base, "email": "not-an-email"}).status_code == 422
    assert test_client.post(f"{API}/register", json={**base, "email": "guest@guest.local"}).status_code == 400
    assert test_client.post(f"{API}/register", json=base).status_code == 200


def test_voice_upload_cap_stops_reading(test_client: TestClient, monkeypatch):
    """An oversized upload is refused with 413 once the cap is crossed (chunked read)."""
    import backend.main as bm
    monkeypatch.setattr(bm, "MAX_VOICE_UPLOAD_BYTES", 1024)
    email = f"{fresh('tu_up')}@x.com"
    test_client.post(f"{API}/register", json={"username": fresh("tu_up"), "email": email, "password": "passpass1"})
    token = test_client.post(f"{API}/login", json={"email": email, "password": "passpass1"}).json()["token"]
    r = test_client.post(f"{API}/voices/upload", headers={"Authorization": f"Bearer {token}"},
                         files={"file": ("a.wav", b"x" * 5000, "audio/wav")}, data={"voice_name": "big", "speaker_lang": "en"})
    assert r.status_code == 413


def test_rate_limit_default_follows_host(monkeypatch):
    from backend.utils.auth import rate_limit_enabled
    for k in ("BP_ENABLE_RATE_LIMIT", "BP_DISABLE_RATE_LIMIT", "BP_HOST"):
        monkeypatch.delenv(k, raising=False)
    assert rate_limit_enabled() is False                      # loopback default
    monkeypatch.setenv("BP_HOST", "0.0.0.0")
    assert rate_limit_enabled() is True                       # exposed server limits by default
    monkeypatch.setenv("BP_DISABLE_RATE_LIMIT", "1")
    assert rate_limit_enabled() is False                      # test switch
    monkeypatch.setenv("BP_ENABLE_RATE_LIMIT", "1")
    assert rate_limit_enabled() is True                       # explicit wins
