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
from app import app # Import app from app.py
from backend.main import get_db # Import get_db from main.py

API = "/api"  # router is mounted under /api (app.py) — bare paths 404

def fresh(prefix: str) -> str:
    """Unique per-test identity: kills cross-test and cross-run DB pollution."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def test_register_user_success(test_client: TestClient):
    """Test successful user registration."""
    client = test_client
    response = client.post(f"{API}/register", json={"username": fresh("tu_reg"), "email": f"{fresh('tu_reg')}@x.com", "password": "pass"})
    assert response.status_code == 200
    assert response.json() == {"message": "User registered successfully"}

def test_register_user_already_exists(test_client: TestClient):
    """Test registration of an already existing user."""
    client = test_client
    uname, email = fresh("tu_dup"), f"{fresh('tu_dup')}@x.com"
    # Register user first
    client.post(f"{API}/register", json={"username": uname, "email": email, "password": "pass"})

    # Attempt to register again
    response = client.post(f"{API}/register", json={"username": uname, "email": email, "password": "pass"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Email already registered"} # email check runs first

def test_login_user_success(test_client: TestClient):
    """Test successful user login."""
    client = test_client
    uname, email = fresh("tu_login"), f"{fresh('tu_login')}@x.com"
    # Register user first
    client.post(f"{API}/register", json={"username": uname, "email": email, "password": "pass"})

    response = client.post(f"{API}/login", json={"email": email, "password": "pass"})
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
    client.post(f"{API}/register", json={"username": uname, "email": email, "password": "pass"})

    response = client.post(f"{API}/login", json={"email": email, "password": "wrong"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect email or password"}

def test_login_user_not_found(test_client: TestClient):
    """Test login with a non-existent user."""
    client = test_client
    response = client.post(f"{API}/login", json={"email": f"{fresh('tu_nobody')}@x.com", "password": "pass"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect email or password"}

# Test for the /initialize endpoint to ensure it starts without torio/ffmpeg errors
def test_initialize_pipeline_success(test_client: TestClient):
    """Test successful initialization of the pipeline."""
    client = test_client
    # This test primarily checks if the endpoint can be called and returns a success status,
    # indicating that the backend successfully attempted to initialize models without crashing.
    response = client.post(
        "/initialize",
        params={
            "source_lang": "en",
            "target_lang": "sk",
            "tts_model_choice": "piper",
            "vad_enabled_param": True
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["message"] == "Models initialization triggered."
