import pytest
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session # Import Session, create_engine, sessionmaker
from fastapi.testclient import TestClient # Import TestClient

# Add the backend directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

import backend.utils.db_manager # Import the db_manager module

from backend.utils.db_manager import Base, User # Import Base and User from db_manager
from backend.utils.auth import get_password_hash, verify_password # Import auth functions
from app import app # Import app from app.py
from backend.main import get_db # Import get_db from main.py

def test_register_user_success(test_client: TestClient):
    """Test successful user registration."""
    client = test_client
    response = client.post("/register", json={"username": "unique_testuser_reg_success", "email": "unique_testuser_reg_success@example.com", "password": "pass"})
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "User registered successfully."}

def test_register_user_already_exists(test_client: TestClient):
    """Test registration of an already existing user."""
    client = test_client
    # Register user first
    client.post("/register", json={"username": "unique_existinguser", "email": "unique_existinguser@example.com", "password": "pass"})
    
    # Attempt to register again
    response = client.post("/register", json={"username": "unique_existinguser", "email": "unique_existinguser@example.com", "password": "pass"})
    assert response.status_code == 400
    assert response.json() == {"status": "error", "message": "Username already registered"} # Expect username error first

def test_login_user_success(test_client: TestClient):
    """Test successful user login."""
    client = test_client
    # Register user first
    client.post("/register", json={"username": "unique_testuser_login_success", "email": "unique_testuser_login_success@example.com", "password": "pass"})
    
    response = client.post("/login", json={"email": "unique_testuser_login_success@example.com", "password": "pass"})
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "success"
    assert response_json["message"] == "Login successful."
    assert "token" in response_json
    assert response_json["token"].startswith("mock-jwt-token-for-")

def test_login_user_incorrect_password(test_client: TestClient):
    """Test login with incorrect password."""
    client = test_client
    # Register user first
    client.post("/register", json={"username": "unique_testuser_login_incorrect", "email": "unique_testuser_login_incorrect@example.com", "password": "pass"})
    
    response = client.post("/login", json={"email": "unique_testuser_login_incorrect@example.com", "password": "wrong"})
    assert response.status_code == 401
    assert response.json() == {"status": "error", "message": "Incorrect email or password"}

def test_login_user_not_found(test_client: TestClient):
    """Test login with a non-existent user."""
    client = test_client
    response = client.post("/login", json={"email": "unique_nonexistentuser@example.com", "password": "pass"})
    assert response.status_code == 401
    assert response.json() == {"status": "error", "message": "Incorrect email or password"}

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
