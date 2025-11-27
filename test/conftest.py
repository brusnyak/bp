import pytest
import os
import sys
import time
import subprocess
import threading
import uvicorn
import requests
import httpx
import asyncio # Import asyncio for async operations
import shutil # Import shutil for directory removal
from fastapi.testclient import TestClient

# Add the backend directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from app import create_app # Import the app-creating function
from backend.main import get_db_dependency # Import get_db_dependency
from backend.utils.db_manager import Base, get_db_session_and_engine, SQLALCHEMY_DATABASE_URL, User # Import User

# Setup in-memory SQLite database URL for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session", autouse=True)
def app_with_test_db():
    """
    Configures the backend.database to use an in-memory SQLite database for all tests
    and ensures tables are created and dropped once per test session.
    This fixture runs before any tests are collected and before backend.main is fully loaded.
    It also creates and configures the FastAPI app instance.
    """
    print("\nSetting up in-memory database for tests...")
    # Create a test engine and session for the entire test session
    test_engine, TestSessionLocal = get_db_session_and_engine(SQLALCHEMY_DATABASE_URL)
    
    # Create tables for the User model in the test database
    Base.metadata.create_all(bind=test_engine)
    
    # Override the get_db dependency for the FastAPI app to use the test session
    def get_test_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    # Create the app instance using the factory function, passing the test session local
    _app = create_app(db_session_local_override=TestSessionLocal)
    # The dependency override for get_db_dependency is now handled within create_app
    # No need to explicitly override it here anymore.

    yield _app # Yield the configured app instance

    print("\nResetting in-memory database after tests...")
    # Clean up the database after all tests in the session are done
    Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_output_dir():
    """Ensures the test_output directory exists before tests and cleans it up after."""
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    yield
    # Clean up the entire directory after tests
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"\nCleaned up test_output directory: {output_dir}")

@pytest.fixture(scope="session")
def live_server_url():
    """Starts the FastAPI application in a separate process and yields its URL."""
    host = "127.0.0.1"
    port = 8000
    url = f"https://{host}:{port}" # Use https as the backend uses SSL

    # Command to run the FastAPI app using uvicorn
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host", host,
        "--port", str(port),
        "--ssl-keyfile", "./certs/key.pem",
        "--ssl-certfile", "./certs/cert.pem",
        "--log-level", "info",
    ]

    process = None
    try:
        # Get current environment variables
        env = os.environ.copy()
        # Set DYLD_LIBRARY_PATH for macOS to include Homebrew's FFmpeg lib directory
        ffmpeg_lib_path = "/opt/homebrew/opt/ffmpeg/lib"
        if "DYLD_LIBRARY_PATH" in env:
            env["DYLD_LIBRARY_PATH"] = f"{ffmpeg_lib_path}:{env['DYLD_LIBRARY_PATH']}"
        else:
            env["DYLD_LIBRARY_PATH"] = ffmpeg_lib_path

        # Set PYTORCH_MPS_HIGH_WATERMARK_RATIO to disable memory allocation limit for MPS
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Start the server in a separate process with modified environment
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        print(f"\nStarted FastAPI server with PID: {process.pid}")

        # Wait for the server to start up by looking for a specific log message
        timeout = 120 # Increased timeout for server startup
        start_time = time.time()
        server_ready = False
        while time.time() - start_time < timeout:
            output = process.stdout.readline().decode().strip()
            if output:
                print(f"Server output: {output}")
                if "Application startup complete." in output or "Uvicorn running on" in output:
                    server_ready = True
                    break
            time.sleep(0.1) # Small delay to avoid busy-waiting

        if not server_ready:
            raise RuntimeError("FastAPI server did not start in time.")
        print(f"FastAPI server is running at {url}")

        yield url
    finally:
        if process:
            print(f"Terminating FastAPI server with PID: {process.pid}")
            process.terminate()
            process.wait(timeout=20) # Increased timeout for graceful shutdown
            if process.poll() is None:
                print(f"Killing FastAPI server with PID: {process.pid}")
                process.kill()
            print("FastAPI server terminated.")

@pytest.fixture(scope="function")
def test_client(app_with_test_db):
    """Provides a TestClient instance for each test function."""
    with TestClient(app=app_with_test_db) as client:
        yield client

# Helper function to initialize models via HTTP endpoint
async def _initialize_pipeline(live_server_url: str, stt_model_size: str = "tiny", vad_enabled: bool = False):
    """Initializes the backend models via the /initialize HTTP endpoint."""
    init_payload = {
        "source_lang": "en",
        "target_lang": "sk",
        "tts_model_choice": "piper",
        "stt_model_size": stt_model_size,
        "vad_enabled_param": vad_enabled,
        "speaker_wav_path": None,
        "speaker_text": None,
        "speaker_lang": None
    }
    headers = {"Authorization": "Bearer mock-jwt-token-for-test@example.com"}
    async with httpx.AsyncClient(verify=False, headers=headers) as http_client:
        init_response = await http_client.post(f"{live_server_url}/initialize", params=init_payload)
        init_response.raise_for_status()
        print(f"Backend initialized with {stt_model_size} model: {init_response.json()}")
        return init_response.json()
