import pytest
import httpx
import asyncio
import os
import json
import soundfile as sf
import numpy as np
from unittest.mock import patch, AsyncMock
import io

# For testing, we'll use a TestClient
from fastapi.testclient import TestClient
from app import create_app
from backend.main import get_current_session
from backend.utils.auth import GuestSession
app = create_app()
client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def override_session_dependency():
    app.dependency_overrides[get_current_session] = lambda: GuestSession("0" * 32)
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
    session = active_sessions.get("http_init_session_00000000000000000000000000000000")
    assert session is not None
    assert session["stt_model"] is not None
    assert session["tts_engine"] is not None
    assert session["tts_engine_name"] == session["session_config"]["tts_model_choice"] == "piper_generic_sk"


# --- Ratings sync (Voice Lab persistence, 2026-10-01) ---
@pytest.fixture()
def grades_snapshot(tmp_path, monkeypatch):
    """Snapshot processed/ear_grades.json around each ratings test (real file,
    gitignored — same destroy-instead-of-restore lesson as speaker metadata)."""
    import backend.main as m
    real = m.EAR_GRADES_PATH
    backup = None
    if os.path.exists(real):
        with open(real, encoding="utf-8") as f:
            backup = f.read()
    else:  # fresh clone: the store is generated + gitignored (setup.py builds it); build it here so the tests don't depend on the Mac's data
        import subprocess, sys
        subprocess.run([sys.executable, "scripts/grade_library.py", "--apply"], check=True, capture_output=True)
    yield
    if backup is not None:
        with open(real, "w", encoding="utf-8") as f:
            f.write(backup)
    elif os.path.exists(real):
        os.remove(real)


def test_ratings_get_returns_store(grades_snapshot):
    response = client.get("/api/ratings")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("grades"), list)
    assert any(g["name"] == "clean_sk_personal" for g in body["grades"])


def test_ratings_post_merges_ear_slot(grades_snapshot):
    payload = {"name": "clean_sk_personal",
               "ear": {"grade": 1, "keep": 2, "defects": {"steadiness": 3},
                       "note": "test trembling", "sim": 20}}
    response = client.post("/api/ratings", json=payload)
    assert response.status_code == 200
    slot = response.json()["ear"]
    assert slot["grade"] == 1 and slot["keep"] == "kill"
    assert slot["defects"] == {"steadiness": 3}
    # persisted to disk, readable back
    assert client.get("/api/ratings").status_code == 200


def test_ratings_post_unknown_name_404(grades_snapshot):
    response = client.post("/api/ratings", json={"name": "no_such_clip", "ear": {}})
    assert response.status_code == 404


def test_ratings_post_bad_grade_400(grades_snapshot):
    response = client.post("/api/ratings",
                           json={"name": "clean_sk_personal", "ear": {"grade": 9}})
    assert response.status_code == 400


def test_ratings_require_auth():
    saved = dict(app.dependency_overrides)
    app.dependency_overrides.clear()
    try:
        assert client.get("/api/ratings").status_code in (401, 403)
        assert client.post("/api/ratings", json={}).status_code in (401, 403)
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(saved)
