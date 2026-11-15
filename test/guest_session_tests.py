"""Anonymous sessions: persistence, expiry, isolation and exposed-host limits."""
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi import HTTPException
from starlette.websockets import WebSocketDisconnect
import backend.utils.auth as auth


def test_guests_distinct_short_lived_and_always_enabled(test_client, monkeypatch):
    monkeypatch.setenv("BP_GUEST", "0")  # retired switch cannot disable local operation
    monkeypatch.setenv("BP_HOST", "0.0.0.0")
    guests = [test_client.post("/api/guest").json() for _ in range(3)]
    assert len({g["guest_id"] for g in guests}) == 3
    for guest in guests:
        assert auth.decode_access_token(guest["token"]) == guest["guest_id"]
        claims = jwt.decode(guest["token"], options={"verify_signature": False})
        assert claims["exp"] - claims["iat"] == 7200
        assert claims["exp"] > time.time()
        assert uuid.UUID(guest["guest_id"]).hex == guest["guest_id"]


def test_guest_expiry_rejected_over_http_and_all_socket_modes(test_client):
    token = auth.create_access_token(uuid.uuid4().hex, hours=-1)
    with pytest.raises(HTTPException) as error:
        auth.decode_access_token(token)
    assert error.value.status_code == 401
    headers = {"Authorization": f"Bearer {token}"}
    assert test_client.post("/api/room", headers=headers).status_code == 401
    assert test_client.get("/api/voice-reference", headers=headers).status_code == 401
    for path in ("/ws", "/ws/meeting", "/ws/listen?room=nope&lang=de"):
        with pytest.raises(WebSocketDisconnect):
            with test_client.websocket_connect(path + ("&" if "?" in path else "?") + "token=" + token) as ws:
                ws.receive_json()


@pytest.mark.parametrize("claims", [
    {"sub": "1" * 32},
    {"sub": "guest@guest.local", "iat": 1, "exp": 9999999999},
    {"sub": "1" * 32, "iat": 1},
])
def test_tokens_require_valid_id_and_expiry(claims):
    token = jwt.encode(claims, auth._jwt_secret(), algorithm="HS256")
    with pytest.raises(HTTPException):
        auth.decode_access_token(token)


def test_first_start_secret_is_atomic_and_private(tmp_path, monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setattr(auth, "JWT_SECRET_FILE", tmp_path / ".jwt_secret")
    with ThreadPoolExecutor(max_workers=8) as pool:
        secrets = list(pool.map(lambda _: auth._jwt_secret(), range(24)))
    assert len(set(secrets)) == 1
    assert auth.JWT_SECRET_FILE.stat().st_mode & 0o777 == 0o600
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == [".jwt_secret"]
    # A fresh interpreter uses the same file and accepts the previous token.
    token = auth.create_access_token("2" * 32)
    code = "from pathlib import Path; import backend.utils.auth as a; a.JWT_SECRET_FILE=Path(__import__('sys').argv[1]); assert a.decode_access_token(__import__('sys').argv[2]) == '2'*32"
    subprocess.run([sys.executable, "-c", code, str(auth.JWT_SECRET_FILE), token], check=True, capture_output=True)


def test_environment_secret_creates_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "env-only-private-test-secret-32-chars")
    monkeypatch.setattr(auth, "JWT_SECRET_FILE", tmp_path / ".jwt_secret")
    assert auth._jwt_secret() == os.environ["JWT_SECRET"]
    assert not auth.JWT_SECRET_FILE.exists()


def test_removed_routes_and_pages_are_gone(test_client):
    for path in ("/api/login", "/api/register", "/api/token", "/api/auth/google", "/api/voices/upload", "/api/voices/process"):
        assert test_client.post(path, json={}).status_code == 404
    for path in ("/api/auth/config", "/ui/auth/auth.html", "/ui/enroll/enroll.html", "/ui/auth/session-nav.js"):
        assert test_client.get(path).status_code == 404


def test_guest_preparation_without_models(test_client, monkeypatch):
    import app
    initialize = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(app, "initialize_all_models", initialize)
    token = test_client.post("/api/guest").json()["token"]
    response = test_client.post("/initialize", params={"source_lang": "en", "target_lang": "de", "tts_model_choice": "piper"},
                                headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    initialize.assert_awaited_once()


def test_guest_owns_room_and_another_guest_cannot_publish(test_client):
    from backend.room import get_room
    speaker = test_client.post("/api/guest").json()
    listener = test_client.post("/api/guest").json()
    made = test_client.post("/api/room", headers={"Authorization": f"Bearer {speaker['token']}"}).json()
    room = get_room(made["room_id"], create=False)
    assert room.owner == speaker["guest_id"] != listener["guest_id"]
    assert test_client.get(f"/api/room/{room.room_id}/qr", headers={"Authorization": f"Bearer {listener['token']}"}).status_code == 404
    with test_client.websocket_connect(f"/ws?token={listener['token']}") as ws:
        ws.receive_json()
        ws.send_json({"type": "config_update", "room_id": room.room_id, "source_lang": "en", "target_lang": "de"})
        error = ws.receive_json()
        assert error["type"] == "error" and "room" in error["message"].lower()
    assert room.owner == speaker["guest_id"]


def test_exposed_host_rate_limit_is_enforced():
    # Decorators are configured at import time, so test a fresh process as on startup.
    code = '''
from fastapi.testclient import TestClient
from app import create_app
with TestClient(create_app()) as c:
    statuses = [c.post('/api/guest').status_code for _ in range(21)]
    assert statuses[:20] == [200]*20, statuses
    assert statuses[-1] == 429, statuses
'''
    env = dict(os.environ, BP_HOST="0.0.0.0", BP_ENABLE_RATE_LIMIT="1", BP_DISABLE_RATE_LIMIT="0")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
