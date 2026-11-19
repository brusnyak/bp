"""Regression tests for the 2026-09 security audit fixes (see documentation/security_audit_2026-09.md)."""
import os
import time
import uuid
from pathlib import Path

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import backend.main as bm
from backend.utils.auth import decode_access_token

API = "/api"


def test_token_signed_with_old_published_secret_is_rejected():
    forged = jwt.encode({"sub": "test@example.com", "exp": int(time.time()) + 3600}, "dev-only-insecure-secret-change-me", algorithm="HS256")
    with pytest.raises(HTTPException) as e:
        decode_access_token(forged)
    assert e.value.status_code == 401


def test_missing_jwt_secret_persists_random_secret(monkeypatch, tmp_path):
    import backend.utils.auth as auth
    monkeypatch.delenv("JWT_SECRET", raising=False)
    path = tmp_path / ".jwt_secret"
    monkeypatch.setattr(auth, "JWT_SECRET_FILE", path)
    secret = auth._jwt_secret()
    assert secret != "dev-only-insecure-secret-change-me" and len(secret) >= 32
    assert path.read_text() == secret == auth._jwt_secret()
    # POSIX mode bits are not meaningful on Windows/NTFS.
    # Windows ACL protection needs a separate platform-specific test.
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600


def test_voice_path_helpers_block_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "VOICES_DIR", str(tmp_path))
    for bad in ("../x.wav", "..\\x.wav", "/etc/passwd", "sub/../../x.wav"):
        with pytest.raises(HTTPException):
            bm._voice_path(bad)
    assert bm._voice_path("1_ok.wav").startswith(os.path.realpath(tmp_path))
    for bad in ("", "..", "///", "..."):
        with pytest.raises(HTTPException):
            bm._safe_voice_name(bad)
    assert bm._safe_voice_name("my voice_1") == "my voice_1"


def test_speaker_wav_path_limited_to_voices_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "VOICES_DIR", str(tmp_path))
    inside = str(tmp_path / "1_a.wav")
    assert bm._safe_speaker_wav(inside) == inside
    assert bm._safe_speaker_wav(None) is None
    assert bm._safe_speaker_wav(os.path.abspath(__file__)) is None
    assert bm._safe_speaker_wav("../../../../etc/passwd") is None


def test_delete_cannot_escape_voices_dir(test_client, tmp_path):
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"private")
    token = test_client.post("/api/guest").json()["token"]
    for name in ("../outside.wav", "..\\outside.wav", str(outside)):
        response = test_client.delete("/api/voice-reference", params={"name": name},
                                      headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 400
    assert outside.read_bytes() == b"private"


def test_speaker_voices_json_is_not_served(test_client: TestClient):
    probe = Path("speaker_voices") / "_probe_secret.json"
    probe.parent.mkdir(exist_ok=True)
    probe.write_text("{}", encoding="utf-8")
    try:
        assert test_client.get("/speaker_voices/_probe_secret.json").status_code == 404
        assert test_client.get("/speaker_voices/speaker_voices.json").status_code == 404
    finally:
        probe.unlink(missing_ok=True)
