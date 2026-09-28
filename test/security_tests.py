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
from backend.utils.db_manager import User

API = "/api"


def _user_id(app, email: str) -> int:
    db = next(app.dependency_overrides[bm.get_db_dependency]())
    try:
        return db.query(User).filter(User.email == email).one().id
    finally:
        db.close()


def test_token_signed_with_old_published_secret_is_rejected():
    forged = jwt.encode({"sub": "test@example.com", "exp": int(time.time()) + 3600}, "dev-only-insecure-secret-change-me", algorithm="HS256")
    with pytest.raises(HTTPException) as e:
        decode_access_token(forged)
    assert e.value.status_code == 401


def test_missing_jwt_secret_falls_back_to_random_per_process_secret(monkeypatch):
    import backend.utils.auth as auth
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setattr(auth, "_ephemeral_secret", None)
    secret = auth._jwt_secret()
    assert secret != "dev-only-insecure-secret-change-me" and len(secret) >= 32
    assert auth._jwt_secret() == secret  # stable within the process


async def test_demo_user_is_opt_in(monkeypatch):
    from sqlalchemy.orm import sessionmaker
    from app import create_default_user_if_empty
    from backend.utils.db_manager import Base, get_db_session_and_engine

    engine, Session = get_db_session_and_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    monkeypatch.delenv("BP_DEMO_USER", raising=False)
    await create_default_user_if_empty(Session)
    with Session() as db:
        assert db.query(User).count() == 0  # no well-known test@example.com / password account
    monkeypatch.setenv("BP_DEMO_USER", "1")
    await create_default_user_if_empty(Session)
    with Session() as db:
        assert db.query(User).count() == 1


def test_voice_path_helpers_block_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "SPEAKER_VOICES_DIR", str(tmp_path))
    for bad in ("../x.wav", "..\\x.wav", "/etc/passwd", "sub/../../x.wav"):
        with pytest.raises(HTTPException):
            bm._voice_path(bad)
    assert bm._voice_path("1_ok.wav").startswith(os.path.realpath(tmp_path))
    for bad in ("", "..", "///", "..."):
        with pytest.raises(HTTPException):
            bm._safe_voice_name(bad)
    assert bm._safe_voice_name("my voice_1") == "my voice_1"


def test_speaker_wav_path_limited_to_voices_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "SPEAKER_VOICES_DIR", str(tmp_path))
    inside = str(tmp_path / "1_a.wav")
    assert bm._safe_speaker_wav(inside) == inside
    assert bm._safe_speaker_wav(None) is None
    assert bm._safe_speaker_wav(os.path.abspath(__file__)) is None
    assert bm._safe_speaker_wav("../../../../etc/passwd") is None


def test_rename_cannot_escape_voices_dir(test_client: TestClient, app_with_test_db, tmp_path, monkeypatch):
    voices = tmp_path / "voices"
    voices.mkdir()
    monkeypatch.setattr(bm, "SPEAKER_VOICES_DIR", str(voices))
    monkeypatch.setattr(bm, "SPEAKER_VOICES_METADATA_FILE", str(voices / "speaker_voices.json"))

    email = f"sec_{uuid.uuid4().hex[:8]}@x.com"
    test_client.post(f"{API}/register", json={"username": email.split("@")[0], "email": email, "password": "pass"})
    token = test_client.post(f"{API}/login", json={"email": email, "password": "pass"}).json()["token"]
    uid = _user_id(app_with_test_db, email)

    fname = f"{uid}_mine.wav"
    (voices / fname).write_bytes(b"RIFF")
    bm._write_speaker_voices_metadata([{"id": "1", "user_id": uid, "name": "mine", "filename": fname, "path": str(voices / fname), "language": "en"}])

    r = test_client.put(f"{API}/voices/rename", json={"old_name": "mine", "new_name": "../../../pwned"},
                        headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert not (tmp_path / "pwned.wav").exists() and not list(tmp_path.glob("*pwned*"))
    assert all(p.parent == voices for p in voices.iterdir())  # everything stayed inside


def test_speaker_voices_json_is_not_served(test_client: TestClient):
    probe = Path("speaker_voices") / "_probe_secret.json"
    probe.parent.mkdir(exist_ok=True)
    probe.write_text("{}", encoding="utf-8")
    try:
        assert test_client.get("/speaker_voices/_probe_secret.json").status_code == 404
        assert test_client.get("/speaker_voices/speaker_voices.json").status_code == 404
    finally:
        probe.unlink(missing_ok=True)
