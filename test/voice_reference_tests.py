"""Local references: real format decoding, limits, safe names and synthesis selection."""
import io
import subprocess
import wave
from pathlib import Path

import pytest
import backend.main as bm


def wav(seconds=0.1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * int(seconds * 16000))
    return buffer.getvalue()


@pytest.fixture
def headers(test_client):
    return {"Authorization": "Bearer " + test_client.post("/api/guest").json()["token"]}


def upload(client, headers, data=None, name="sample.wav"):
    return client.post("/api/voice-reference", headers=headers,
                       files={"file": (name, wav() if data is None else data, "audio/wav")})


@pytest.mark.parametrize("suffix,codec", [("wav", "pcm_s16le"), ("m4a", "aac"), ("mp3", "libmp3lame"), ("webm", "libopus")])
def test_supported_formats_round_trip(test_client, headers, suffix, codec, tmp_path):
    source = tmp_path / "input.wav"; source.write_bytes(wav())
    target = tmp_path / ("voice." + suffix)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-c:a", codec, str(target)], check=True, capture_output=True)
    response = upload(test_client, headers, target.read_bytes(), target.name)
    assert response.status_code == 200, response.text
    name = response.json()["name"]
    assert name.endswith("." + suffix)
    assert test_client.get("/api/voice-reference", headers=headers).json() == [{"name": name}]
    assert Path(bm.SPEAKER_VOICES_DIR, name).read_bytes() == target.read_bytes()
    assert bm.authorize_speaker_wav(name) == str(Path(bm.SPEAKER_VOICES_DIR, name))
    other = {"Authorization": "Bearer " + test_client.post("/api/guest").json()["token"]}
    assert test_client.get("/api/voice-reference", headers=other).json() == [{"name": name}]
    assert test_client.delete("/api/voice-reference", headers=other, params={"name": name}).status_code == 200
    assert test_client.get("/api/voice-reference", headers=headers).json() == []


def test_reference_selected_by_exact_filename_for_synthesis(test_client, headers, monkeypatch):
    from unittest.mock import AsyncMock
    import app
    name = upload(test_client, headers).json()["name"]
    initialize = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(app, "initialize_all_models", initialize)
    response = test_client.post("/initialize", headers=headers, params={"source_lang": "en", "target_lang": "sk", "tts_model_choice": "omnivoice", "speaker_wav_path": name})
    assert response.status_code == 200
    assert initialize.call_args.kwargs["speaker_wav_path"] == str(Path(bm.SPEAKER_VOICES_DIR, name))


def test_five_file_limit_and_no_duplicate_overwrites(test_client, headers):
    names = [upload(test_client, headers).json()["name"] for _ in range(5)]
    assert len(set(names)) == 5
    assert upload(test_client, headers).status_code == 409
    assert len(test_client.get("/api/voice-reference", headers=headers).json()) == 5
    assert test_client.delete("/api/voice-reference", headers=headers, params={"name": names[0]}).status_code == 200
    assert upload(test_client, headers).status_code == 200


@pytest.mark.parametrize("seconds,status", [(60, 200), (60.01, 413), (80, 413)])
def test_duration_limit(test_client, headers, seconds, status):
    assert upload(test_client, headers, wav(seconds)).status_code == status
    assert len(list(Path(bm.SPEAKER_VOICES_DIR).iterdir())) == (1 if status == 200 else 0)


def test_size_limit_does_not_store_partial_upload(test_client, headers):
    assert upload(test_client, headers, b"x" * (15 * 1024 * 1024 + 1)).status_code == 413
    assert list(Path(bm.SPEAKER_VOICES_DIR).iterdir()) == []


@pytest.mark.parametrize("name,data,status", [("bad.exe", wav(), 415), ("bad.ogg", wav(), 415), ("fake.wav", b"not audio", 422), ("empty.wav", b"", 422), ("disguised.mp3", wav(), 422)])
def test_invalid_file_rejected(test_client, headers, name, data, status):
    assert upload(test_client, headers, data, name).status_code == status
    assert list(Path(bm.SPEAKER_VOICES_DIR).iterdir()) == []


def test_upload_name_is_sanitized(test_client, headers):
    response = upload(test_client, headers, name="../../bad/<voice>?.wav")
    assert response.status_code == 200
    name = response.json()["name"]
    assert name.startswith("voice_") and name.endswith(".wav")
    assert Path(bm.SPEAKER_VOICES_DIR, name).is_file()


@pytest.mark.parametrize("name", ["../outside.wav", "..\\outside.wav", "/tmp/outside.wav", "a/../voice.wav", "..", ""])
def test_selection_and_delete_traversal_rejected(test_client, headers, name):
    assert test_client.delete("/api/voice-reference", headers=headers, params={"name": name}).status_code == 400
    response = test_client.post("/initialize", headers=headers, params={"source_lang": "en", "target_lang": "de", "tts_model_choice": "piper", "speaker_wav_path": name})
    if name:
        assert response.status_code in (400, 403)


def test_symlink_reference_cannot_read_or_remove_external_file(test_client, headers, tmp_path):
    outside = tmp_path / "outside.wav"; outside.write_bytes(wav())
    Path(bm.SPEAKER_VOICES_DIR, "link.wav").symlink_to(outside)
    assert test_client.get("/api/voice-reference", headers=headers).json() == []
    assert test_client.delete("/api/voice-reference", headers=headers, params={"name": "link.wav"}).status_code == 400
    assert outside.exists()


def test_voice_operations_require_valid_session(test_client):
    assert test_client.get("/api/voice-reference").status_code in (401, 403)
    assert upload(test_client, {}).status_code in (401, 403)
    assert test_client.delete("/api/voice-reference", params={"name": "test.wav"}).status_code in (401, 403)
