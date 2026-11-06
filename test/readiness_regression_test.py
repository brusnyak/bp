"""No-model regression checks for auth, target voices, privacy and route evidence."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app as server
import backend.main as bm
from backend.registry import target_tts_engine


class FakeDB:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def query(self, *args): return self
    def filter(self, *args): return self
    def first(self): return SimpleNamespace(id=7, email="test@example.invalid")
    def close(self): pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_default_SessionLocal", FakeDB)
    monkeypatch.setattr(bm, "SessionLocal", FakeDB)
    import backend.utils.auth as auth
    def decode(token):
        if token != "valid": raise ValueError("expired")
        return "test@example.invalid"
    monkeypatch.setattr(auth, "decode_access_token", decode)
    return TestClient(server.create_app())


def test_ws_rejects_missing_and_expired_before_model_loading(client, monkeypatch):
    initialize = AsyncMock()
    monkeypatch.setattr(server, "initialize_all_models", initialize)
    for suffix in ("", "?token=expired"):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws" + suffix) as ws:
                ws.receive_json()
    initialize.assert_not_called()


def test_authenticated_connect_waits_for_explicit_initialization(client, monkeypatch):
    initialize = AsyncMock()
    handler = AsyncMock()
    monkeypatch.setattr(server, "initialize_all_models", initialize)
    monkeypatch.setattr(server, "handle_audio_stream", handler)
    with client.websocket_connect("/ws?token=valid") as ws:
        status = ws.receive_json()
        assert status["fully_loaded"] is False
        assert status["route_status"]["status"] == "experimental"
    initialize.assert_not_called()
    handler.assert_awaited_once()


def test_cross_origin_socket_rejected_before_model_loading(client, monkeypatch):
    initialize = AsyncMock()
    monkeypatch.setattr(server, "initialize_all_models", initialize)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws?token=valid", headers={"origin": "https://attacker.invalid"}): pass
    initialize.assert_not_called()


def test_http_initialize_requires_auth_before_model_loading(client, monkeypatch):
    initialize = AsyncMock()
    monkeypatch.setattr(server, "initialize_all_models", initialize)
    response = client.post("/initialize", params={"source_lang": "en", "target_lang": "sk", "tts_model_choice": "piper"})
    assert response.status_code in (401, 403)
    initialize.assert_not_called()


def test_research_library_qc_and_unowned_media_are_not_public(client, monkeypatch):
    monkeypatch.setattr(bm, "_read_speaker_voices_metadata", lambda: [{"filename": "private.wav", "user_id": 8}])
    monkeypatch.setenv("BP_LAB", "1")  # even with the Lab enabled, research media stays private
    for path in ("/ui/voice-lab/library.json", "/voice_qc/private.wav", "/processed/omnivoice/private.wav",
                 "/speaker_voices/private.wav", "/speaker_voices/private.wav?token=valid"):
        assert client.get(path).status_code == 403
    # The Lab itself is development only: hidden unless BP_LAB=1.
    assert client.get("/ui/voice-lab/lab.html").status_code == 200
    monkeypatch.delenv("BP_LAB")
    assert client.get("/ui/voice-lab/lab.html").status_code == 404


def test_speaker_media_requires_explicit_public_receipt_or_owner(client, monkeypatch):
    from starlette.responses import Response
    async def audio(*args): return Response(b"fake-audio", media_type="audio/wav")
    monkeypatch.setattr(server._AudioOnlyStatic, "get_response", audio)
    entries = [{"filename": "owned.wav", "user_id": 7}, {"filename": "old.wav", "user_id": None},
               {"filename": "demo.wav", "public_demo": True}]
    monkeypatch.setattr(bm, "_read_speaker_voices_metadata", lambda: entries)
    assert client.get("/speaker_voices/owned.wav").status_code == 403
    assert client.get("/speaker_voices/owned.wav?token=valid").status_code == 200
    assert client.get("/speaker_voices/old.wav").status_code == 403
    assert client.get("/speaker_voices/demo.wav").status_code == 200


def test_mps_is_never_selected_for_ctranslate2(monkeypatch):
    monkeypatch.setenv("BP_DEVICE", "mps")
    assert bm._resolve_device() == "cpu"


def test_automatic_voice_matches_target_and_explicit_choice_preserved():
    assert target_tts_engine("piper", "en") == "piper_generic_en"
    assert target_tts_engine("piper", "sk") == "piper_sk_personal"
    assert target_tts_engine("piper", "cs") == "piper"
    assert target_tts_engine("custom", "en") == "custom"
    pack = json.loads(Path("languages/packs/cs-en.json").read_text())
    assert pack["tts"]["voice"].startswith("en_")


def packs_module():
    spec = importlib.util.spec_from_file_location("pack_evidence", "scripts/language_packs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_wer_and_changed_configuration_never_verify_route(tmp_path, monkeypatch):
    mod = packs_module()
    monkeypatch.setattr(mod, "EVAL_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "resolve", lambda p: {"resolved": True})
    (tmp_path / "cs-en.json").write_text(json.dumps({"status": "verified", "wer": 0.01}))
    assert mod.status({"pair": "cs-en", "tts": {"voice": "cs_CZ-jirka-medium"}}) == "experimental"


def test_full_route_record_binds_reviews_fingerprint_and_output(tmp_path, monkeypatch):
    mod = packs_module()
    monkeypatch.setattr(mod, "EVAL_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "resolve", lambda p: {"resolved": True})
    monkeypatch.setattr(mod, "evidence_fingerprint", lambda p, r: p["fingerprint"])
    audio = tmp_path / "output.wav"; audio.write_bytes(b"recorded-output")
    record = {"schema": 2, "acceptance": "full_route_verified", "fingerprint": "current", "reference_sha256": "ref",
              "quality_review": {"reviewer": "bilingual-test-reviewer", "reference_sha256": "ref",
               **{k: "pass" for k in ("stt", "mt_meaning", "tts_intelligibility", "live_latency", "runtime_policy")}},
              "audio_out": str(audio), "audio_sha256": mod._sha256(str(audio))}
    (tmp_path / "cs-en.json").write_text(json.dumps(record))
    assert mod.status({"pair": "cs-en", "fingerprint": "current"}) == "verified"
    assert mod.status({"pair": "cs-en", "fingerprint": "changed"}) == "experimental"
    audio.write_bytes(b"different-output")
    assert mod.status({"pair": "cs-en", "fingerprint": "current"}) == "experimental"


def test_stop_cancels_capture_and_discards_later_bytes(monkeypatch):
    import backend.utils.auth as auth
    monkeypatch.setattr(auth, "decode_access_token", lambda token: "test@example.invalid")
    monkeypatch.setattr(bm, "SessionLocal", FakeDB)
    class Socket:
        scope = {"client": ("test-stop", 1)}
        sent = []
        async def receive(self): return self.messages.pop(0)
        async def send_text(self, text): self.sent.append(json.loads(text))
    socket = Socket()
    socket.messages = [{"type": "websocket.receive", "text": json.dumps({"type": t})} for t in ("start", "stop")]
    socket.messages.extend([{"type": "websocket.receive", "bytes": b"invalid-audio-after-stop"}, {"type": "websocket.disconnect", "code": 1000}])
    asyncio.run(bm.handle_audio_stream(socket, token="valid"))
    assert {"type": "session_stopped"} in socket.sent
    assert "test-stop:1" not in bm.active_sessions


def test_sdk_pools_beam_and_target_voice_separately(monkeypatch):
    import backend.pipeline as facade
    import backend.main as bm
    import faster_whisper.utils as fw_utils
    monkeypatch.setattr(fw_utils, "download_model", lambda model, **kw: "installed-fake-model")
    calls = []
    def instantiate(stage, engine, **cfg):
        calls.append((stage, engine, cfg))
        return SimpleNamespace(beam_size=cfg.get("beam_size"))
    monkeypatch.setattr(facade, "instantiate", instantiate)
    monkeypatch.setattr(bm, "_model_pool", {})
    monkeypatch.setattr(bm, "_model_pool_locks", {})
    async def check():
        en = facade.Pipeline(facade.PipelineConfig("sk", "en", beam_size=1))
        sk = facade.Pipeline(facade.PipelineConfig("en", "sk", beam_size=5))
        await en.initialize(); await sk.initialize()
        assert en._tts is not sk._tts
        assert en._stt is not sk._stt
        assert en._stt.beam_size == 1
        assert sk._stt.beam_size == 5
        assert ("tts", "piper_generic_en") in [(a, b) for a, b, c in calls]
        assert ("tts", "piper_sk_personal") in [(a, b) for a, b, c in calls]
        await en.close(); await sk.close()
        assert all(e["ref_count"] == 0 for e in bm._model_pool.values())
    asyncio.run(check())


def test_client_voice_reference_rejects_other_users(monkeypatch, tmp_path):
    from fastapi import HTTPException
    monkeypatch.setattr(bm, "SPEAKER_VOICES_DIR", str(tmp_path))
    monkeypatch.setattr(bm, "_read_speaker_voices_metadata", lambda: [{"filename": "mine.wav", "user_id": 7}])
    assert bm.authorize_speaker_wav(str(tmp_path / "mine.wav"), 7)
    with pytest.raises(HTTPException): bm.authorize_speaker_wav(str(tmp_path / "mine.wav"), 8)
    with pytest.raises(HTTPException): bm.authorize_speaker_wav(str(tmp_path / "unknown.wav"), 7)


def test_runtime_missing_stt_requests_local_cache_only(monkeypatch):
    from fastapi import HTTPException
    import faster_whisper.utils as fw_utils
    seen = []
    def download(model, **kwargs):
        seen.append(kwargs)
        raise FileNotFoundError("not installed")
    monkeypatch.setattr(fw_utils, "download_model", download)
    with pytest.raises(HTTPException) as exc:
        bm.local_stt_source("missing-audit-model")
    assert exc.value.status_code == 409
    assert seen == [{"local_files_only": False}]


def test_runtime_missing_mt_does_not_convert(monkeypatch):
    from fastapi import HTTPException
    convert = AsyncMock()
    monkeypatch.setattr(bm, "_convert_mt_model", convert)
    monkeypatch.setattr(bm, "get_mt_model_name", lambda *a: "Helsinki-NLP/audit-nonexistent")
    session = {"client_info": "test", "mt_models": {}, "session_config": {}}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(bm._initialize_mt_model(session, "sk", "en"))
    assert exc.value.status_code == 409
    convert.assert_not_called()


def test_optional_captioner_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("BP_ENABLE_STREAMING_CAPTIONS", raising=False)
    monkeypatch.setattr(bm, "STREAMING_CAPTIONS_AVAILABLE", True)
    captioner = AsyncMock()
    monkeypatch.setattr(bm, "SessionCaptioner", captioner)
    asyncio.run(bm._initialize_captioner({"client_info": "test"}, "en"))
    captioner.assert_not_called()


def test_mt_tokenizer_cannot_fetch_network(monkeypatch, tmp_path):
    import backend.mt.ctranslate2_mt as mt
    (tmp_path / "model.bin").write_bytes(b"mock-model")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr(mt.ctranslate2, "Translator", lambda *a, **k: object())
    seen = []
    monkeypatch.setattr(mt.AutoTokenizer, "from_pretrained", lambda *a, **k: seen.append(k))
    mt.CTranslate2MT(model_path=str(tmp_path), device="cpu")
    assert seen == [{"local_files_only": True}]


def test_security_headers_and_no_text_in_logs(client):
    r = client.get("/ui/home/home.html")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert "microphone=(self)" in r.headers["Permissions-Policy"]
    import backend.main as bm_
    assert bm_._log_text("secret words") == "<12 chars>"


def test_home_page_renders_with_switcher_and_guest_first_copy(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Try it now" in r.text and "/ui/i18n/i18n.js" in r.text
    assert "googleapis" not in r.text and "cdnjs" not in r.text and "jsdelivr" not in r.text
