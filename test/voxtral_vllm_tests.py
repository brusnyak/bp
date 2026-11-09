"""Model-free test of the vLLM realtime client (backend/stt/voxtral_vllm_stt.py) against a fake server that speaks the same protocol."""
import json
import threading
import time

import numpy as np
from websockets.sync.server import serve


def _fake_server():
    def handler(ws):
        ws.send(json.dumps({"type": "session.created", "id": "x"}))
        n = 0
        for raw in ws:
            m = json.loads(raw)
            if m["type"] == "input_audio_buffer.append":
                n += 1
                ws.send(json.dumps({"type": "transcription.delta", "delta": [" good", " morning", " and"][min(n - 1, 2)]}))
            elif m["type"] == "input_audio_buffer.commit" and m.get("final"):
                ws.send(json.dumps({"type": "transcription.delta", "delta": " welcome."}))
                ws.send(json.dumps({"type": "transcription.done", "text": "good morning and welcome."}))
                return
    srv = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.socket.getsockname()[1]


def test_stream_feed_returns_whole_words_and_flushes_on_final(monkeypatch):
    srv, port = _fake_server()
    try:
        monkeypatch.setenv("BP_VOXTRAL_URL", f"ws://127.0.0.1:{port}/v1/realtime")
        from backend.stt.voxtral_vllm_stt import VoxtralVllmSTT
        eng = VoxtralVllmSTT()
        st = eng.open_stream()
        one_chunk = np.zeros(2048, dtype=np.float32)            # 4096 PCM16 bytes = one append event
        got = []
        for _ in range(3):
            got += eng.stream_feed(st, one_chunk)
            time.sleep(0.15)                                    # let the fake server answer; the client must not block waiting
        got += eng.stream_feed(st, np.zeros(0, dtype=np.float32), final=True)
        assert got == ["good", "morning", "and", "welcome."]   # last word held until the next token, flushed at final
    finally:
        srv.shutdown()


def test_missing_server_fails_at_startup_with_the_fix(monkeypatch):
    import pytest
    monkeypatch.setenv("BP_VOXTRAL_URL", "ws://127.0.0.1:9/v1/realtime")
    from backend.stt.voxtral_vllm_stt import VoxtralVllmSTT
    with pytest.raises(RuntimeError, match="vllm serve"):
        VoxtralVllmSTT()
