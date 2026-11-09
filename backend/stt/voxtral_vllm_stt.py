"""Voxtral Mini 4B Realtime through a vLLM server (Realtime websocket API): the engine for NVIDIA GPUs.

Measured 2026-11-09 on a Kaggle T4 (kernel scripts/gpu_bench/kaggle_voxtral_vllm, vLLM 0.33.0, --enforce-eager, fp16, --max-model-len 3584):
audio sent at real time, text deltas arrive 0.07-0.08 s median (p90 0.12 s, max 0.67 s) after the audio, no drift; word error 1.0-3.1%.
The Transformers loop on the same T4 drifted by seconds (decode at 1.0-1.1 x real time), CUDA-graph mode did not start.

The vLLM server runs as its own process (its own venv: vLLM pins torch/transformers versions the app does not):
    vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --dtype float16 --max-model-len 3584 --enforce-eager --port 8001
One token is 80 ms, so max-model-len 3584 allows ~4.8 minutes per utterance (the route opens a stream per utterance, <= 20 s). A T4 (15 GB) cannot hold the
default 16384 (needs 5.72 GiB KV cache, 4.37 GiB free).
Same streaming interface as voxtral_stt.py: open_stream / stream_feed / transcribe_audio. BP_VOXTRAL_URL sets the server (default ws://127.0.0.1:8001/v1/realtime)."""
from __future__ import annotations

import base64
import json
import os
import time
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np

from backend.stt.voxtral_stt import LANGUAGES, _Stream  # noqa: F401  (LANGUAGES re-exported)

DEFAULT_URL = "ws://127.0.0.1:8001/v1/realtime"
MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
CHUNK_BYTES = 4096   # raw PCM16 bytes per append event (128 ms), as in vLLM's own client


class _VllmStream(_Stream):
    def __init__(self, ws):
        super().__init__(None)
        self.ws = ws


class VoxtralVllmSTT:
    incr_step_s = float(os.environ.get("BP_VOXTRAL_STEP_S", "0.2"))
    incr_min_audio_s = 0.0

    def __init__(self, model_size: str = "", device: str = "auto", compute_type: str = "", beam_size: int = 1):
        self.url = os.environ.get("BP_VOXTRAL_URL", DEFAULT_URL)
        self.repo = self.model_size = os.environ.get("BP_VOXTRAL_MODEL", MODEL)
        try:   # fail at startup, with the fix, not at the first utterance
            self._close(self._open())
        except Exception as e:
            raise RuntimeError(f"VoxtralVllmSTT: no vLLM realtime server at {self.url} ({type(e).__name__}: {e}). Start one: "
                               f"vllm serve {self.repo} --dtype float16 --max-model-len 3584 --enforce-eager --port 8001") from e
        print(f"VoxtralVllmSTT initialized: {self.url}")

    def _open(self):
        from websockets.sync.client import connect
        ws = connect(self.url, max_size=None, open_timeout=10)
        first = json.loads(ws.recv(timeout=10))
        if first.get("type") != "session.created":
            ws.close()
            raise RuntimeError(f"unexpected first event {first.get('type')!r}")
        ws.send(json.dumps({"type": "session.update", "model": self.repo}))
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))   # "ready": audio follows
        return ws

    @staticmethod
    def _close(ws):
        try:
            ws.close()
        except Exception:
            pass

    @staticmethod
    def _send_pcm(ws, audio: np.ndarray):
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        for i in range(0, len(pcm), CHUNK_BYTES):
            ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[i:i + CHUNK_BYTES]).decode()}))

    # --- streaming (incremental route in backend/main.py) ---
    def open_stream(self, language=None) -> _VllmStream:
        return _VllmStream(self._open())

    def stream_feed(self, st: _VllmStream, audio: np.ndarray, final: bool = False) -> list:
        """Push 16 kHz mono float32 audio (may be empty); return the words completed by it. final=True flushes the tail and closes the socket."""
        if len(audio):
            self._send_pcm(st.ws, np.ascontiguousarray(audio, dtype=np.float32))
        deltas = []
        if final:
            st.ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            deadline = time.time() + 20
            while time.time() < deadline:
                try:
                    m = json.loads(st.ws.recv(timeout=max(0.1, deadline - time.time())))
                except TimeoutError:
                    break
                if m.get("type") == "transcription.delta":
                    deltas.append(m["delta"])
                elif m.get("type") in ("transcription.done", "error"):
                    if m.get("type") == "error":
                        print(f"VoxtralVllmSTT: server error {str(m.get('error'))[:200]}")
                    break
            self._close(st.ws)
        else:
            while True:   # whatever has arrived, without waiting
                try:
                    m = json.loads(st.ws.recv(timeout=0))
                except TimeoutError:
                    break
                if m.get("type") == "transcription.delta":
                    deltas.append(m["delta"])
                elif m.get("type") == "error":
                    print(f"VoxtralVllmSTT: server error {str(m.get('error'))[:200]}")
                    break
        return st._words(deltas, final)

    # --- whole clip (phrase/clause/chunks strategies) ---
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, language: Optional[str] = None, vad_filter: bool = True,
                         initial_prompt: Optional[str] = None, word_timestamps: bool = False) -> Tuple[list, float, Optional[str]]:
        audio = np.ascontiguousarray(audio_data, dtype=np.float32)
        if sample_rate != 16000:
            import scipy.signal as ss
            audio = ss.resample_poly(audio, 16000, sample_rate).astype(np.float32)
        if len(audio) / 16000 < 0.3:
            return [], 0.0, language
        t0 = time.time()
        st = self.open_stream()
        words = self.stream_feed(st, audio, final=True)
        text = " ".join(words).strip()
        segs = [SimpleNamespace(start=0.0, end=len(audio) / 16000, text=text, words=None, no_speech_prob=0.0, avg_logprob=0.0)] if text else []
        return segs, time.time() - t0, language
