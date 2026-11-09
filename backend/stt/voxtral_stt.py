"""Voxtral Mini 4B Realtime (Apache 2.0) on Apple Silicon via mlx-audio: a recogniser built for streaming.

Unlike Whisper (re-decodes a padded 30 s window on every call), the model consumes audio as it arrives and emits
tokens ~0.5 s behind the speech, and never revises them, so no local-agreement step is needed.
Measured 2026-11-08 on M1 Pro (4-bit, owner English script, fed in real time): 0.3-0.5 s lag while the GPU is free.
13 languages (no Slovak): see LANGUAGES. MLX streams are thread-local, so every MLX call runs on one worker thread.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np

DEFAULT_REPO = "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"
LANGUAGES = frozenset("ar de en es fr hi it nl pt ru zh ja ko".split())


class _Stream:
    """One utterance: raw model tokens in, whole words out (the last word is held until the next token starts a new one)."""
    def __init__(self, session):
        self.session, self.buf = session, ""

    def _words(self, deltas, final=False):
        self.buf += "".join(deltas)
        parts = self.buf.split()
        if final or not parts or self.buf[-1:].isspace():
            self.buf = ""
            return parts
        self.buf = parts[-1]
        return parts[:-1]


class VoxtralSTT:
    incr_step_s = float(os.environ.get("BP_VOXTRAL_STEP_S", "0.64"))   # min audio between pushes. M1 Pro 2026-11-08: 0.2 s pushes fell 6 s behind (per-push overhead), 0.64 s and 1.28 s held lag <= 0.7 s
    incr_min_audio_s = 0.0

    def __init__(self, model_size: str = "", device: str = "auto", compute_type: str = "", beam_size: int = 1):
        self.repo = os.environ.get("BP_VOXTRAL_MODEL", DEFAULT_REPO)
        self.delay_ms = int(os.environ.get("BP_VOXTRAL_DELAY_MS", "480"))
        self.model_size = self.repo
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxtral")
        self._ex.submit(self._load).result()
        print(f"VoxtralSTT initialized with {self.repo} (delay {self.delay_ms} ms)")

    def _load(self):
        from mlx_audio.stt.utils import load_model
        self._model = load_model(self.repo)
        self._model.generate(np.zeros(16000, dtype=np.float32))   # warm-up: kernels + weights

    # --- streaming (used by the incremental route in backend/main.py) ---
    def open_stream(self) -> _Stream:
        return self._ex.submit(lambda: _Stream(self._model.create_streaming_session(transcription_delay_ms=self.delay_ms))).result()

    def stream_feed(self, st: _Stream, audio: np.ndarray, final: bool = False) -> list:
        """Push 16 kHz mono float32 audio (may be empty); return the words completed by it. final=True flushes the tail."""
        def run():
            if len(audio):
                st.session.feed(np.ascontiguousarray(audio, dtype=np.float32))
            if final:
                st.session.close()
            out = []
            for _ in range(400 if final else 4):   # bounded drain; live pushes are tiny, close drains the model's delay window
                d = st.session.step(max_decode_tokens=16)
                out += d
                if st.session.done or (not final and not d):
                    break
            return st._words(out, final)
        return self._ex.submit(run).result()

    # --- whole-clip interface shared with the other engines (phrase/clause/chunks strategies) ---
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, language: Optional[str] = None, vad_filter: bool = True,
                         initial_prompt: Optional[str] = None, word_timestamps: bool = False) -> Tuple[list, float, Optional[str]]:
        import time
        audio = np.ascontiguousarray(audio_data, dtype=np.float32)
        if sample_rate != 16000:
            import scipy.signal as ss
            audio = ss.resample_poly(audio, 16000, sample_rate).astype(np.float32)
        if len(audio) / 16000 < 0.3:
            return [], 0.0, language
        t0 = time.time()
        res = self._ex.submit(lambda: self._model.generate(audio, transcription_delay_ms=self.delay_ms)).result()
        text = (getattr(res, "text", "") or "").strip()
        segs = [SimpleNamespace(start=0.0, end=len(audio) / 16000, text=text, words=None, no_speech_prob=0.0, avg_logprob=0.0)] if text else []
        return segs, time.time() - t0, language
