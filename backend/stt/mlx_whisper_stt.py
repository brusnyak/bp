"""Whisper on Apple Silicon via mlx-whisper (MIT). Same interface as FasterWhisperSTT.

Measured 2026-11-07 on M1 Pro (scripts/mac_stt_bench.py): large-v3-turbo WER 4.4% de / 5.4% en (FLEURS), RTF 0.04-0.09,
i.e. ~1 s per 10 s clip on the GPU, where faster-whisper large-v3-turbo on CPU int8 ran ~0.5x real time.
The model is fixed by BP_MLX_MODEL (default mlx-community/whisper-large-v3-turbo); the per-pack model_size is ignored
on purpose, because the pack sizes (base/small) were chosen for CPU speed.
MLX streams are thread-local, so every call runs on one dedicated worker thread (the pipeline calls from many).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np

DEFAULT_REPO = "mlx-community/whisper-large-v3-turbo"
from backend.stt.guards import GUARD, HALLUCINATIONS, SHORT_CHUNK_S, _norm, is_garbage  # noqa: F401  (shared with every engine; re-exported for tests)


class MLXWhisperSTT:
    def __init__(self, model_size: str = "large-v3-turbo", device: str = "auto", compute_type: str = "float16", beam_size: int = 5):
        self.repo = os.environ.get("BP_MLX_MODEL", DEFAULT_REPO)
        self.model_size = self.repo
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-whisper")
        self._ex.submit(self._load).result()
        print(f"MLXWhisperSTT initialized with {self.repo}")

    def _load(self):
        import mlx_whisper
        self._mw = mlx_whisper
        # Warm-up compiles kernels and loads weights so the first live segment is not slow.
        self._mw.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=self.repo, language="en", condition_on_previous_text=False)

    def _run(self, audio: np.ndarray, language, initial_prompt, word_timestamps):
        kw = dict(path_or_hf_repo=self.repo, condition_on_previous_text=False, no_speech_threshold=0.3, word_timestamps=bool(word_timestamps))
        if language:
            kw["language"] = language
        if initial_prompt:
            kw["initial_prompt"] = initial_prompt
        return self._mw.transcribe(audio, **kw)

    def transcribe_audio(
        self, audio_data: np.ndarray, sample_rate: int, language: Optional[str] = None, vad_filter: bool = True,
        initial_prompt: Optional[str] = None, word_timestamps: bool = False,
    ) -> Tuple[list, float, Optional[str]]:
        audio = np.ascontiguousarray(audio_data, dtype=np.float32)
        if sample_rate != 16000:
            import scipy.signal as ss
            audio = ss.resample_poly(audio, 16000, sample_rate).astype(np.float32)
        if len(audio) / 16000 < MIN_CHUNK_S:
            return [], 0.0, language
        t0 = time.time()
        res = self._ex.submit(self._run, audio, language, initial_prompt, word_timestamps).result()
        dt = time.time() - t0
        segs = []
        chunk_s = len(audio) / 16000
        for s in res.get("segments", []):
            if GUARD and is_garbage(s["text"], chunk_s, s.get("compression_ratio", 0.0)):
                print(f"MLXWhisperSTT: dropped garbage {s['text'][:60]!r} (chunk {chunk_s:.2f}s, ratio {s.get('compression_ratio', 0):.1f})")
                continue
            if GUARD and chunk_s < SHORT_CHUNK_S and _norm(s["text"]) in HALLUCINATIONS:
                print(f"MLXWhisperSTT: dropped likely hallucination {s['text']!r} (chunk {chunk_s:.2f}s, no_speech_prob {s.get('no_speech_prob', 0):.2f})")
                continue
            words = [SimpleNamespace(word=w["word"], start=w["start"], end=w["end"], probability=w.get("probability", 1.0))
                     for w in s.get("words", [])]
            segs.append(SimpleNamespace(start=s["start"], end=s["end"], text=s["text"], words=words or None,
                                        no_speech_prob=s.get("no_speech_prob", 0.0), avg_logprob=s.get("avg_logprob", 0.0)))
        return segs, dt, res.get("language")
