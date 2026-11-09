"""NVIDIA Nemotron 3.5 ASR Streaming 0.6B on Apple Silicon via mlx-audio: a small streaming recogniser (cache-aware FastConformer-RNNT, 600M parameters,
~0.64 GB at 8-bit) that covers 40 language-locales including German, Czech and Slovak, with a language prompt. UNTESTED on this repo's route (not
downloaded yet when this file was written, 2026-11-09): the model card and the mlx-audio session API (installed 0.6.4) are the evidence, not a run.

Same streaming interface as voxtral_stt.py (open_stream / stream_feed / transcribe_audio), which it extends. Licence: NVIDIA Open Model License (not Apache:
read it before shipping). Delay: 80, 160, 560 or 1120 ms (BP_NEMOTRON_DELAY_MS, default 560)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from backend.stt.voxtral_stt import VoxtralSTT, _Stream

DEFAULT_REPO = "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
LANGUAGES = frozenset("en es de fr it ar ja ko pt ru hi zh vi he nl cs da pl no sv th tr bg el et fi hr hu lt lv ro sk uk mt sl".split())


def prompt_key(prompt_dictionary, lang: Optional[str]) -> Optional[str]:
    """The model's own spelling of a language ('sk' may be 'sk-SK' there): first key equal to it or starting with 'sk-'; None = let the model use its default."""
    if not lang:
        return None
    lang = lang.lower()
    keys = list(prompt_dictionary or [])
    for k in keys:
        if k.lower() == lang:
            return k
    return next((k for k in keys if k.lower().startswith(lang + "-")), None)


class NemotronSTT(VoxtralSTT):
    incr_step_s = float(os.environ.get("BP_NEMOTRON_STEP_S", "0.3"))   # a guess until measured: cache-aware chunks should tolerate smaller pushes than Voxtral

    def __init__(self, model_size: str = "", device: str = "auto", compute_type: str = "", beam_size: int = 1):
        self.repo = os.environ.get("BP_NEMOTRON_MODEL", DEFAULT_REPO)
        self.delay_ms = int(os.environ.get("BP_NEMOTRON_DELAY_MS", "560"))
        self.model_size = self.repo
        from concurrent.futures import ThreadPoolExecutor
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nemotron")
        self._ex.submit(self._load).result()
        print(f"NemotronSTT initialized with {self.repo} (delay {self.delay_ms} ms)")

    def _load(self):
        from mlx_audio.stt.utils import load_model
        self._model = load_model(self.repo)
        self._model.generate(np.zeros(16000, dtype=np.float32))   # warm-up

    def open_stream(self, language=None) -> _Stream:
        key = prompt_key(getattr(self._model, "prompt_dictionary", None), language)
        kw = {"transcription_delay_ms": self.delay_ms}
        if key:
            kw["language"] = key
        return self._ex.submit(lambda: _Stream(self._model.create_streaming_session(**kw))).result()

    def transcribe_audio(self, audio_data, sample_rate, language=None, vad_filter=True, initial_prompt=None, word_timestamps=False):
        import time
        from types import SimpleNamespace
        audio = np.ascontiguousarray(audio_data, dtype=np.float32)
        if sample_rate != 16000:
            import scipy.signal as ss
            audio = ss.resample_poly(audio, 16000, sample_rate).astype(np.float32)
        if len(audio) / 16000 < 0.3:
            return [], 0.0, language
        key = prompt_key(getattr(self._model, "prompt_dictionary", None), language)
        t0 = time.time()
        res = self._ex.submit(lambda: self._model.generate(audio, **({"language": key} if key else {}))).result()
        text = (getattr(res, "text", "") or "").strip()
        segs = [SimpleNamespace(start=0.0, end=len(audio) / 16000, text=text, words=None, no_speech_prob=0.0, avg_logprob=0.0)] if text else []
        return segs, time.time() - t0, language
