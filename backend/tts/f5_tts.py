"""F5-TTS zero-shot voice cloning engine (SWivid/F5-TTS, MIT).

Flow-matching DiT TTS with zero-shot cloning from a short reference clip.
Whole-utterance generation (not token-streaming), but fast enough per segment
to use inside a streaming STT→MT→TTS cascade on GPU.

Languages: EN/ZH native per model card; SK/CZ not natively supported —
Czech proxy via LANGUAGE_OVERRIDES for evaluation. CC-BY-NC was reported
in archive docs; PyPI package is MIT — verify licence before live use.

Not a live path on Mac (PyTorch, no MPS kernels); GPU engine for Kaggle T4.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

import numpy as np

from backend import hardware

logger = logging.getLogger(__name__)

try:
    import torch
    from f5_tts.api import F5TTS
    F5_TTS_AVAILABLE = True
except ImportError:
    torch = None
    F5TTS = None
    F5_TTS_AVAILABLE = False


class F5TTSWrapper:
    """F5-TTS zero-shot cloning engine.

    Follows the same interface as CoquiTTS/OMniVoice so main.py never changes.
    """
    SUPPORTS_CLONING = True
    REQUIRES_SPEAKER_WAV = True
    SUPPORTS_STREAMING = True
    LANGUAGE_OVERRIDES = {"sk": "cs"}  # no native SK; Czech proxy for eval
    sample_rate = 24000

    def __init__(self, device: str = "auto", model_name: str = "F5TTS_v1_Base"):
        if not F5_TTS_AVAILABLE:
            raise ImportError("f5-tts package is not installed. `pip install f5-tts`")
        self.device = hardware.detect_backend("tts_clone") if device == "auto" else device
        self.model_name = model_name
        self._model = None
        self._speaker_cache: dict = {}  # ref_wav_path -> (ref_audio, ref_text)
        self._warmed = False
        logger.info("F5TTS: loading %s on %s", model_name, self.device)
        t0 = time.perf_counter()
        self._load()
        logger.info("F5TTS: loaded in %.1fs", time.perf_counter() - t0)

    def _load(self):
        self._model = F5TTS(model=self.model_name, device=self.device)
        self._warmed = True

    def prepare_voice(self, speaker_wav_path: str, ref_text: Optional[str] = None):
        """Cache reference audio + text for zero-shot cloning."""
        if speaker_wav_path in self._speaker_cache:
            return self._speaker_cache[speaker_wav_path]
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"F5TTS: reference WAV not found: {speaker_wav_path}")
        self._speaker_cache[speaker_wav_path] = (speaker_wav_path, ref_text)
        return self._speaker_cache[speaker_wav_path]

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker_wav_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        lang = self.LANGUAGE_OVERRIDES.get(language, language)
        kwargs = {}
        if speaker_wav_path:
            ref_wav, ref_text = self.prepare_voice(speaker_wav_path)
            kwargs["ref_audio"] = ref_wav
            if ref_text:
                kwargs["ref_text"] = ref_text
        audio = self._model.infer(text=text, language=lang, **kwargs)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        dt = time.perf_counter() - t0
        if output_path:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
        logger.info("F5TTS: %r (%s) in %.2fs", text[:50], lang, dt)
        return audio, self.sample_rate, dt

    def synthesize_stream(
        self,
        text: str,
        language: str = "en",
        speaker_wav_path: Optional[str] = None,
        **kwargs,
    ):
        """Sentence-level streaming: yield one chunk per sentence."""
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text]
        for sentence in sentences:
            wav, _, _ = self.synthesize(sentence, language=language,
                                        speaker_wav_path=speaker_wav_path, **kwargs)
            yield wav
