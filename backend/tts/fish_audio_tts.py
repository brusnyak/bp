"""Fish Audio S2-Pro zero-shot voice cloning engine.

Fish Audio S2-Pro — ~100 languages including Slovak and Czech, instant zero-shot
voice cloning from a 10–30 s reference WAV, production streaming (RTF 0.195 on H200,
~100 ms first audio).

Self-hosted open-weights model (non-commercial: Fish Audio Research License for S2-Pro
weights; BSD-3-Clause for code). Runs on GPU (CUDA); not a live path on Mac.

Install from source: git clone https://github.com/fishaudio/fish-speech && pip install -e .
Model weights: https://huggingface.co/fishaudio/fish-speech-1
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
    # Fish Speech is installed from GitHub source, not PyPI.
    # Import path depends on installation: fish_speech package from the repo.
    from fish_speech.api import FishSpeechAPI  # type: ignore
    FISH_AUDIO_AVAILABLE = True
except ImportError:
    torch = None
    FishSpeechAPI = None
    FISH_AUDIO_AVAILABLE = False


class FishAudioTTS:
    """Fish Audio S2-Pro zero-shot cloning engine.

    SUPPORTS_CLONING: True — zero-shot from reference WAV.
    SUPPORTS_STREAMING: True — production streaming supported.
    LANGUAGE_OVERRIDES: SK/CZ supported natively (~100 languages).
    """
    SUPPORTS_CLONING = True
    REQUIRES_SPEAKER_WAV = True
    SUPPORTS_STREAMING = True
    LANGUAGE_OVERRIDES = {}
    sample_rate = 44100  # Fish Speech outputs 44.1 kHz

    def __init__(self, device: str = "auto", model_name: str = "fishaudio/fish-speech-1"):
        if not FISH_AUDIO_AVAILABLE:
            raise ImportError(
                "fish-speech is not installed. "
                "Clone https://github.com/fishaudio/fish-speech and pip install -e ."
            )
        self.device = hardware.detect_backend("tts_clone") if device == "auto" else device
        self.model_name = model_name
        self._api = None
        self._speaker_cache: dict = {}
        self._warmed = False
        logger.info("FishAudio: loading %s on %s", model_name, self.device)
        t0 = time.perf_counter()
        self._load()
        logger.info("FishAudio: loaded in %.1fs", time.perf_counter() - t0)

    def _load(self):
        # Fish Speech API takes model_dir or model_name and downloads weights from HF
        self._api = FishSpeechAPI(model_name=self.model_name, device=self.device)
        self._warmed = True

    def prepare_voice(self, speaker_wav_path: str, ref_text: Optional[str] = None):
        """Cache reference audio for zero-shot cloning."""
        if speaker_wav_path in self._speaker_cache:
            return self._speaker_cache[speaker_wav_path]
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"FishAudio: reference WAV not found: {speaker_wav_path}")
        ref = self._api.clone_voice(speaker_wav_path, ref_text=ref_text)
        self._speaker_cache[speaker_wav_path] = ref
        return ref

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
            ref = self.prepare_voice(speaker_wav_path)
            kwargs["reference_audio"] = ref
        audio = self._api.infer(text=text, language=lang, **kwargs)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        dt = time.perf_counter() - t0
        if output_path:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
        logger.info("FishAudio: %r (%s) in %.2fs", text[:50], lang, dt)
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
