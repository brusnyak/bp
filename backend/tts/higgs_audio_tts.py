"""Higgs TTS 3 zero-shot voice cloning engine.

Higgs TTS 3 — 100+ languages including Czech and Slovak (single-digit WER/CER),
instant zero-shot voice cloning from a short reference WAV, streaming via SSE
(sub-second time-to-first-audio).

Self-hosted open-weights model (non-commercial: Boson Higgs TTS 3 Research and
Non-Commercial License). Runs on GPU (CUDA); not a live path on Mac.

Install: via transformers + AutoModelForSeq2SeqLM from HF weights,
or via SGLang-Omni / vLLM-Omni serving stack.
Model weights: https://huggingface.co/bosonai/higgs-tts-3-4b
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
    from transformers import AutoModelForSeq2SeqLM  # type: ignore
    HIGGS_AUDIO_AVAILABLE = True
except ImportError:
    torch = None
    AutoModelForSeq2SeqLM = None
    HIGGS_AUDIO_AVAILABLE = False


class HiggsAudioTTS:
    """Higgs TTS 3 zero-shot cloning engine.

    SUPPORTS_CLONING: True — zero-shot from reference audio.
    SUPPORTS_STREAMING: True — SSE streaming supported.
    LANGUAGE_OVERRIDES: SK/CZ supported natively (100+ languages).
    """
    SUPPORTS_CLONING = True
    REQUIRES_SPEAKER_WAV = True
    SUPPORTS_STREAMING = True
    LANGUAGE_OVERRIDES = {}
    sample_rate = 24000  # Higgs TTS 3 outputs 24 kHz

    def __init__(self, device: str = "auto", model_name: str = "bosonai/higgs-tts-3-4b"):
        if not HIGGS_AUDIO_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for Higgs TTS 3. "
                "Install: pip install transformers accelerate"
            )
        self.device = hardware.detect_backend("tts_clone") if device == "auto" else device
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._speaker_cache: dict = {}
        self._warmed = False
        logger.info("HiggsAudio: loading %s on %s", model_name, self.device)
        t0 = time.perf_counter()
        self._load()
        logger.info("HiggsAudio: loaded in %.1fs", time.perf_counter() - t0)

    def _load(self):
        # Higgs TTS 3 loads via transformers AutoModelForSeq2SeqLM
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, device_map=self.device, trust_remote_code=True
        )
        try:
            from transformers import AutoProcessor  # type: ignore
            self._processor = AutoProcessor.from_pretrained(self.model_name)
        except ImportError:
            self._processor = None
        self._warmed = True

    def prepare_voice(self, speaker_wav_path: str, ref_text: Optional[str] = None):
        """Cache reference audio for zero-shot cloning."""
        if speaker_wav_path in self._speaker_cache:
            return self._speaker_cache[speaker_wav_path]
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"HiggsAudio: reference WAV not found: {speaker_wav_path}")
        ref = self._model.clone_voice(speaker_wav_path, ref_text=ref_text)
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
        kwargs = {"language": lang}
        if speaker_wav_path:
            ref = self.prepare_voice(speaker_wav_path)
            kwargs["reference_audio"] = ref
        audio = self._model.generate(text=text, **kwargs)
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        dt = time.perf_counter() - t0
        if output_path:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
        logger.info("HiggsAudio: %r (%s) in %.2fs", text[:50], lang, dt)
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
