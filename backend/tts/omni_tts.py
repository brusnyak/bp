"""OmniVoice zero-shot cloning engine (k2-fsa/OmniVoice, Apache-2.0).

Rewritten 2026-10-02 against the real OmniVoice API (the first version passed a non-existent `speaker_audio` kwarg, so it
never cloned). Measured on a free Kaggle T4: RTF 0.18-0.34 at num_step=16 (0.2 at 12) (documentation/load_tests_2026-10.md); on Apple MPS
RTF ~0.8-1, so it is a GPU engine.

- The voice-clone prompt (reference audio tokens + reference text) is built once per reference file and cached: this is the
  "warm prep" that belongs at enrollment time, not per sentence. Reference text comes from a `<ref>.txt` sidecar if present;
  otherwise OmniVoice transcribes the reference once (extra model load) and the result is cached with the prompt.
- Pace/latency knobs, env-overridable: OMNIVOICE_STEPS (default 12: ear-checked 2026-10-02, EN 5/5 both, SK 4/5 vs 2/5 for 16), OMNIVOICE_SPEED (default 1.0). Defaults are the measured
  baseline; tune with scripts/gpu_bench/tts_tune.py before changing them.
- synthesize_stream yields clause-level chunks (first audio after the first clause, not the whole sentence).
"""
import logging
import os
import threading
import time
from typing import Optional, Tuple

import numpy as np

from backend import hardware

try:
    import torch
    from omnivoice import OmniVoice
    OMNIVOICE_AVAILABLE = True
except ImportError:  # lean runtime (no torch): the factory in base.py skips registration
    torch = None
    OmniVoice = None
    OMNIVOICE_AVAILABLE = False

logger = logging.getLogger(__name__)


class OmniVoiceTTS:
    SUPPORTS_CLONING = True
    SUPPORTS_STREAMING = True
    REQUIRES_SPEAKER_WAV = False  # falls back to OmniVoice's auto voice if no reference is given

    def __init__(self, model_name: str = "k2-fsa/OmniVoice", device: str = "auto"):
        if not OMNIVOICE_AVAILABLE:
            raise ImportError("OmniVoice package is not installed. Install it with: pip install omnivoice")
        self.model_name = model_name
        self.device = hardware.detect_backend("tts_clone") if device == "auto" else device
        self.num_step = int(os.environ.get("OMNIVOICE_STEPS", "12"))
        self.speed = float(os.environ.get("OMNIVOICE_SPEED", "1.0"))
        logger.info("OmniVoiceTTS: loading %s on %s (num_step=%s, speed=%s)", model_name, self.device, self.num_step, self.speed)
        t0 = time.perf_counter()
        self.model = OmniVoice.from_pretrained(
            model_name, device_map=self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)  # fp16 on MPS segfaults
        self.sample_rate = getattr(self.model, "sampling_rate", 24000)
        self._prompts: dict = {}  # reference wav path -> VoiceClonePrompt
        self._warmed = False
        self._lock = threading.Lock()  # one generate() at a time, same as the single TTS worker measured
        logger.info("OmniVoiceTTS: loaded in %.1fs", time.perf_counter() - t0)

    def prepare_voice(self, speaker_wav_path: str):
        """Build (and cache) the clone prompt for a reference recording. Call at enrollment / session start."""
        if speaker_wav_path in self._prompts:
            return self._prompts[speaker_wav_path]
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"Speaker WAV file not found: {speaker_wav_path}")
        sidecar = os.path.splitext(speaker_wav_path)[0] + ".txt"
        ref_text = open(sidecar, encoding="utf-8").read().strip() if os.path.exists(sidecar) else None
        t0 = time.perf_counter()
        prompt = self.model.create_voice_clone_prompt(speaker_wav_path, ref_text)
        logger.info("OmniVoiceTTS: clone prompt for %s in %.2fs (ref text: %s)", os.path.basename(speaker_wav_path),
                    time.perf_counter() - t0, "sidecar" if ref_text else "auto-transcribed")
        self._prompts[speaker_wav_path] = prompt
        if not self._warmed:  # first generate() after load pays one-off kernel/alloc cost: pay it at enrollment, not on the first sentence
            t1 = time.perf_counter()
            with self._lock:
                self.model.generate(text="Test.", language="en", voice_clone_prompt=prompt, num_step=self.num_step)
            self._warmed = True
            logger.info("OmniVoiceTTS: warm-up generate in %.2fs", time.perf_counter() - t1)
        return prompt

    def synthesize(self, text: str, language: str = "sk", speaker_wav_path: Optional[str] = None,
                   output_path: Optional[str] = None) -> Tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        kwargs = {"text": text, "language": language, "num_step": self.num_step}
        if self.speed != 1.0:
            kwargs["speed"] = self.speed
        if speaker_wav_path:
            kwargs["voice_clone_prompt"] = self.prepare_voice(speaker_wav_path)
        with self._lock:
            audio = self.model.generate(**kwargs)[0]
            if self.device == "cuda":
                torch.cuda.synchronize()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        dt = time.perf_counter() - t0
        logger.info("OmniVoiceTTS: %r (%s, %s) in %.2fs", text[:50], language, "cloned" if speaker_wav_path else "auto voice", dt)
        return audio, self.sample_rate, dt

    def synthesize_stream(self, text: str, language: str = "sk", speaker_wav_path: Optional[str] = None, **kwargs):
        """One chunk per clause-level phrase so the first audio leaves after the first clause."""
        from backend.tts.text_chunking import chunk_audio_by_duration, split_into_phrases
        for phrase in split_into_phrases(text):
            wav, sr, _ = self.synthesize(phrase, language=language, speaker_wav_path=speaker_wav_path)
            if wav is not None and len(wav) > 0:
                yield from chunk_audio_by_duration(wav, sr)
