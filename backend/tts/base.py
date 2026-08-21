"""
TTS engine registry — the single place to add, remove, or swap a synthesis engine.

Why this exists: main.py used to have the same 4-way if/elif chain hardcoded in
three separate places (session init, session lookup, WebSocket dispatch), each
one duplicating device-selection logic and each easy to update inconsistently
(hybrid/omnivoice had already fallen out of sync in two of the three spots).
Every engine now goes through this registry instead, so main.py never needs to
know engine names or count.

To add a new engine (e.g. if a faster/better model replaces one of these):
  1. Write a class with:
       .synthesize(text: str, language: str, speaker_wav_path: Optional[str] = None)
           -> (wav: np.ndarray, sample_rate: int, latency_seconds: float)
     (mirror an existing backend/tts/*.py engine for the shape.)
  2. Optionally set class attributes read by main.py's dispatcher:
       SUPPORTS_CLONING: bool        (default False)
       REQUIRES_SPEAKER_WAV: bool    (default False — clone-capable but has a generic fallback)
       SUPPORTS_STREAMING: bool      (default False — if True, also implement .synthesize_stream(...))
       LANGUAGE_OVERRIDES: dict      (default {} — e.g. {"sk": "cs"} to proxy an unsupported language)
       sample_rate: int              (used for streamed WAV framing; defaults to 24000 if absent)
  3. Add a one-line factory below and register it in TTS_ENGINES.

Nothing in main.py changes when you do this.
"""
from typing import Callable, Dict

from backend import hardware
from backend.tts.piper_tts import PiperTTS
from backend.tts.coqui_tts import CoquiTTS
from backend.tts.hybrid_tts import HybridTTS
from backend.tts.omni_tts import OmniVoiceTTS


def _piper_factory() -> PiperTTS:
    return PiperTTS(model_id="cs_CZ-jirka-medium", device=hardware.detect_backend("tts_baseline"))


def _piper_personal_factory() -> PiperTTS:
    # The user's own fine-tuned voice (backend/tts/piper_models/en_US-personal-medium.onnx,
    # RTF 0.056 per documentation/personal_voice_bootstrap_2026-08-19.md). EN-only — the fine-tune
    # was trained on EN-only recordings, so this is not a generic-language voice like the other
    # Piper entries; no SUPPORTS_CLONING flag exists for "clones one specific pre-baked speaker",
    # it's just a different fixed voice.
    return PiperTTS(model_id="en_US-personal-medium", device=hardware.detect_backend("tts_baseline"))


def _xtts_factory() -> CoquiTTS:
    return CoquiTTS(device=hardware.detect_backend("tts_clone"))


def _hybrid_factory() -> HybridTTS:
    return HybridTTS(device=hardware.detect_backend("tts_clone"))


def _omnivoice_factory() -> OmniVoiceTTS:
    return OmniVoiceTTS(device=hardware.detect_backend("tts_clone"))


TTS_ENGINES: Dict[str, Callable[[], object]] = {
    "piper": _piper_factory,
    "piper_personal": _piper_personal_factory,
    "xtts": _xtts_factory,
    "hybrid": _hybrid_factory,
    "omnivoice": _omnivoice_factory,
}
