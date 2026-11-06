"""T044: thin wrapper around the vendored whisper_streaming (LocalAgreement + VAC) processor,
verified viable on this hardware in documentation/streaming_stt_2026-08-21.md (RTF ~1.001, first
partial at 3.3s). Additive captions only — never touches the STT model or SILENCE_TIMEOUT-driven
final-segment path that actually produces translated audio (backend/main.py).

Construction matches the exact CLI defaults used in that benchmark run (`--vac`, no `--vad`,
min-chunk-size 1.0s, buffer_trimming="segment"/15s) so real-world behavior here isn't an untested
deviation from what was already measured.
"""
import os
import sys
import logging

_DEVNULL = open(os.devnull, "w")  # the vendored library does a raw print() per chunk at DEBUG-ish
# verbosity (not gated by logging level) — silence it here rather than spam stderr at chunk cadence.

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_streaming_vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

try:
    from whisper_online import FasterWhisperASR, VACOnlineASRProcessor
    STREAMING_CAPTIONS_AVAILABLE = True
except ImportError as e:
    STREAMING_CAPTIONS_AVAILABLE = False
    logging.warning(f"backend.stt.streaming_captions: vendored whisper_streaming not importable ({e}). Captions disabled, main translation path unaffected.")


class SessionCaptioner:
    """One per session. NOTE: loads its own separate FasterWhisperASR ('base', int8) — a second
    whisper model instance alongside the session's primary STT model, not shared/pooled. Same
    per-session-memory-multiplication caveat already flagged for every engine in this project
    (T018, deferred). Not fixed here — out of scope for T044."""

    def __init__(self, source_lang: str = "en", model_dir: str | None = None):
        if not STREAMING_CAPTIONS_AVAILABLE:
            raise ImportError("whisper_streaming_vendor not available")
        lan = "auto" if not source_lang or source_lang == "auto" else source_lang
        if not model_dir or not os.path.isfile(os.path.join(model_dir, "model.bin")):
            raise FileNotFoundError("Streaming caption STT must be preinstalled locally")
        vad_dir = os.environ.get("BP_SILERO_VAD_DIR", "")
        if not os.path.isfile(os.path.join(vad_dir, "hubconf.py")):
            raise FileNotFoundError("Set BP_SILERO_VAD_DIR to a preinstalled local Silero repository")
        import torch
        vad_model, _ = torch.hub.load(repo_or_dir=vad_dir, model="silero_vad", source="local")
        self._asr = FasterWhisperASR(lan=lan, model_dir=model_dir)
        self._online = VACOnlineASRProcessor(
            1.0,  # min-chunk-size / online_chunk_size, matches benchmark default
            self._asr,
            None,  # tokenizer — None is correct for buffer_trimming="segment"
            buffer_trimming=("segment", 15),
            logfile=_DEVNULL,
            vad_model=vad_model,
        )

    def feed(self, audio_chunk_f32_16k) -> str:
        """Insert one chunk of 16kHz float32 audio, return newly committed partial text ('' if none).
        Safe to call at any chunk cadence — the processor self-throttles actual inference to
        roughly once per online_chunk_size seconds of new audio."""
        self._online.insert_audio_chunk(audio_chunk_f32_16k)
        _, _, text = self._online.process_iter()
        return text or ""

    def reset(self):
        """Call when the main pipeline's own SILENCE_TIMEOUT-based final segment fires, so this
        processor's internal buffer doesn't keep accumulating across utterance boundaries. Discards
        whatever partial was still pending — the real transcript for that segment comes from the
        primary STT model via the existing final-segment path, not from this captioner."""
        self._online.init()
