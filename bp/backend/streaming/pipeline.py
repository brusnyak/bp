"""BP v2 Streaming Pipeline — true near real-time STT→MT→TTS.

Unlike v1 (wait for silence → full segment → translate), this processes audio
in small overlapping windows and emits translated audio chunks as soon as words
are stable. Target: first translated audio ≤ ~1s after the speaker's word.

Architecture per speaker→language direction:
  VAD (20ms frames) → STT windows (0.5-1s, word timestamps) → local-agreement
  word commit → fragment buffer (3-8 words) → MT (chunk translate) → TTS
  (streaming synthesis) → WebSocket WAV chunks → listener playback.

Designed to run on CPU (Piper + faster-whisper int8) or GPU (OmniVoice +
faster-whisper CUDA fp16).
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf

from backend.hardware_scan import AutoConfig
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.stt.incremental import Agreement, Fragmenter
from backend.tts.base import TTS_ENGINES
from backend.registry import target_tts_engine

log = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 16000
VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * VAD_FRAME_MS / 1000)

# STT window size for streaming transcription (seconds)
STT_WINDOW_S = 0.7
# Minimum audio before we attempt a hypothesis
MIN_AUDIO_S = 1.0
# Local-agreement: word committed after N consecutive hypotheses agree
AGREE_WINDOWS = 2
# Fragment flush thresholds
FRAGMENT_MAX_WORDS = 9
FRAGMENT_MAX_WAIT_S = 1.6


@dataclass
class StreamingResult:
    """One emitted translation result."""
    speaker_id: str
    source_text: str
    translated_text: str
    target_lang: str
    audio: bytes            # WAV-encoded chunk
    sample_rate: int
    timestamp: float
    sequence: int
    stt_ms: float
    mt_ms: float
    tts_ms: float


class _VADGate:
    """WebRTC VAD wrapper that feeds 20ms frames and reports speech/silence."""

    def __init__(self, aggressiveness: int = 3):
        import webrtcvad
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_samples = VAD_FRAME_SAMPLES

    def is_speech(self, frame_int16: bytes) -> bool:
        return self._vad.is_speech(frame_int16, AUDIO_SAMPLE_RATE)


class StreamingPipeline:
    """One speaker→language streaming pipeline.

    Created per (speaker, target_language) pair inside a meeting room.
    Holds its own STT/MT/TTS model instances (pooled where possible).
    """

    def __init__(self, speaker_id: str, source_lang: str, target_lang: str,
                 voice: str, tts_engine: str, config: AutoConfig,
                 allow_noncommercial: bool = False,
                 allow_unknown_license: bool = False):
        self.speaker_id = speaker_id
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.voice = voice
        self.tts_engine_name = target_tts_engine(tts_engine, target_lang)
        self.config = config
        self.allow_nc = allow_noncommercial
        self.allow_unk = allow_unknown_license

        # Models (lazy init)
        self._stt: Optional[FasterWhisperSTT] = None
        self._mt: Optional[CTranslate2MT] = None
        self._tts = None
        self._vad: Optional[_VADGate] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Streaming state
        self._agreement = Agreement()
        self._fragmenter = Fragmenter(
            min_words=int(os_env("BP_INCR_MIN_WORDS", "3")),
            max_words=int(os_env("BP_INCR_MAX_WORDS", "9")),
            max_wait_s=float(os_env("BP_INCR_MAX_WAIT_S", "1.6")),
        )
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_start_sample = 0
        self._received_samples = 0
        self._has_speech = False
        self._sequence = 0
        self._last_step = 0.0
        self._step_interval = float(os_env("BP_INCR_STEP_S", "0.7"))

    # ------------------------------------------------------------------ init

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return

        # STT model size from auto-config, overridden by env if set
        stt_model_size = os_env("BP_STT_MODEL_SIZE", self.config.stt.model_size)
        stt_device = self.config.stt.device
        stt_compute = self.config.stt.compute_type
        beam = int(os_env("BP_STT_BEAM_SIZE", "5"))

        def stt_factory():
            return FasterWhisperSTT(
                model_size=stt_model_size,
                device=stt_device,
                compute_type=stt_compute,
                beam_size=beam,
            )
        self._stt = stt_factory()

        # MT model
        mt_model_name = f"Helsinki-NLP/opus-mt-{self.source_lang}-{self.target_lang}"
        from backend.registry import instantiate, select
        select("mt", "opus-mt",
               allow_noncommercial=self.allow_nc,
               allow_unknown=self.allow_unk)
        self._mt = instantiate(
            "mt", "opus-mt",
            model_path=f"ct2_models/{mt_model_name.replace('/', '--')}",
            device=self.config.mt.device,
            allow_noncommercial=self.allow_nc,
            allow_unknown=self.allow_unk,
        )

        # TTS engine
        self._tts = TTS_ENGINES[self.tts_engine_name]()

        # VAD
        self._vad = _VADGate()

        self._initialized = True
        log.info(
            "StreamingPipeline %s: %s→%s ready (stt=%s mt=%s tts=%s)",
            self.speaker_id, self.source_lang, self.target_lang,
            self.config.stt.engine, self.config.mt.engine, self.tts_engine_name,
        )

    async def close(self) -> None:
        self._initialized = False
        self._stt = self._mt = self._tts = self._vad = None

    # ---------------------------------------------------------------- streaming

    async def process_audio_chunk(self, audio_np: np.ndarray, sample_rate: int) -> Optional[StreamingResult]:
        """Feed one PCM chunk (16kHz mono float32). Returns a StreamingResult
        when a translated audio chunk is ready, else None.

        This is the hot path: called for every 20ms VAD frame from the WebSocket
        receive loop.
        """
        if not self._initialized:
            await self.initialize()

        now = time.perf_counter()
        t_stt_start = now

        # Buffer audio until we have enough for a hypothesis window
        mono = audio_np if audio_np.ndim == 1 else audio_np.mean(axis=1)
        self._audio_buffer.append(mono)
        self._buffer_samples += len(mono)
        self._received_samples += len(mono)

        # VAD: check if current frame is speech
        frame_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
        is_speech = self._vad.is_speech(frame_int16.tobytes()) if self._vad else True

        self._has_speech = self._has_speech or is_speech
        if not self._has_speech:
            # Silence before any speech — drop
            self._audio_buffer.clear()
            self._buffer_samples = 0
            self._buffer_start_sample = self._received_samples
            return await self.flush_pending(now)

        # Only run STT step if enough audio has accumulated since last step
        if now - self._last_step < self._step_interval:
            return await self.flush_pending(now)

        # Build the audio window for STT
        window = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([], dtype=np.float32)
        window_s = len(window) / AUDIO_SAMPLE_RATE

        if window_s < MIN_AUDIO_S:
            return await self.flush_pending(now)

        self._last_step = now

        # Agreement expects the next window to start immediately after the last
        # commit. Never roll away uncommitted source audio just to cap the window.
        window_start_s = self._buffer_start_sample / AUDIO_SAMPLE_RATE

        t_stt = time.perf_counter()
        try:
            segments, stt_time, detected = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._stt.transcribe_audio(
                    window, AUDIO_SAMPLE_RATE,
                    language=self.source_lang if self.source_lang != "auto" else None,
                    vad_filter=False,
                    word_timestamps=True,
                ),
            )
        except Exception as exc:
            log.error("StreamingPipeline %s: STT error: %s", self.speaker_id, exc)
            return None
        t_stt_end = time.perf_counter()

        # Extract words with timestamps
        words = []
        for seg in segments:
            for w in getattr(seg, "words", None) or []:
                if w.word and w.word.strip():
                    words.append((w.word.strip(), float(w.start), float(w.end)))

        if not words:
            return await self.flush_pending()

        # Local-agreement commit
        committed = self._agreement.update(words)
        if not committed:
            return await self.flush_pending()

        now_wall = time.perf_counter()
        self._fragmenter.push([(w[0], window_start_s + w[1], window_start_s + w[2])
                               for w in committed], now_wall)
        trim_samples = min(len(window), max(0, round(committed[-1][2] * AUDIO_SAMPLE_RATE)))
        remaining = window[trim_samples:]
        self._audio_buffer = [remaining] if len(remaining) else []
        self._buffer_samples = len(remaining)
        self._buffer_start_sample += trim_samples
        return await self.flush_pending(now_wall, stt_ms=(t_stt_end - t_stt_start) * 1000)

    async def flush_pending(self, now=None, *, final=False, stt_ms=0.0) -> Optional[StreamingResult]:
        """Clock tick usable without new audio or an STT commit (e.g. idle transport timer)."""
        now = time.perf_counter() if now is None else now
        pending_since = self._fragmenter.since
        expired = pending_since is not None and now - pending_since >= self._fragmenter.max_wait_s
        # Fragmenter intentionally requires two words for normal timeout. A clock
        # deadline must also release a single committed tail in this route.
        fragment = self._fragmenter.pop(now, final=final or expired)
        if not fragment:
            return None

        # We have a committed fragment → translate + synthesize
        t_mt_start = time.perf_counter()
        try:
            translated, _ = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._mt.translate(fragment, self.source_lang, self.target_lang),
            )
        except Exception as exc:
            log.error("StreamingPipeline %s: MT error: %s", self.speaker_id, exc)
            return None
        t_mt_end = time.perf_counter()

        if not translated or not translated.strip():
            return None

        # Streaming TTS: synthesize the fragment
        t_tts_start = time.perf_counter()
        try:
            audio_wav, sample_rate_out, tts_latency = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._tts.synthesize(
                    translated, language=self.target_lang,
                    speaker_wav_path=None,  # voice selection handled by engine factory
                ),
            )
        except Exception as exc:
            log.error("StreamingPipeline %s: TTS error: %s", self.speaker_id, exc)
            return None
        t_tts_end = time.perf_counter()

        # Encode to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio_wav, sample_rate_out, format="WAV")
        audio_bytes = buf.getvalue()

        self._sequence += 1
        result = StreamingResult(
            speaker_id=self.speaker_id,
            source_text=fragment,
            translated_text=translated,
            target_lang=self.target_lang,
            audio=audio_bytes,
            sample_rate=sample_rate_out,
            timestamp=time.perf_counter(),
            sequence=self._sequence,
            stt_ms=round(stt_ms, 1),
            mt_ms=round((t_mt_end - t_mt_start) * 1000, 1),
            tts_ms=round((t_tts_end - t_tts_start) * 1000, 1),
        )
        return result

    async def flush_final(self) -> Optional[StreamingResult]:
        """Flush any remaining committed words as a final fragment."""
        return await self.flush_pending(final=True)


# ---------------------------------------------------------------------------
# env helper (avoid importing os at module top for cleanliness)
# ---------------------------------------------------------------------------

import os  # noqa: E402


def os_env(key: str, default: str) -> str:
    return os.environ.get(key, default)