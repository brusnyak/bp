"""BP v2 Streaming Pipeline — TypeWhisper-style continuous recording.

Architecture per speaker→language direction:
  Continuous audio buffer → VAD (segmentation only, never drops speech) →
  STT on overlapping windows (0.3s step, 1.5s window) → word timestamps →
  phrase chunking (gap >0.3s between words = new phrase) → MT → TTS
  → WebSocket WAV chunks → listener playback.

Preserves the speaker's natural pauses and prosody by synthesizing each
phrase as a complete utterance rather than word-by-word fragments.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf

from backend.hardware_scan import AutoConfig
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.base import TTS_ENGINES
from backend.registry import target_tts_engine

log = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 16000
VAD_FRAME_MS = 20
VAD_FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * VAD_FRAME_MS / 1000)

# ---------------------------------------------------------------------------
# Continuous streaming constants
# ---------------------------------------------------------------------------
STT_STEP_S = 0.5          # run STT every 0.5s on the audio buffer
STT_WINDOW_S = 2.0        # 2.0s audio window per STT call
MIN_UTTERANCE_S = 1.0     # don't synthesize utterances shorter than this
SILENCE_GAP_S = 0.3       # gap between words that starts a new phrase
UTTERANCE_SILENCE_S = 0.8 # silence that ends an utterance (VAD frames)
MAX_UTTERANCE_S = 30.0    # hard cap so we never buffer forever


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
    """WebRTC VAD wrapper for utterance boundary detection only.
    
    Used to detect speech start/end — never drops audio during speech.
    """

    def __init__(self, aggressiveness: int = 1):
        import webrtcvad
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_samples = VAD_FRAME_SAMPLES

    def is_speech(self, frame_int16: bytes) -> bool:
        return self._vad.is_speech(frame_int16, AUDIO_SAMPLE_RATE)


class ContinuousStreamingPipeline:
    """One speaker→language continuous streaming pipeline.

    Records ALL audio continuously into a rolling buffer. STT runs on
    overlapping windows every STT_STEP_S. Words are grouped into phrases
    by natural pauses (SILENCE_GAP_S between words). Each phrase is
    translated and synthesized as a complete utterance, preserving the
    speaker's pauses and prosody.
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

        # Continuous audio buffer
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_start_sample = 0
        self._received_samples = 0

        # STT scheduling
        self._last_stt_time = 0.0

        # Phrase state
        self._committed_words: list[tuple[str, float, float]] = []  # (word, start_s, end_s)
        self._last_word_end = 0.0
        self._synthesized_text = ""  # full text already synthesized, for dedup

        # Utterance detection (VAD for segmentation only)
        self._in_utterance = False
        self._silence_frames = 0
        self._silence_frame_threshold = int(UTTERANCE_SILENCE_S / (VAD_FRAME_MS / 1000.0))

        self._sequence = 0

    # ------------------------------------------------------------------ init

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return

            stt_model_size = self.config.stt.model_size
            stt_device = self.config.stt.device
            stt_compute = self.config.stt.compute_type
            beam = 5
            stt_engine = self.config.stt.engine
            log.info(f"Pipeline STT engine: {stt_engine}, model: {stt_model_size}, device: {stt_device}")

            if stt_engine == "parakeet":
                from backend.stt.parakeet_stt import ParakeetSTT
                self._stt = ParakeetSTT(
                    model_name=stt_model_size,
                    device=stt_device,
                    compute_type=stt_compute,
                    language=self.source_lang if self.source_lang != "auto" else None,
                )
            else:
                self._stt = FasterWhisperSTT(
                    model_size=stt_model_size,
                    device=stt_device,
                    compute_type=stt_compute,
                    beam_size=beam,
                )

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

            tts_engine_name = self.tts_engine_name
            select("tts", tts_engine_name,
                   allow_noncommercial=self.allow_nc,
                   allow_unknown=self.allow_unk)
            from backend.tts.base import TTS_ENGINES
            self._tts = TTS_ENGINES[tts_engine_name]()

            self._vad = _VADGate(aggressiveness=1)

            self._initialized = True
            log.info(
                "ContinuousPipeline %s: %s→%s ready (stt=%s mt=%s tts=%s)",
                self.speaker_id, self.source_lang, self.target_lang,
                self.config.stt.engine, self.config.mt.engine, self.tts_engine_name,
            )

    async def close(self) -> None:
        self._initialized = False
        self._stt = self._mt = self._tts = self._vad = None

    # ---------------------------------------------------------------- streaming

    async def process_audio_chunk(self, audio_np: np.ndarray, sample_rate: int) -> Optional[StreamingResult]:
        """Feed one PCM chunk (16kHz mono float32).

        Records audio continuously. Runs STT on overlapping windows every
        STT_STEP_S. Returns a StreamingResult when a complete phrase is
        translated and synthesized.
        """
        if not self._initialized:
            await self.initialize()

        now = time.perf_counter()

        # --- Always buffer audio (continuous recording) ---
        mono = audio_np if audio_np.ndim == 1 else audio_np.mean(axis=1)
        self._audio_buffer.append(mono)
        self._buffer_samples += len(mono)
        self._received_samples += len(mono)

        # --- VAD for utterance boundary detection ---
        # WebRTC VAD requires fixed frame sizes (10/20/30ms at 16kHz = 160/320/480 samples).
        # Process the chunk in 20ms frames and count speech vs silence.
        if self._vad:
            vad_frame_samples = VAD_FRAME_SAMPLES  # 320 at 16kHz (20ms)
            frame_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
            speech_frames = 0
            total_frames = 0
            for offset in range(0, len(frame_int16) - vad_frame_samples + 1, vad_frame_samples):
                frame = frame_int16[offset:offset + vad_frame_samples]
                if self._vad.is_speech(frame.tobytes()):
                    speech_frames += 1
                total_frames += 1
            is_speech = speech_frames > total_frames / 2 if total_frames > 0 else True
        else:
            is_speech = True

        if is_speech:
            self._silence_frames = 0
            if not self._in_utterance:
                self._in_utterance = True
        else:
            self._silence_frames += 1

        # --- Run STT on schedule ---
        if now - self._last_stt_time >= STT_STEP_S:
            self._last_stt_time = now
            result = await self._run_stt_and_process()
            if result:
                return result

        # --- Silence ended an utterance ---
        if self._in_utterance and self._silence_frames >= self._silence_frame_threshold:
            self._in_utterance = False
            result = await self._flush_phrase(final=True)
            if result:
                return result

        return None

    # ------------------------------------------------------------------ STT

    async def _run_stt_and_process(self) -> Optional[StreamingResult]:
        """Run STT on the current audio buffer window and process new text.

        Uses text-based deduplication: tracks synthesized text and only
        processes new text from each STT call. Works with engines that
        don't provide word-level timestamps (Parakeet).
        """
        if self._buffer_samples < int(MIN_UTTERANCE_S * AUDIO_SAMPLE_RATE):
            return None

        # Take the last STT_WINDOW_S of audio
        window_samples = int(STT_WINDOW_S * AUDIO_SAMPLE_RATE)
        if self._buffer_samples > window_samples:
            window = np.concatenate(self._audio_buffer)[-window_samples:]
            window_start_sample = self._buffer_start_sample + self._buffer_samples - window_samples
        else:
            window = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([], dtype=np.float32)
            window_start_sample = self._buffer_start_sample

        if len(window) == 0:
            return None

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
            log.error("ContinuousPipeline %s: STT error: %s", self.speaker_id, exc)
            return None

        # Combine all segment text into one transcript
        full_text = " ".join(getattr(seg, "text", "").strip() for seg in segments if getattr(seg, "text", "").strip())
        if not full_text:
            return None

        # Find new text: text in full_text that hasn't been synthesized yet
        new_text = self._find_new_text(full_text)
        if not new_text:
            return None

        # Update synthesized text tracker
        self._synthesized_text = (self._synthesized_text + " " + new_text).strip()

        # Synthesize the new text as a complete phrase
        return await self._synthesize_phrase(new_text, stt_time)

    def _find_new_text(self, full_text: str) -> str:
        """Find text in full_text that hasn't been synthesized yet.

        Uses simple prefix matching: if full_text starts with _synthesized_text,
        return the suffix. Otherwise, find the longest common prefix and
        return the remainder.
        """
        if not self._synthesized_text:
            return full_text

        # If full_text starts with what we already synthesized, return the suffix
        if full_text.startswith(self._synthesized_text):
            return full_text[len(self._synthesized_text):].strip()

        # Find longest common prefix
        min_len = min(len(full_text), len(self._synthesized_text))
        common = 0
        while common < min_len and full_text[common] == self._synthesized_text[common]:
            common += 1

        # Return text after the common prefix
        return full_text[common:].strip()

    def _extract_complete_phrases(self) -> list[tuple[str, float, float]]:
        """Group committed words into phrases by natural pauses.

        Returns all complete phrases except the last (which may continue
        as the speaker keeps talking).
        """
        if len(self._committed_words) < 2:
            return []

        phrases = []
        current_phrase = [self._committed_words[0]]

        for i in range(1, len(self._committed_words)):
            prev_end = self._committed_words[i - 1][2]
            curr_start = self._committed_words[i][1]
            gap = curr_start - prev_end

            if gap > SILENCE_GAP_S:
                # Natural pause = end of phrase
                phrase_text = " ".join(w[0] for w in current_phrase)
                phrases.append((phrase_text, current_phrase[0][1], current_phrase[-1][2]))
                current_phrase = [self._committed_words[i]]
            else:
                current_phrase.append(self._committed_words[i])

        # Keep only complete phrases; the last one stays in committed_words
        # in case more words follow it
        if phrases:
            # Remove words belonging to flushed phrases from committed_words
            flushed_words_count = sum(len(p[0].split()) for p in phrases)
            self._committed_words = self._committed_words[flushed_words_count:]

        return phrases

    async def _flush_phrase(self, final: bool = False) -> Optional[StreamingResult]:
        """Flush any remaining state at end of utterance/session.

        With text-based dedup, all text is synthesized in _run_stt_and_process,
        so there's nothing left to flush. Just reset phrase state.
        """
        # Reset for next utterance
        self._committed_words = []
        self._last_word_end = 0.0
        if final:
            self._synthesized_text = ""
        return None

    async def _synthesize_phrase(self, text: str, stt_time: float,
                                 final: bool = False) -> Optional[StreamingResult]:
        """Translate and synthesize one phrase as a complete utterance."""
        if not text or not text.strip():
            return None

        # Translate
        t_mt_start = time.perf_counter()
        try:
            translated, _ = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._mt.translate(text, self.source_lang, self.target_lang),
            )
        except Exception as exc:
            log.error("ContinuousPipeline %s: MT error: %s", self.speaker_id, exc)
            return None
        t_mt_end = time.perf_counter()

        if not translated or not translated.strip():
            return None

        # Synthesize as a complete utterance
        t_tts_start = time.perf_counter()
        try:
            audio_wav, sample_rate_out, tts_latency = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._tts.synthesize(
                    translated, language=self.target_lang,
                    speaker_wav_path=None,
                ),
            )
        except Exception as exc:
            log.error("ContinuousPipeline %s: TTS error: %s", self.speaker_id, exc)
            return None
        t_tts_end = time.perf_counter()

        # Encode to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio_wav, sample_rate_out, format="WAV")
        audio_bytes = buf.getvalue()

        self._sequence += 1
        result = StreamingResult(
            speaker_id=self.speaker_id,
            source_text=text,
            translated_text=translated,
            target_lang=self.target_lang,
            audio=audio_bytes,
            sample_rate=sample_rate_out,
            timestamp=time.perf_counter(),
            sequence=self._sequence,
            stt_ms=round(stt_time * 1000, 1),
            mt_ms=round((t_mt_end - t_mt_start) * 1000, 1),
            tts_ms=round((t_tts_end - t_tts_start) * 1000, 1),
        )
        return result

    async def flush_final(self) -> Optional[StreamingResult]:
        """Flush any remaining committed words."""
        return await self._flush_phrase(final=True)


# ---------------------------------------------------------------------------
# env helper
# ---------------------------------------------------------------------------

import os  # noqa: E402


def os_env(key: str, default: str) -> str:
    return os.environ.get(key, default)
