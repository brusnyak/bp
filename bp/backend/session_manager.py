"""BP v2 Multi-Speaker Meeting Session Manager.

Manages meeting rooms with multiple speaker and listener sessions. Each speaker
has a dedicated WebSocket session with its own language + voice. Listeners
subscribe to speaker→language output streams.

Architecture:
  MeetingRoom
    ├── SpeakerSession (per speaker WebSocket)
    │     ├── source_lang, voice, vad_state, stream_clock
    │     └── streaming_pipeline (STT→MT→TTS)
    └── ListenerSession (per listener WebSocket)
          ├── target_lang, output_device, playback_rate
          └── subscribes to one or more SpeakerSession outputs

The server mixes all speaker audio into one output stream per listener.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from backend.hardware_scan import autoconfigure, AutoConfig, HardwareProfile

log = logging.getLogger(__name__)


@dataclass
class SpeakerIdentity:
    """A speaker in the meeting."""
    session_id: str
    name: str
    source_lang: str
    voice: str            # TTS voice id for this speaker's output
    tts_engine: str       # "piper" | "omnivoice" | ...
    joined_at: float
    active: bool = True


@dataclass
class ListenerIdentity:
    """A listener in the meeting."""
    session_id: str
    name: str
    target_lang: str
    output_device: Optional[str]  # browser-selected sinkId or None
    playback_rate: float          # 1.0 | adaptive | 1.25 | ...
    joined_at: float
    active: bool = True


@dataclass
class TranslationResult:
    """A chunk of translated output ready for listeners."""
    speaker_session_id: str
    source_text: str
    translated_text: str
    target_lang: str
    audio_chunk: bytes          # WAV-encoded bytes
    sample_rate: int
    timestamp: float
    sequence: int               # per-speaker sequence for ordering


class SpeakerSession:
    """One speaker's capture + streaming pipeline state."""

    def __init__(self, identity: SpeakerIdentity, config: AutoConfig):
        self.identity = identity
        self.config = config
        self.pipeline_task: Optional[asyncio.Task] = None
        self.capture_active = False
        self.capture_generation = 0
        self.last_speech_time = 0.0
        self.sequence = 0
        # Models (lazily initialized per language pair)
        self.stt_model = None
        self.mt_model = None
        self.tts_engine = None
        self.vad_instance = None
        self._lock = asyncio.Lock()

    async def start(self, websocket_send):
        """Begin capture. websocket_send is a callback to send messages to this speaker's frontend."""
        self.capture_active = True
        self.capture_generation += 1
        self.last_speech_time = time.perf_counter()
        log.info("Speaker %s (%s) capture started", self.identity.session_id, self.identity.source_lang)

    async def stop(self):
        self.capture_active = False
        self.capture_generation += 1
        if self.pipeline_task and not self.pipeline_task.done():
            self.pipeline_task.cancel()
        log.info("Speaker %s capture stopped", self.identity.session_id)

    async def process_audio_chunk(self, audio_np, sample_rate, on_result):
        """Feed one PCM chunk into the speaker's VAD+streaming pipeline.

        on_result(TranslationResult) is called when a translated audio chunk is ready.
        """
        if not self.capture_active:
            return
        # VAD gate: only process if speech detected
        # (implementation delegates to the streaming pipeline module)
        await self._run_streaming(audio_np, sample_rate, on_result)

    async def _run_streaming(self, audio_np, sample_rate, on_result):
        """Core streaming loop: VAD → STT → MT → TTS → emit."""
        # Placeholder: actual implementation lives in backend/streaming/pipeline.py
        # This session class holds state; the streaming module does the work.
        raise NotImplementedError("Use backend.streaming.pipeline.StreamingPipeline")


class MeetingRoom:
    """One meeting room: N speakers, M listeners."""

    def __init__(self, room_id: str, config: AutoConfig):
        self.room_id = room_id
        self.config = config
        self.speakers: dict[str, SpeakerSession] = {}
        self.listeners: dict[str, ListenerIdentity] = {}
        self._speaker_lock = asyncio.Lock()
        self._listener_lock = asyncio.Lock()
        self.created_at = time.time()
        # Translation result broadcast queue: listener session_id -> list of results
        self._output_queues: dict[str, asyncio.Queue] = {}
        self._broadcast_task: Optional[asyncio.Task] = None

    async def add_speaker(self, session_id: str, name: str, source_lang: str,
                          voice: str, tts_engine: str) -> SpeakerIdentity:
        async with self._speaker_lock:
            if len(self.speakers) >= self.config.max_concurrent_sessions:
                raise RuntimeError(f"Meeting room full: {self.config.max_concurrent_sessions} speaker limit")
            identity = SpeakerIdentity(
                session_id=session_id, name=name, source_lang=source_lang,
                voice=voice, tts_engine=tts_engine, joined_at=time.time(),
            )
            session = SpeakerSession(identity, self.config)
            self.speakers[session_id] = session
            log.info("Room %s: speaker %s added (%s -> %s)", self.room_id, session_id, source_lang, voice)
            return identity

    async def remove_speaker(self, session_id: str):
        async with self._speaker_lock:
            session = self.speakers.pop(session_id, None)
            if session:
                await session.stop()
                log.info("Room %s: speaker %s removed", self.room_id, session_id)

    async def add_listener(self, session_id: str, name: str, target_lang: str,
                           output_device: Optional[str], playback_rate: float) -> ListenerIdentity:
        async with self._listener_lock:
            identity = ListenerIdentity(
                session_id=session_id, name=name, target_lang=target_lang,
                output_device=output_device, playback_rate=playback_rate,
                joined_at=time.time(),
            )
            self.listeners[session_id] = identity
            self._output_queues[session_id] = asyncio.Queue()
            log.info("Room %s: listener %s added (target %s)", self.room_id, session_id, target_lang)
            return identity

    async def remove_listener(self, session_id: str):
        async with self._listener_lock:
            self.listeners.pop(session_id, None)
            self._output_queues.pop(session_id, None)
            log.info("Room %s: listener %s removed", self.room_id, session_id)

    async def broadcast_result(self, result: TranslationResult):
        """Send a translation result to all listeners who want the target language."""
        # Determine which listeners need this (matching target_lang, or any if listener wants all)
        for listener_id, queue in self._output_queues.items():
            listener = self.listeners.get(listener_id)
            if not listener or not listener.active:
                continue
            if listener.target_lang == result.target_lang:
                try:
                    queue.put_nowait(result)
                except asyncio.QueueFull:
                    log.warning("Room %s: listener %s queue full, dropping result", self.room_id, listener_id)

    async def start_broadcast(self):
        """Start the broadcast loop that pulls from queues and sends to WebSockets."""
        # This is wired into the WebSocket handler in app.py
        pass


# ---------------------------------------------------------------------------
# Global meeting room registry
# ---------------------------------------------------------------------------

_rooms: dict[str, MeetingRoom] = {}
_rooms_lock = asyncio.Lock()
_auto_config: Optional[AutoConfig] = None


async def get_auto_config() -> AutoConfig:
    global _auto_config
    if _auto_config is None:
        _auto_config = autoconfigure()
    return _auto_config


async def get_or_create_room(room_id: str) -> MeetingRoom:
    async with _rooms_lock:
        if room_id not in _rooms:
            config = await get_auto_config()
            _rooms[room_id] = MeetingRoom(room_id, config)
        return _rooms[room_id]


async def remove_room(room_id: str):
    async with _rooms_lock:
        room = _rooms.pop(room_id, None)
        if room:
            for sid in list(room.speakers.keys()):
                await room.remove_speaker(sid)
            for sid in list(room.listeners.keys()):
                await room.remove_listener(sid)
            log.info("Room %s removed", room_id)


async def list_rooms() -> list[str]:
    async with _rooms_lock:
        return list(_rooms.keys())


def get_config() -> Optional[AutoConfig]:
    return _auto_config