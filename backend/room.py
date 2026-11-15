"""Room mode: one speaker, many listeners, each with their own language (the earpiece scenario).

The speaker's normal /ws session recognises speech once. Every recognised phrase is published to the room; the room translates it ONCE per distinct
listener language and synthesises it ONCE per language, then sends the same audio to every listener of that language. Cost therefore grows with the
number of languages, not with the number of listeners. Each language has its own ordered queue (so phrases arrive in speaking order) and its own
playout scheduler (so every listener hears the speaker's pauses, see backend/prosody.py).

Listeners connect to /ws/listen?room=ID&lang=de&token=... (app.py). They receive JSON captions and binary WAV chunks.
Voices: the generic Piper voice of the target language (the speaker's own fine-tune where one exists, e.g. Slovak). Cloned voices are not part of this.
Feedback rule: listeners must wear earpieces; the speaker's microphone must not pick up the translated audio.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend import prosody

log = logging.getLogger(__name__)
MAX_LISTENERS = int(os.environ.get("BP_ROOM_MAX_LISTENERS", "50"))
MAX_LANGS = int(os.environ.get("BP_ROOM_MAX_LANGS", "6"))
SEND_TIMEOUT_S = float(os.environ.get("BP_ROOM_SEND_TIMEOUT_S", "3.0"))   # a listener that cannot take a phrase in this time is dropped, the others are not delayed

Translate = Callable[[str, str, str], Awaitable[str]]
Synth = Callable[[str, str], Awaitable[Tuple[np.ndarray, int]]]


@dataclass
class Listener:
    id: str
    ws: Any
    lang: str
    joined: float = field(default_factory=time.time)
    captions: int = 0
    audio_chunks: int = 0


class _Channel:
    """Everything for one target language: ordered queue, worker, playout scheduler."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.sched = prosody.PlayoutScheduler()
        self.task: Optional[asyncio.Task] = None
        self.seq = 0


class Room:
    def __init__(self, room_id: str) -> None:
        self.room_id = room_id
        self.listeners: Dict[str, Listener] = {}
        self.channels: Dict[str, _Channel] = {}
        self.created = time.time()
        self.published = 0
        self.owner: Optional[str] = None        # guest id of the speaker who created the room (None: unowned, tests)
        self.src_lang: Optional[str] = None     # language the speaker's session recognises, set at the first publish

    # -- membership ---------------------------------------------------------------------------------------------------
    def languages(self) -> set:
        return {l.lang for l in self.listeners.values()}

    def add_listener(self, ws: Any, lang: str) -> Listener:
        if len(self.listeners) >= MAX_LISTENERS:
            raise ValueError(f"room is full ({MAX_LISTENERS} listeners)")
        if lang not in self.languages() and len(self.languages()) >= MAX_LANGS:
            raise ValueError(f"room already serves {MAX_LANGS} languages")
        lst = Listener(id=secrets.token_hex(4), ws=ws, lang=lang)
        self.listeners[lst.id] = lst
        return lst

    def remove_listener(self, listener_id: str) -> None:
        self.listeners.pop(listener_id, None)

    def set_language(self, listener_id: str, lang: str) -> None:
        lst = self.listeners.get(listener_id)
        if lst and lang != lst.lang:
            if lang not in self.languages() and len(self.languages()) >= MAX_LANGS:
                raise ValueError(f"room already serves {MAX_LANGS} languages")
            lst.lang = lang

    # -- publishing ---------------------------------------------------------------------------------------------------
    def publish(self, text: str, src_lang: str, onset: Optional[float] = None, translate: Optional[Translate] = None,
                synth: Optional[Synth] = None, now: Callable[[], float] = time.perf_counter) -> int:
        """Queue one recognised phrase for every listener language. Synchronous, so phrases keep their order. Returns the number of languages queued."""
        if not text or not text.strip():
            return 0
        n = 0
        for lang in self.languages():
            ch = self.channels.setdefault(lang, _Channel())
            if ch.task is None or ch.task.done():
                ch.task = asyncio.get_running_loop().create_task(self._worker(lang, ch))
            ch.queue.put_nowait((text, src_lang, onset, translate or default_translate, synth or default_synth, now, now()))
            n += 1
        self.published += 1
        return n

    async def drain(self) -> None:
        """Wait until every queued phrase has been delivered (tests, shutdown)."""
        for ch in list(self.channels.values()):
            await ch.queue.join()

    def close(self) -> None:
        for ch in self.channels.values():
            if ch.task and not ch.task.done():
                ch.task.cancel()

    async def _worker(self, lang: str, ch: _Channel) -> None:
        while True:
            item = await ch.queue.get()
            try:
                await self._deliver(lang, ch, item)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("room %s: delivery to '%s' failed", self.room_id, lang)
            finally:
                ch.queue.task_done()

    async def _deliver(self, lang: str, ch: _Channel, item: tuple) -> None:
        text, src_lang, onset, translate, synth, now, queued_at = item
        t_start = now()
        targets: List[Listener] = [l for l in self.listeners.values() if l.lang == lang]
        if not targets:
            return
        ch.seq += 1
        caption = {"type": "room_caption", "lang": lang, "seq": ch.seq, "source": text}
        wav: Optional[bytes] = None
        if lang == src_lang:                                  # the listener understands the speaker: captions only, no synthetic voice
            caption["translated"] = text
        else:
            t_mt = now()
            translated = (await translate(text, src_lang, lang)).strip()
            mt_s = now() - t_mt
            if not translated:
                return
            caption["translated"] = translated
            t_tts = now()
            audio, sr = await synth(translated, lang)
            tts_s = now() - t_tts
            lead = ch.sched.lead_silence(onset, now()) if (prosody.ENABLED and onset is not None) else 0.0
            if lead > 0:
                audio = np.concatenate([np.zeros(int(sr * lead), dtype=np.asarray(audio).dtype), np.asarray(audio)])
            ch.sched.committed(len(audio) / sr - lead)
            import soundfile as sf
            buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV"); wav = buf.getvalue()
        t_send = now()
        await asyncio.gather(*(self._send(lst, caption, wav) for lst in targets))   # concurrent: one slow phone must not delay the rest
        if wav is not None:   # where a phrase's time went in this language, so a room run explains its own lag (queue = waited behind earlier phrases of the same language)
            log.info("ROOMPARTS lang=%s seq=%d listeners=%d queue=%.2f mt=%.2f tts=%.2f send=%.2f", lang, ch.seq, len(targets), t_start - queued_at, mt_s, tts_s, now() - t_send)

    async def _send(self, lst: Listener, caption: dict, wav: Optional[bytes]) -> None:
        try:
            async def go():
                await lst.ws.send_text(json.dumps(caption))
                lst.captions += 1
                if wav is not None:
                    await lst.ws.send_bytes(wav)
                    lst.audio_chunks += 1
            await asyncio.wait_for(go(), SEND_TIMEOUT_S)
        except Exception:                                     # dead or too slow: drop it, never stall the others
            self.remove_listener(lst.id)
            try:
                await asyncio.wait_for(lst.ws.close(), 1.0)
            except Exception:
                pass


# -- default engines: shared across rooms, built on first use -----------------------------------------------------------------------------
_mt_cache: Dict[str, Tuple[Any, asyncio.Lock]] = {}
_tts_cache: Dict[str, Tuple[Any, asyncio.Lock]] = {}


async def default_translate(text: str, src: str, tgt: str) -> str:
    key = f"{src}-{tgt}"
    if key not in _mt_cache:
        from backend.registry import instantiate
        path = f"ct2_models/Helsinki-NLP--opus-mt-{src}-{tgt}"
        if not os.path.isfile(os.path.join(path, "model.bin")):
            raise RuntimeError(f"translation model for {key} is not installed ({path}); run scripts/setup.py --langs ...")
        _mt_cache[key] = (instantiate("mt", "opus-mt", model_path=path, device=os.environ.get("BP_MT_DEVICE", "cpu")), asyncio.Lock())
    mt, lock = _mt_cache[key]
    async with lock:
        out, _ = await asyncio.get_running_loop().run_in_executor(None, lambda: mt.translate(text, src, tgt))
    return out


async def default_synth(text: str, lang: str) -> Tuple[np.ndarray, int]:
    if lang not in _tts_cache:
        from backend.registry import instantiate, target_tts_engine
        _tts_cache[lang] = (instantiate("tts", target_tts_engine("piper", lang)), asyncio.Lock())
    tts, lock = _tts_cache[lang]
    tts_lang = getattr(tts, "LANGUAGE_OVERRIDES", {}).get(lang, lang)
    async with lock:
        wav, sr, _ = await asyncio.get_running_loop().run_in_executor(None, lambda: tts.synthesize(text, language=tts_lang))
    return np.asarray(wav), int(sr)


def pair_unavailable(src: str, tgt: str) -> Optional[str]:
    """Why a listener in `tgt` cannot be served from a speaker in `src` (no declared pack, or its model assets are not installed); None when it can."""
    if src == tgt:
        return None
    try:
        from scripts.language_packs import all_packs, resolve
        pack = next((p for p in all_packs() if p["source"] == src and p["target"] == tgt), None)
        if pack is None:
            return f"no {src} to {tgt} route is declared on this server"
        errors = resolve(pack).get("errors") or []
        return f"{src} to {tgt} is declared but not installed: {'; '.join(errors)[:160]}" if errors else None
    except Exception as e:          # never block a room on a broken inventory: the delivery path still drops phrases it cannot translate
        log.warning("room pair check failed: %s", e)
        return None


def listen_url(base_url: str, room_id: str, lan_ip: Optional[str] = None) -> str:
    """The link a phone must open. A page opened as localhost gives a link no phone can reach, so localhost/127.0.0.1 is swapped for the machine's LAN address."""
    from urllib.parse import quote, urlsplit, urlunsplit
    parts = urlsplit(base_url)
    host = parts.hostname or ""
    if host in ("localhost", "127.0.0.1", "::1") and lan_ip:
        parts = parts._replace(netloc=f"{lan_ip}:{parts.port}" if parts.port else lan_ip)
    return urlunsplit((parts.scheme, parts.netloc, "/ui/listen/listen.html", f"room={quote(room_id)}", ""))


def lan_address() -> Optional[str]:
    """This machine's address on the local network (no packet is sent), None when offline."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            ip = sk.getsockname()[0]
            return None if ip.startswith("127.") else ip
    except Exception:
        return None


def qr_svg(url: str) -> str:
    import io
    import segno
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="svg", scale=5, border=2, xmldecl=False, svgns=True)
    return buf.getvalue().decode("utf-8")


# -- registry -----------------------------------------------------------------------------------------------------------------------------
_ROOMS: Dict[str, Room] = {}


def new_room_id() -> str:
    return secrets.token_urlsafe(6)


def get_room(room_id: str, create: bool = True) -> Optional[Room]:
    if room_id not in _ROOMS and create:
        _ROOMS[room_id] = Room(room_id)
    return _ROOMS.get(room_id)


def close_room(room_id: str) -> None:
    room = _ROOMS.pop(room_id, None)
    if room:
        room.close()


async def end_room(room_id: str) -> None:
    """The speaker left: tell every listener, close their sockets, free the room (a room id is not reusable by guessing)."""
    room = _ROOMS.pop(room_id, None)
    if room is None:
        return
    room.close()
    for lst in list(room.listeners.values()):
        try:
            await asyncio.wait_for(lst.ws.send_text(json.dumps({"type": "room_closed"})), 1.0)
            await asyncio.wait_for(lst.ws.close(), 1.0)
        except Exception:
            pass
    room.listeners.clear()
