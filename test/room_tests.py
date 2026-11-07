"""Model-free tests for room mode: fan-out per language, ordering, pauses, same-language captions, dead sockets, limits."""
import asyncio
import json

import numpy as np
import pytest

from backend import room as R


class FakeWS:
    def __init__(self, fail=False):
        self.texts, self.blobs, self.fail = [], [], fail

    async def send_text(self, t):
        if self.fail: raise RuntimeError("closed")
        self.texts.append(json.loads(t))

    async def send_bytes(self, b):
        if self.fail: raise RuntimeError("closed")
        self.blobs.append(b)


async def fake_translate(text, src, tgt):
    fake_translate.calls.append((text, src, tgt)); return f"{tgt}:{text}"
fake_translate.calls = []


async def fake_synth(text, lang):
    fake_synth.calls.append((text, lang)); return np.ones(1600, dtype=np.float32) * 0.1, 16000
fake_synth.calls = []


def run(coro):
    return asyncio.run(coro)


def fresh():
    fake_translate.calls.clear(); fake_synth.calls.clear()
    return R.Room("t")


def test_one_translation_and_one_voice_per_language_not_per_listener():
    async def go():
        room = fresh(); a, b, c = FakeWS(), FakeWS(), FakeWS()
        room.add_listener(a, "de"); room.add_listener(b, "de"); room.add_listener(c, "sk")
        assert room.publish("hello", "en", translate=fake_translate, synth=fake_synth) == 2
        await room.drain()
        assert sorted(x[2] for x in fake_translate.calls) == ["de", "sk"] and len(fake_synth.calls) == 2   # 3 listeners, 2 languages
        assert len(a.blobs) == len(b.blobs) == len(c.blobs) == 1 and a.blobs[0] == b.blobs[0]               # identical audio to both German listeners
        assert a.texts[0]["translated"] == "de:hello" and c.texts[0]["lang"] == "sk"
        room.close()
    run(go())


def test_phrases_keep_speaking_order_per_language():
    async def go():
        room = fresh(); w = FakeWS(); room.add_listener(w, "de")
        for t in ("one", "two", "three"): room.publish(t, "en", translate=fake_translate, synth=fake_synth)
        await room.drain()
        assert [c["source"] for c in w.texts] == ["one", "two", "three"] and [c["seq"] for c in w.texts] == [1, 2, 3]
        room.close()
    run(go())


def test_same_language_listener_gets_captions_only():
    async def go():
        room = fresh(); w = FakeWS(); room.add_listener(w, "en")
        room.publish("hello", "en", translate=fake_translate, synth=fake_synth); await room.drain()
        assert w.texts[0]["translated"] == "hello" and w.blobs == [] and fake_translate.calls == []
        room.close()
    run(go())


def test_speaker_pause_becomes_leading_silence():
    async def go():
        room = fresh(); w = FakeWS(); room.add_listener(w, "de")
        t = [0.0]
        room.publish("a", "en", onset=0.0, translate=fake_translate, synth=fake_synth, now=lambda: 1.0)
        room.publish("b", "en", onset=3.0, translate=fake_translate, synth=fake_synth, now=lambda: 3.2)   # speaker paused ~2 s: the output must too
        await room.drain()
        import io, soundfile as sf
        n = [len(sf.read(io.BytesIO(b))[0]) for b in w.blobs]
        assert n[1] > n[0] + 8000                                  # second chunk carries extra leading silence (>0.5 s at 16 kHz)
        room.close()
    run(go())


def test_dead_listener_is_dropped_and_others_still_served():
    async def go():
        room = fresh(); good, bad = FakeWS(), FakeWS(fail=True)
        room.add_listener(good, "de"); lb = room.add_listener(bad, "de")
        room.publish("x", "en", translate=fake_translate, synth=fake_synth); await room.drain()
        assert lb.id not in room.listeners and len(good.blobs) == 1
        room.close()
    run(go())


def test_limits_and_language_switch():
    room = R.Room("lim")
    for i in range(R.MAX_LANGS): room.add_listener(FakeWS(), f"l{i}")
    with pytest.raises(ValueError): room.add_listener(FakeWS(), "one-too-many")
    lst = next(iter(room.listeners.values())); room.set_language(lst.id, "l1"); assert lst.lang == "l1"


def test_empty_text_and_empty_room_publish_nothing():
    async def go():
        room = fresh(); assert room.publish("hello", "en") == 0          # nobody listening
        room.add_listener(FakeWS(), "de"); assert room.publish("   ", "en") == 0
        room.close()
    run(go())
