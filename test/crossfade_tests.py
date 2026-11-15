"""Model-free waveform and queue tests; no server or TTS models."""
import asyncio

import numpy as np
import pytest

from backend.prosody import crossfade_audio, fade_fragment, PlayoutScheduler


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 48000])
@pytest.mark.parametrize("ms", [0, 40, 60, 80])
def test_exact_overlap_length_and_no_clipping(rate, ms):
    a, b = np.ones(5000), np.ones(4000)
    out = crossfade_audio(a, b, rate, ms)
    assert len(out) == len(a) + len(b) - min(len(a), len(b), round(rate * ms / 1000))
    assert np.isfinite(out).all() and np.max(np.abs(out)) <= 1
    assert np.all(a == 1) and np.all(b == 1)


def test_equal_power_weights_with_headroom():
    a, b = np.full(100, 0.2), np.full(100, -0.1)
    out = crossfade_audio(a, b, 1000, 60)
    theta = np.linspace(0, np.pi / 2, 60)
    np.testing.assert_allclose(out[40:100], 0.2 * np.cos(theta) - 0.1 * np.sin(theta), atol=1e-7)


@pytest.mark.parametrize("a,b", [(0, 0), (0, 100), (100, 0), (1, 1), (10, 20)])
def test_empty_and_short_inputs(a, b):
    out = crossfade_audio(np.ones(a), -np.ones(b), 24000, 60)
    assert len(out) == a + b - min(a, b, 1440)
    assert np.isfinite(out).all() and np.all(np.abs(out) <= 1)


def test_fades_taper_edges_without_mutation_or_length_change():
    audio = np.ones(1000)
    out = fade_fragment(audio, 1000, fade_out_ms=60)
    assert len(out) == len(audio)
    assert out[0] == out[-1] == 0
    assert out[4] == out[-60] == 1
    assert np.all(out[4:-60] == 1) and np.all(audio == 1)


def test_same_clause_scheduler_adds_no_silence():
    s = PlayoutScheduler(extra_lag_s=0.6)
    s.lead_silence(0, 1)
    s.committed(0.1)
    assert s.lead_silence(2, 2.5, continuation=True) == 0
    s.committed(0.2)
    assert s.play_end == pytest.approx(2.7)


@pytest.mark.asyncio
async def test_transport_joins_only_unsent_tail_and_releases_final_tail():
    from backend.main import _send_fragment_audio
    sent, session = [], {}

    async def send(audio, rate):
        sent.append(audio.copy())

    await _send_fragment_audio(session, np.full(1000, 0.2), 1000, "utterance", send, 60)
    assert sum(map(len, sent)) == 940
    await _send_fragment_audio(session, np.full(1000, 0.3), 1000, "utterance", send, 60)
    await session["audio_join_timer"]
    assert sum(map(len, sent)) == 1940
    out = np.concatenate(sent)
    assert out[0] == out[-1] == 0
    assert np.max(np.abs(out)) <= 1
    assert "audio_join_tail" not in session


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["utterance", "sample_rate", "gap", "disabled"])
async def test_transport_does_not_overlap_across_boundaries(boundary):
    from backend.main import _send_fragment_audio
    sent, session = [], {}

    async def send(audio, rate):
        sent.append((audio.copy(), rate))

    await _send_fragment_audio(session, np.ones(1000), 1000, "first", send, 60)
    key, rate, ms = "first", 1000, 60
    if boundary == "utterance":
        key = "second"
    elif boundary == "sample_rate":
        rate = 2000
    elif boundary == "disabled":
        ms = 0
    else:
        # Age the saved arrival without wall-clock sleeps.
        tail, sr, k, arrived, sender = session["audio_join_tail"]
        session["audio_join_tail"] = (tail, sr, k, arrived - 0.15, sender)
    await _send_fragment_audio(session, np.ones(1000), rate, key, send, ms)
    if "audio_join_timer" in session:
        await session["audio_join_timer"]
    assert sum(len(a) for a, _ in sent) == 2000
    assert sent[1][0][-1] == 0  # previous fragment's isolated tail fades out


@pytest.mark.asyncio
async def test_disabled_transport_is_exact_passthrough():
    from backend.main import _send_fragment_audio
    sent, session = [], {}
    audio = np.linspace(-1, 1, 1000)

    async def send(samples, rate):
        sent.append(samples)

    await _send_fragment_audio(session, audio, 1000, "u", send, 0)
    np.testing.assert_array_equal(sent[0], audio)
    assert "audio_join_timer" not in session


@pytest.mark.asyncio
async def test_lead_silence_keeps_actual_speech_fade_and_prevents_overlap():
    from backend.main import _send_fragment_audio
    sent, session = [], {}

    async def send(audio, rate):
        sent.append(audio.copy())

    await _send_fragment_audio(session, np.ones(1000), 1000, "u", send, 60)
    await _send_fragment_audio(session, np.ones(1000), 1000, "u", send, 60, lead_s=0.2)
    await session["audio_join_timer"]
    assert sum(map(len, sent)) == 2200
    assert np.all(sent[2][:201] == 0)
    assert sent[2][204] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stopped", [False, True])
async def test_real_pipeline_wires_clause_continuation_and_stop_guard(monkeypatch, stopped):
    import io
    from collections import defaultdict
    from types import SimpleNamespace
    import soundfile as sf
    from starlette.websockets import WebSocketState
    from backend import main as bm
    from backend.tts.piper_tts import PiperTTS

    class Socket:
        client = SimpleNamespace(host="join-test", port=1)
        client_state = WebSocketState.CONNECTED
        scope = {"client": ("join-test", 1)}

        def __init__(self):
            self.blobs, self.timestamps = [], defaultdict(list)

        async def send_bytes(self, data):
            self.blobs.append(data)

        async def send_text(self, text):
            pass

    class Scheduler:
        def __init__(self):
            self.continuations = []

        def lead_silence(self, onset, now, continuation=False):
            self.continuations.append(continuation)
            return 0 if continuation else 0.2

        def committed(self, duration):
            pass

    tts = object.__new__(PiperTTS)
    tts.sample_rate = 1000
    tts.synthesize_stream = lambda **kwargs: iter([np.full(1000, 0.2)])
    scheduler = Scheduler()
    session = {"tts_engine": tts, "tts_engine_name": "piper", "model_call_lock": asyncio.Lock(),
               "mt_models": {"en-sk": SimpleNamespace(translate=lambda *a: ("translated words", 0))},
               "capture_generation": 1, "capture_active": True, "prosody_sched": scheduler}
    monkeypatch.setitem(bm.active_sessions, "join-test:1", session)
    monkeypatch.setattr(bm, "_log_session_event", lambda *a, **k: None)
    monkeypatch.setattr(bm._prosody, "ENABLED", True)
    config = {"source_lang": "en", "target_lang": "sk", "tts_model_choice": "piper", "vad_enabled": True,
              "speaker_wav_path": None, "speaker_text": None, "speaker_lang": None, "xfade_ms": 60}
    ws = Socket()
    for text in ("Hello world", "more words"):
        await bm._process_speech_segment_pipeline(
            ws, np.zeros(1), 1, session_config=config, pretranscribed=text,
            source_span=(0, 0.5), source_onset_time=0.5)
    if stopped:
        session["capture_active"] = False
        session["capture_generation"] += 1
    await session["audio_join_timer"]
    assert scheduler.continuations == [False, True]
    audio = np.concatenate([sf.read(io.BytesIO(b))[0] for b in ws.blobs])
    assert len(audio) == 2140 - (60 if stopped else 0)
    assert audio[200] == 0  # fade begins at speech onset, after the initial silence
    if not stopped:
        assert audio[-1] == 0


def test_correlated_full_scale_has_no_gain_step_at_overlap_edges():
    out = crossfade_audio(np.ones(100), np.ones(100), 1000, 60)
    np.testing.assert_allclose(out, 1, atol=1e-7)


def test_one_sample_is_faded_to_silence():
    np.testing.assert_array_equal(fade_fragment(np.ones(1), 1000, fade_out_ms=60), [0])
