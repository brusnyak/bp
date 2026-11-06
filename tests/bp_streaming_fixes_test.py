"""Model-free production-body tests; avoid app import/config/model side effects."""
import ast
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
import io
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import typing

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def bodies(path, names=None, **bindings):
    """Execute exact source definitions, with model factories replaced by stubs.

    No application imports: main.py reads config and test/conftest.py deletes
    test_output. Neither belongs in this strictly model-free, nondeleting check.
    """
    tree = ast.parse((ROOT / path).read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and (names is None or n.name in names)]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected,
                        type_ignores=[])
    ns = dict(Optional=typing.Optional, np=np, asyncio=asyncio, io=io, sf=sf, logging=logging, json=json,
              os=os, deque=deque, dataclass=dataclass, field=field, __name__=__name__)
    ns.update(bindings)
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), 'exec'), ns)
    return SimpleNamespace(**ns)


class Clock:
    def __init__(self, now=10.0): self.now = now
    def perf_counter(self): return self.now
    def time(self): return self.now


@pytest.mark.parametrize('streaming', [False, True])
def test_ws_first_audio_metric_includes_three_second_endpoint_wait(streaming):
    async def run():
        clock = Clock(13.0)  # source onset 10, endpoint/STT dispatch 13
        import threading
        first_sent = threading.Event()
        class TTS:
            SUPPORTS_STREAMING = streaming
            sample_rate = 16000
            def synthesize(self, *args, **kwargs):
                clock.now += 0.2
                return np.ones(160, dtype=np.float32), 16000, 0.2
            def synthesize_stream(self, *args, **kwargs):
                clock.now += 0.2
                yield np.ones(160, dtype=np.float32)
                assert first_sent.wait(timeout=1.0)
                clock.now += 0.7
                yield np.ones(160, dtype=np.float32)
        class Socket:
            client = SimpleNamespace(host='fake', port=1)
            client_state = 'connected'
            timestamps = defaultdict(list)
            messages, sent_at = [], []
            async def send_text(self, data): self.messages.append(json.loads(data))
            async def send_bytes(self, data):
                assert data[:4] == b'RIFF'
                self.sent_at.append(clock.now)
                first_sent.set()
        ws = Socket()
        cfg = dict(source_lang='cs', target_lang='en', tts_model_choice='fake',
                   vad_enabled=True, speaker_wav_path=None, speaker_text=None, speaker_lang=None)
        session = dict(tts_engine=TTS(), tts_engine_name='fake',
                       stt_model=SimpleNamespace(transcribe_audio=lambda *a, **k: ([SimpleNamespace(text='ahoj')], 0, 'cs')),
                       mt_models={'cs-en': SimpleNamespace(translate=lambda *a: ('hello', 0))},
                       model_call_lock=asyncio.Lock())
        import threading
        m = bodies('backend/main.py', ['_process_speech_segment_pipeline'],
                   time=clock, threading=threading, torch=None, AUDIO_SAMPLE_RATE=16000,
                   active_sessions={'fake:1': session}, _log_session_event=lambda *a: None,
                   _log_text=str, _save_audio_segment_for_debug=lambda *a: None,
                   WebSocketState=SimpleNamespace(DISCONNECTED='disconnected'))
        await m._process_speech_segment_pipeline(ws, np.ones(160), 13.0,
                                                session_config=cfg, source_onset_time=10.0)
        metrics = next(x['metrics'] for x in ws.messages if x['type'] == 'final_metrics')
        assert metrics['onset_to_first_audio_s'] == pytest.approx(ws.sent_at[0] - 10.0)
        assert metrics['onset_to_first_audio_s'] == pytest.approx(3.2)
        assert metrics['first_audio_sent_time'] == ws.sent_at[0]
        assert metrics['source_onset_time'] == 10.0
        assert metrics['total_latency'] < metrics['onset_to_first_audio_s']
    asyncio.run(run())


def ingress(threshold):
    return bodies('backend/main.py', ['_queue_pre_vad_audio'],
                  SILENCE_RMS_THRESHOLD=threshold, AUDIO_SAMPLE_RATE=16000,
                  PRE_VAD_BUFFER_DURATION=0.5)._queue_pre_vad_audio


def test_disabled_rms_short_speech_and_zeros_reach_stub_vad_once_in_one_frame():
    feed = ingress(0)
    queue, pending, opened = deque(), deque(), False
    source = np.r_[np.linspace(.1, .2, 320, dtype=np.float32), np.zeros(640, dtype=np.float32)]
    frames = []
    for i in range(0, len(source), 160):
        opened = feed(source[i:i+160], queue, pending, opened)
        while len(queue) >= 320:
            frames.append(np.array([queue.popleft() for _ in range(320)]))
        if i == 160: assert len(frames) == 1  # first valid 20ms VAD quantum
    np.testing.assert_array_equal(np.concatenate(frames), source)
    assert not queue and not pending


def test_positive_rms_still_waits_half_second_and_uses_strict_threshold_once():
    feed = ingress(.125)
    queue, pending, opened = deque(), deque(), False
    # At/below threshold: same discarded admission-window behavior as before.
    assert not feed(np.full(8000, .125, dtype=np.float32), queue, pending, opened)
    assert not queue and not pending
    source = np.linspace(.25, .5, 8320, dtype=np.float32)
    assert not feed(source[:7680], queue, pending, False)
    assert len(pending) == 7680 and not queue
    opened = feed(source[7680:8000], queue, pending, False)
    assert opened and len(queue) == 8000 and not pending
    assert feed(source[8000:], queue, pending, opened)
    np.testing.assert_array_equal(np.array(queue), source)


def v2(clock):
    incremental = bodies('backend/stt/incremental.py', None,
                         re=__import__('re'), SENTENCE_END=('.', '!', '?', ';', ':'),
                         CLAUSE_END=('.', '!', '?', ';', ':', ','))
    resolver = bodies('backend/registry.py', ['target_tts_engine']).target_tts_engine
    m = bodies('backend/streaming/pipeline.py', None,
               time=clock, log=logging.getLogger('test'),
               Agreement=incremental.Agreement, Fragmenter=incremental.Fragmenter,
               target_tts_engine=resolver, AUDIO_SAMPLE_RATE=16000,
               MIN_AUDIO_S=1.0, VAD_FRAME_SAMPLES=320)
    p = m.StreamingPipeline('s', 'cs', 'en', '', 'piper', SimpleNamespace())
    p._initialized = True
    p._vad = SimpleNamespace(is_speech=lambda frame: True)
    p._mt = SimpleNamespace(translate=lambda text, *a: (text, 0))
    p._tts = SimpleNamespace(synthesize=lambda *a, **k: (np.ones(16, dtype=np.float32), 16000, 0))
    return p


def test_v2_resolves_english_voice_with_shared_resolver():
    p = v2(Clock())
    assert p.tts_engine_name == 'piper_generic_en'


def test_v2_trims_commits_rebases_spans_and_emits_every_word_once():
    async def run():
        clock = Clock()
        p = v2(clock)
        p._fragmenter.max_words = 2
        hypotheses = iter([
            [('prvni', .1, .4), ('druhy', .4, .8)],
            [('prvni', .1, .4), ('druhy', .4, .8)],
            [('treti', .1, .4), ('ctvrty', .4, .8)],
            [('treti', .1, .4), ('ctvrty', .4, .8)],
            [('konec', .1, .4)], [('konec', .1, .4)], [], [],
        ])
        windows = []
        def stt(audio, *a, **k):
            windows.append(audio.copy())
            words = [SimpleNamespace(word=w, start=a, end=b) for w, a, b in next(hypotheses)]
            return [SimpleNamespace(words=words)], 0, 'cs'
        p._stt = SimpleNamespace(transcribe_audio=stt)
        emitted = []
        for i in range(8):
            clock.now = 10 + i
            result = await p.process_audio_chunk(np.arange(i*16000, (i+1)*16000, dtype=np.float32), 16000)
            if result: emitted.append(result.source_text)
            if i == 1:
                assert p._buffer_start_sample == 12800
                assert p._fragmenter.span == (.1, .8)
            if i == 3:
                assert p._buffer_start_sample == 25600
                assert p._fragmenter.span == pytest.approx((.9, 1.6))
        assert ' '.join(emitted).split() == ['prvni', 'druhy', 'treti', 'ctvrty', 'konec']
        assert windows[2][0] == 12800  # cursor advances only by committed audio
        assert windows[4][0] == 25600
        assert await p.flush_final() is None
    asyncio.run(run())


def test_v2_one_word_flushes_on_clock_without_audio_or_new_commit():
    async def run():
        clock = Clock()
        p = v2(clock)
        p._fragmenter.push([('konec', 0, .2)], clock.now)
        assert await p.flush_pending() is None
        clock.now += p._fragmenter.max_wait_s + .001
        result = await p.flush_pending()
        assert result.source_text == 'konec' and result.audio[:4] == b'RIFF'
        assert await p.flush_pending() is None
    asyncio.run(run())


def test_ws_ingress_dispatches_all_samples_across_two_utterances(monkeypatch):
    """Run the real receive/VAD loop with auth, models, socket and clock stubbed."""
    import sys
    monkeypatch.setitem(sys.modules, 'backend.utils.auth',
                        SimpleNamespace(decode_access_token=lambda token: 'fake@example.invalid'))
    async def run():
        clock = Clock(10.84)
        samples = np.r_[np.linspace(.1, .2, 320, dtype=np.float32), np.zeros(6400, dtype=np.float32),
                        np.linspace(.3, .4, 320, dtype=np.float32), np.zeros(6400, dtype=np.float32)]
        seen, emissions = [], []
        def vad(frame, rate):
            seen.append(np.frombuffer(frame, dtype=np.int16).copy())
            clock.now += .02
            return bool(np.any(seen[-1]))
        class Socket:
            scope = {'client': ('fake', 1)}
            client_state = 'connected'
            def __init__(self):
                self.messages = iter([{'type': 'websocket.receive', 'text': '{"type":"start"}'},
                                      {'type': 'websocket.receive', 'bytes': samples.tobytes()},
                                      {'type': 'websocket.disconnect'}])
            async def receive(self):
                await asyncio.sleep(0)
                return next(self.messages)
            async def send_text(self, data): pass
        class DB:
            def query(self, *a): return self
            def filter(self, *a): return self
            def first(self): return SimpleNamespace(id=1)
            def close(self): pass
        cfg = dict(source_lang='cs', target_lang='en', tts_model_choice='fake', vad_enabled=True,
                   vad_silence_timeout_s=.3, vad_max_segment_s=3, incremental=False)
        session = dict(session_config=cfg, stt_model=SimpleNamespace(),
                       vad_instance=SimpleNamespace(is_speech=vad), captioner=None)
        async def emit(ws, audio, *args, **kwargs):
            emissions.append((audio.copy(), kwargs['source_onset_time']))
        m = bodies('backend/main.py', ['handle_audio_stream', '_queue_pre_vad_audio', '_Frames'],
                   time=clock, SessionLocal=DB, User=SimpleNamespace(email='email'),
                   active_sessions={'fake:1': session},
                   DEFAULT_TTS_MODEL='fake', DEFAULT_STT_MODEL_SIZE='fake',
                   SILENCE_TIMEOUT=.3, MAX_SEGMENT_S=3, AUDIO_SAMPLE_RATE=16000,
                   VAD_FRAME_DURATION=20, STREAMING_CHUNK_LENGTH=.5, PRE_VAD_BUFFER_DURATION=.5,
                   SILENCE_RMS_THRESHOLD=0, _run_pipeline_task=emit,
                   cleanup_session_models=lambda *a: None, _session_locks={}, _session_log_files={},
                   WebSocketDisconnect=type('Disconnected', (Exception,), {}),
                   websockets=SimpleNamespace(exceptions=SimpleNamespace(ConnectionClosed=type('Closed', (Exception,), {}))))
        await m.handle_audio_stream(Socket(), token='fake')
        np.testing.assert_array_equal(np.concatenate(seen), (samples * 32767).astype(np.int16))
        assert len(emissions) == 2
        np.testing.assert_array_equal(emissions[0][0], samples[:320])
        np.testing.assert_array_equal(emissions[1][0], samples[6720:7040])
        assert [e[1] for e in emissions] == pytest.approx([10.0, 10.42])
    asyncio.run(run())


def test_v2_initialize_calls_resolved_english_factory_only(monkeypatch):
    import sys
    constructed = []
    fake_registry = SimpleNamespace(select=lambda *a, **k: None,
                                    instantiate=lambda *a, **k: SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'backend.registry', fake_registry)
    async def run():
        p = v2(Clock())
        p._initialized = False
        p.config = SimpleNamespace(stt=SimpleNamespace(model_size='fake', device='cpu', compute_type='int8', engine='fake'),
                                   mt=SimpleNamespace(device='cpu', engine='fake'), tts=SimpleNamespace(engine='fake'))
        namespace = p.initialize.__globals__
        namespace['FasterWhisperSTT'] = lambda **k: SimpleNamespace()
        namespace['_VADGate'] = lambda: SimpleNamespace()
        namespace['TTS_ENGINES'] = {'piper_generic_en': lambda: constructed.append('en_US-john-medium') or SimpleNamespace()}
        await p.initialize()
        assert constructed == ['en_US-john-medium']
    asyncio.run(run())
