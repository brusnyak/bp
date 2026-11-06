"""Model-free checks of pack -> production preparation -> STT factory."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
from bp_streaming_fixes_test import bodies, ROOT

# Load only the pure profile module; never import app or backend.main.
spec = importlib.util.spec_from_file_location('backend.latency_profiles', ROOT / 'backend/latency_profiles.py')
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)
PACK = json.loads((ROOT / 'languages/packs/cs-en.json').read_text())
job_spec = importlib.util.spec_from_file_location('route_job', ROOT / 'scripts/gpu_bench/kaggle_full_route/job.py')
job = importlib.util.module_from_spec(job_spec)
job_spec.loader.exec_module(job)


def test_profile_is_opt_in_and_does_not_mutate_pack():
    before = json.dumps(PACK, sort_keys=True)
    default, _ = profiles.resolve_profile(PACK)
    fast, name = profiles.resolve_profile(PACK, 'low_latency')
    assert default['stt'] == PACK['stt'] and 'device' not in default['stt']
    assert fast['stt'] == dict(engine='faster_whisper', language='cs', model='base',
                               device='cpu', compute_type='int8', beam_size=5)
    assert fast['vad']['silence_timeout_s'] == .3
    assert fast['vad']['max_segment_s'] == 3
    assert fast['vad']['incremental'] is False
    assert name == 'low_latency' and json.dumps(PACK, sort_keys=True) == before
    with pytest.raises(ValueError): profiles.resolve_profile(PACK, 'typo')


@pytest.mark.parametrize('profile,maximum,silence,incremental', [
    ('low_latency', 3., .3, False), ('P_short', 1.5, .15, False),
    ('P_short2', 1., .12, False), ('P_incr', 3., .3, True)])
def test_profile_reaches_production_factory_and_existing_session(monkeypatch, profile, maximum, silence, incremental):
    monkeypatch.setitem(sys.modules, 'backend.latency_profiles', profiles)
    monkeypatch.setitem(sys.modules, 'scripts.language_packs', SimpleNamespace(load=lambda pair: PACK))
    for key in ('BP_PROFILE', 'BP_VAD_SILENCE_S', 'BP_VAD_MAX_SEGMENT_S', 'BP_INCREMENTAL', 'BP_STT_BEAM_SIZE'):
        monkeypatch.delenv(key, raising=False)
    seen = []
    async def pool(stage, engine, factory, **kw): return factory(), True
    async def noop(*a, **kw): pass
    m = bodies('backend/main.py', ['initialize_all_models', '_initialize_stt_model'],
               time=time, DEFAULT_STT_MODEL_SIZE='base', active_sessions={}, _safe_speaker_wav=lambda p: p,
               SILENCE_TIMEOUT=.25, MAX_SEGMENT_S=3., BP_DEVICE='cuda', BP_COMPUTE_TYPE='float16',
               _pick_stt_model=lambda lang, size: size, local_stt_source=lambda size: size,
               instantiate=lambda *a, **kw: seen.append(kw) or SimpleNamespace(),
               get_pooled_model=pool, release_pooled_model=lambda *a, **kw: None,
               _initialize_mt_model=noop, _initialize_tts_models=noop,
               _initialize_vad_instance=noop, _initialize_captioner=noop)
    async def run():
        for key, value in job.profile_env(profile).items(): monkeypatch.setenv(key, value)
        await m.initialize_all_models('test', 'cs', 'en', 'piper', 'small')
        session = m.initialize_all_models.__globals__['active_sessions']['test']
        if profile == 'low_latency':
            assert seen[-1] == dict(model_size='base', device='cpu', compute_type='int8', beam_size=5)
        else:
            assert seen[-1]['device'] == 'cuda' and seen[-1]['compute_type'] == 'float16'
        assert session['session_config']['vad_silence_timeout_s'] == silence
        assert session['session_config']['vad_max_segment_s'] == maximum
        assert session['session_config']['incremental'] is incremental
        # An explicit env endpoint override retains the documented precedence.
        monkeypatch.setenv('BP_VAD_SILENCE_S', '.2')
        await m.initialize_all_models('test', 'cs', 'en', 'piper')
        assert session['session_config']['vad_silence_timeout_s'] == .2
        for key in job.profile_env(profile): monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv('BP_VAD_SILENCE_S', raising=False)
        monkeypatch.setenv('BP_PROFILE', 'default')
        await m.initialize_all_models('test', 'cs', 'en', 'piper')
        assert seen[-1]['device'] == 'cuda' and seen[-1]['compute_type'] == 'float16'
        assert 'beam_size' not in session['session_config']
        assert session['session_config']['latency_profile'] == 'default'
        assert session['session_config']['vad_max_segment_s'] == 3.
        assert session['session_config']['incremental'] is False
    asyncio.run(run())


def test_benchmark_profile_validation_and_fragment_defaults():
    assert job.profile_env('default') == {'BP_PROFILE': 'default'}
    assert job.profile_env('low_latency') == {'BP_PROFILE': 'low_latency'}
    with pytest.raises(ValueError): job.profile_env('typo')
    for name in ('P_short', 'P_short2', 'P_incr'):
        env = job.profile_env(name)
        assert env['BP_PROFILE'] == 'default'
        assert not any(key.startswith('BP_INCR_') for key in env)
    model_free = bodies('backend/main.py', ['_IncState'], Agreement=lambda: None,
                        Fragmenter=lambda **kwargs: kwargs)
    # Inspect constructor defaults via the actual source body without real models.
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {}, clear=True):
        assert model_free._IncState().frag == dict(min_words=4, max_words=9, max_wait_s=1.6)


def test_word_diff_is_relative_not_quality():
    assert job.word_diff('a b c', 'a b b')['duplicated_words'] == 1
    assert job.word_diff('a b c', 'a b b')['dropped_words'] == 1
    assert job.word_diff('a b c', 'a b b')['inserted_words'] == 1
    assert job.word_diff('Český text.', 'český text')['dropped_words'] == 0
