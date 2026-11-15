"""Model-free checks of the fragment trace clock and missing measurements."""
import json
import pytest
from backend.lag_trace import build_lag_trace


def test_stream_clock_and_durations():
    trace = build_lag_trace(source_span=(1, 2), stream_origin=100,
                            emitted_at=103, playback_at=104,
                            hold_s=.5, stt_s=.125, mt_s=.25,
                            tts_started_at=103.5, text="Hello world")
    assert trace == dict(type="lag_trace", speech_start=1000, speech_end=2000, emit_time=3000,
                        hold_ms=500, stt_ms=125, mt_ms=250, tts_first_ms=500,
                        playback_start=4000, total_ms=2000, words=2, text="Hello world")


def test_unknown_measurements_stay_null():
    trace = build_lag_trace(text="Živá ukážka")
    assert all(trace[key] is None for key in (
        "speech_end", "emit_time", "hold_ms", "stt_ms", "mt_ms",
        "tts_first_ms", "playback_start", "total_ms"))
    assert trace["words"] == 2
    assert json.loads(json.dumps(trace, allow_nan=False)) == trace


def test_no_origin_does_not_invent_stream_timestamps():
    trace = build_lag_trace(source_span=(0, 2), emitted_at=104,
                            playback_at=105, tts_started_at=104)
    assert trace["speech_end"] == 2000
    assert trace["emit_time"] is trace["playback_start"] is trace["total_ms"] is None
    assert trace["tts_first_ms"] == 1000


def test_incremental_bypass_is_not_measured_stt():
    trace = build_lag_trace(source_span=(1, 2), stream_origin=10,
                            emitted_at=13, hold_s=0)
    assert trace["stt_ms"] is None
    assert trace["hold_ms"] == 0
    assert trace["total_ms"] is None  # no audio was sent


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurements_are_json_null(bad):
    trace = build_lag_trace(source_span=(0, bad), stream_origin=0,
                            emitted_at=bad, hold_s=bad, stt_s=bad,
                            mt_s=bad, playback_at=bad, tts_started_at=0)
    json.dumps(trace, allow_nan=False)
    assert trace["total_ms"] is None


def test_does_not_clamp_inconsistent_clocks_or_sum_stages():
    trace = build_lag_trace(source_span=(0, 3), stream_origin=10,
                            playback_at=12, hold_s=5, mt_s=1)
    assert trace["total_ms"] == -1000
    assert trace["hold_ms"] == 5000


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [None, "0", "true", "1"])
async def test_one_message_and_exact_opt_in_log(enabled, monkeypatch, tmp_path):
    # Load only the sender: importing main would initialize unrelated runtime assets.
    import ast
    import logging
    import os
    from pathlib import Path
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "backend" / "main.py"
    tree = ast.parse(source.read_text())
    sender = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
                  and node.name == "_send_fragment_lag_trace")
    namespace = dict(json=json, logging=logging, os=os, build_lag_trace=build_lag_trace)
    exec(compile(ast.Module(body=[sender], type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.chdir(tmp_path)
    if enabled is None:
        monkeypatch.delenv("BP_LOG_SESSIONS", raising=False)
    else:
        monkeypatch.setenv("BP_LOG_SESSIONS", enabled)
    sent = []

    async def send_text(value):
        sent.append(json.loads(value))

    trace = {"sent": False, "values": dict(source_span=(1, 2), stream_origin=10,
             emitted_at=13, playback_at=14, tts_started_at=13.5, text="Živá ukážka")}
    websocket = SimpleNamespace(send_text=send_text)
    await namespace["_send_fragment_lag_trace"](websocket, trace)
    await namespace["_send_fragment_lag_trace"](websocket, trace)
    assert len(sent) == 1
    path = tmp_path / "processed" / "sessions" / "lag_trace.jsonl"
    if enabled == "1":
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == sent[0]
    else:
        assert not path.parent.exists()


def test_trace_carries_speech_start_and_end_message_carries_playback_end():
    from backend.lag_trace import build_lag_trace_end
    trace = build_lag_trace(source_span=(1.0, 2.0), stream_origin=100.0, emitted_at=102.5, playback_at=103.0)
    assert trace["speech_start"] == 1000 and trace["speech_end"] == 2000
    end = build_lag_trace_end(emitted_at=102.5, stream_origin=100.0, playback_at=103.0, audio_s=1.5)
    assert end["type"] == "lag_trace_end" and end["emit_time"] == trace["emit_time"]
    assert end["playback_start"] == 3000 and end["playback_end"] == 4500


def test_end_message_without_audio_has_no_end():
    from backend.lag_trace import build_lag_trace_end
    end = build_lag_trace_end(emitted_at=1.0, stream_origin=0.0, playback_at=2.0, audio_s=0.0)
    assert end["playback_end"] is None
