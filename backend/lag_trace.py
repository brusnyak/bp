"""Pure fragment timing conversion. Absolute times use perf_counter seconds.

playback_start means first successful server audio send, not browser playout.
Incremental recognition has no per-fragment STT duration; leave it unknown.
"""
import math


def build_lag_trace(*, source_span=None, stream_origin=None, emitted_at=None,
                    hold_s=None, stt_s=None, mt_s=None, tts_started_at=None,
                    playback_at=None, text=""):
    def ms(value):
        return value * 1000 if value is not None and math.isfinite(value) else None

    def stream_ms(value):
        return ms(value - stream_origin) if value is not None and stream_origin is not None else None

    speech_start = ms(source_span[0]) if source_span is not None and len(source_span) == 2 else None
    speech_end = ms(source_span[1]) if source_span is not None and len(source_span) == 2 else None
    playback_start = stream_ms(playback_at)
    return {
        "type": "lag_trace",
        "speech_start": speech_start,
        "speech_end": speech_end,
        "emit_time": stream_ms(emitted_at),
        "hold_ms": ms(hold_s),
        "stt_ms": ms(stt_s),
        "mt_ms": ms(mt_s),
        "tts_first_ms": ms(playback_at - tts_started_at)
        if playback_at is not None and tts_started_at is not None else None,
        "playback_start": playback_start,
        "total_ms": playback_start - speech_end
        if playback_start is not None and speech_end is not None else None,
        "words": len(text.split()),
        "text": text,
    }


def build_lag_trace_end(*, emitted_at=None, stream_origin=None, playback_at=None, audio_s=0.0):
    """Sent once the fragment's audio has all been handed to the socket.

    playback_end = first send + length of the audio sent (server clock, contiguous playout assumed;
    browser queueing and device latency are not included). emit_time matches the earlier lag_trace.
    """
    def stream_ms(value):
        return (value - stream_origin) * 1000 if value is not None and stream_origin is not None and math.isfinite(value) else None

    start = stream_ms(playback_at)
    return {
        "type": "lag_trace_end",
        "emit_time": stream_ms(emitted_at),
        "playback_start": start,
        "playback_end": start + audio_s * 1000 if start is not None and audio_s else None,
    }
