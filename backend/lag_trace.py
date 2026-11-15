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

    speech_end = ms(source_span[1]) if source_span is not None and len(source_span) == 2 else None
    playback_start = stream_ms(playback_at)
    return {
        "type": "lag_trace",
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
