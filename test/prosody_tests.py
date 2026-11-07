"""Model-free tests for backend/prosody.py: pause mirroring, punctuation pauses, rate following."""
import numpy as np
from backend import prosody as P


def sched(extra=0.0, **kw):
    return P.PlayoutScheduler(extra_lag_s=extra, **kw)


def test_pause_is_mirrored_when_audio_is_shorter_than_source():
    s = sched()
    assert s.lead_silence(10.0, 10.5) == 0.0          # first chunk anchors lag = 0.5 s
    s.committed(1.0)                                  # plays 10.5..11.5
    # next phrase began 3.0 s after the first in the source -> output start = 13.0 + 0.5 = 13.5; ready 11.6 -> pad 1.9 s
    assert abs(s.lead_silence(13.0, 11.6) - 1.9) < 1e-6


def test_backlog_is_absorbed_then_recovers():
    s = sched()
    s.lead_silence(0.0, 0.5); s.committed(4.0)        # plays 0.5..4.5, lag 0.5
    assert s.lead_silence(2.0, 2.2) == 0.0            # wanted 2.5 < ready 4.5: no padding, follows immediately
    s.committed(0.5)                                  # plays 4.5..5.0
    # the speaker paused: next onset 9.0 -> wanted 9.5, ready 5.1 -> pad 4.4 s, capped
    assert s.lead_silence(9.0, 5.1) == P.MAX_LEAD_S


def test_extra_lag_buys_room_for_pauses():
    a, b = sched(0.0), sched(0.6)
    la0, lb0 = a.lead_silence(0.0, 1.0), b.lead_silence(0.0, 1.0)
    assert la0 == 0.0 and abs(lb0 - 0.6) < 1e-6        # the slack is paid once, as extra delay on the first chunk
    a.committed(1.0); b.committed(1.0)                 # a plays 1.0..2.0, b plays 1.6..2.6
    la, lb = a.lead_silence(2.2, 2.4), b.lead_silence(2.2, 2.4)
    assert abs(la - 0.8) < 1e-6 and abs(lb - 1.2) < 1e-6   # wanted starts 3.2 / 3.8 against ready 2.4 / 2.6


def test_lead_is_capped():
    s = sched(0.0, max_lead_s=0.3)
    s.lead_silence(0.0, 1.0); s.committed(0.1)         # lag 1.0, plays 1.0..1.1
    assert s.lead_silence(2.0, 2.1) == 0.3             # wanted 3.0, ready 2.1 -> 0.9 capped at 0.3


def test_long_silence_reanchors_without_dead_air():
    s = sched()
    s.lead_silence(0.0, 0.5); s.committed(0.5)
    assert s.lead_silence(30.0, 31.0) == 0.0          # > RESET_GAP_S: lag re-measured, no padding


def test_missing_onset_never_pads():
    s = sched(); s.lead_silence(1.0, 1.5); s.committed(1.0)
    assert s.lead_silence(None, 2.5) == 0.0


def test_trailing_pause_by_punctuation():
    assert P.trailing_pause_s("Hello,") == P.COMMA_PAUSE_S
    assert P.trailing_pause_s("Hello there.") == P.SENTENCE_PAUSE_S
    assert P.trailing_pause_s("Why?") == P.SENTENCE_PAUSE_S
    assert P.trailing_pause_s("no punctuation") == 0.0


def test_rate_follower_is_clamped_and_smoothed():
    r = P.RateFollower()
    assert r.update(2.0, 4.0) < 1.0                    # tts slower than speaker -> speed up (length_scale < 1)
    for _ in range(20): r.update(1.0, 10.0)
    assert abs(r.scale - P.RATE_MIN) < 1e-3            # converges to the clamp
    for _ in range(40): r.update(10.0, 1.0)
    assert abs(r.scale - P.RATE_MAX) < 1e-3
    before = r.scale; r.update(0.1, 0.1)
    assert r.scale == before                           # too short to trust: unchanged


def test_piper_pause_padding_inserts_silence():
    from backend.tts.piper_tts import PiperTTS
    t = PiperTTS.__new__(PiperTTS)
    calls = []
    t.synthesize = lambda text, **kw: (calls.append(kw.get("length_scale")) or np.ones(1000, dtype=np.float32), 1000, 0.0)
    t.speaker_id = None
    out = list(PiperTTS.synthesize_stream(t, "First, second. Third", language="en", length_scale=0.9, pauses=True))
    assert [len(x) for x in out] == [1000 + int(1000 * P.COMMA_PAUSE_S), 1000 + int(1000 * P.SENTENCE_PAUSE_S), 1000]
    assert calls == [0.9, 0.9, 0.9]
