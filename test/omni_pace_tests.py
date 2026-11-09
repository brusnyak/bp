"""Model-free: OmniVoice follows the route's rate follower (Piper length_scale -> OmniVoice speed) without loading any model."""
import numpy as np
from backend.tts.omni_tts import OmniVoiceTTS


def _stub():
    t = object.__new__(OmniVoiceTTS)
    t.speed, t.seen = 1.0, []
    def fake(text, language="sk", speaker_wav_path=None, output_path=None, speed=None):
        t.seen.append(speed)
        return np.ones(10, dtype=np.float32), 24000, 0.0
    t.synthesize = fake
    return t


def test_length_scale_becomes_speed_clamped():
    t = _stub()
    list(t.synthesize_stream("Hello there. Second clause.", length_scale=1.25))
    assert t.seen and all(abs(s - 0.8) < 1e-9 for s in t.seen)           # 1/1.25
    t = _stub(); list(t.synthesize_stream("Hello.", length_scale=0.5))
    assert t.seen == [1.3]                                                # clamped: faster than 1.3x degrades audio
    t = _stub(); list(t.synthesize_stream("Hello."))
    assert t.seen == [None]                                               # no follower -> engine default untouched
