"""Prosody mirror: reproduce the speaker's pauses and speaking rate in the translated audio.

Why: the /ws route synthesizes each phrase as soon as its text arrives and plays it back-to-back, so the
translation has none of the speaker's rhythm. Three pure, model-free pieces (unit tests: test/prosody_tests.py):

1. PlayoutScheduler  - constant-lag playout on the source clock: if you paused 1.2 s before a phrase, the translated
                       phrase starts 1.2 s after the previous one started (silence is padded, never negative; when
                       playback is behind, pauses are absorbed and the backlog shrinks).
2. trailing_pause_s  - short silence after a comma-level phrase, longer after a sentence end, inside one translation.
3. RateFollower      - smoothed ratio source_speech_duration / tts_duration -> Piper length_scale in [0.85, 1.15].

Default pause sizes come from the owner's reading of the bilingual script (53 pauses >=0.12 s, median 0.46 s;
the 18 sentence-end pauses were 0.8-2.4 s). Disable everything with BP_PROSODY=0.
"""
from __future__ import annotations

import os

ENABLED = os.environ.get("BP_PROSODY", "1") != "0"
MAX_LEAD_S = float(os.environ.get("BP_PROSODY_MAX_LEAD_S", "2.0"))
MIN_LEAD_S = 0.08
EXTRA_LAG_S = float(os.environ.get("BP_PROSODY_EXTRA_LAG_S", "0.6"))
RESET_GAP_S = 6.0          # a longer silence in the source starts a fresh anchor (new topic / speaker returned)
COMMA_PAUSE_S = float(os.environ.get("BP_PAUSE_COMMA_S", "0.25"))
SENTENCE_PAUSE_S = float(os.environ.get("BP_PAUSE_SENTENCE_S", "0.5"))
RATE_MIN, RATE_MAX = 0.85, 1.15


def trailing_pause_s(phrase: str) -> float:
    """Silence to append after a non-final phrase of one translation, from its closing punctuation."""
    p = phrase.rstrip()
    if p.endswith(("...", "…")):
        return SENTENCE_PAUSE_S
    if p.endswith((".", "!", "?")):
        return SENTENCE_PAUSE_S
    if p.endswith((",", ";", ":", "—", "-")):
        return COMMA_PAUSE_S
    return 0.0


class PlayoutScheduler:
    """Constant-lag playout anchored to the SOURCE clock: a phrase that began t seconds after the previous one in the
    source begins t seconds after it in the output, unless the queue is still busy (then it follows immediately, and the
    backlog shrinks whenever the voice is faster than the speaker). Lag = first chunk's natural latency + EXTRA_LAG_S;
    that slack is the price of reproducing pauses (set BP_PROSODY_EXTRA_LAG_S=0 for minimum delay, pauses mostly lost)."""

    def __init__(self, max_lead_s: float = MAX_LEAD_S, min_lead_s: float = MIN_LEAD_S, reset_gap_s: float = RESET_GAP_S,
                 extra_lag_s: float = EXTRA_LAG_S):
        self.max_lead_s, self.min_lead_s, self.reset_gap_s, self.extra_lag_s = max_lead_s, min_lead_s, reset_gap_s, extra_lag_s
        self.lag = None            # output start minus source onset, seconds
        self.last_src = None
        self.play_end = 0.0        # estimated end of everything queued so far
        self._start = 0.0

    def lead_silence(self, src_onset: float | None, now: float) -> float:
        ready = max(now, self.play_end)
        if src_onset is None:
            lead = 0.0
        else:
            if self.lag is None or self.last_src is None or src_onset - self.last_src > self.reset_gap_s or src_onset < self.last_src:
                self.lag = max(0.0, ready - src_onset) + self.extra_lag_s     # (re)anchor on the first chunk or after a long silence
            self.last_src = src_onset
            lead = min(self.max_lead_s, max(0.0, src_onset + self.lag - ready))
            if lead < self.min_lead_s:
                lead = 0.0
        self._start = ready + lead
        return lead

    def committed(self, audio_dur_s: float) -> None:
        """Call once the chunk (lead silence excluded) has been queued."""
        self.play_end = self._start + max(0.0, audio_dur_s)


class RateFollower:
    """Follows the speaker's pace: length_scale < 1 speeds the voice up, > 1 slows it down."""

    def __init__(self, alpha: float = 0.5):
        self.alpha, self.scale = alpha, 1.0

    def update(self, source_speech_s: float, tts_speech_s: float) -> float:
        if source_speech_s > 0.3 and tts_speech_s > 0.3:
            target = min(RATE_MAX, max(RATE_MIN, (source_speech_s / tts_speech_s) * self.scale))
            self.scale = (1 - self.alpha) * self.scale + self.alpha * target
        return self.scale
