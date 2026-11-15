"""Incremental (simultaneous) transcription logic, model-free so it can be unit-tested.

Local agreement (n=2): a word is committed once two consecutive hypotheses over the same growing
audio start with the same words. Committed words are grouped into short fragments (clause-sized)
that can be translated and voiced while the speaker is still talking. A fragment is flushed when it
ends a clause, reaches max_words, or has waited max_wait_s. Nothing here touches audio or models.
"""
import os
import re

SENTENCE_END = (".", "!", "?", ";", ":")
CLAUSE_END = SENTENCE_END + (",",)
FUNCTION_WORDS = {
    "en": set("a an the to of in on at for from with by and or but if that as".split()),
    "de": set("der die das den dem des ein eine einen einem einer zu von in an auf mit für und oder aber dass weil".split()),
    "cs": set("v ve na do z ze k ke s se pro od u o a i ale nebo že aby".split()),
    "sk": set("v vo na do z zo k ku s so pre od u o a i ale alebo že aby".split()),
}


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


class Agreement:
    """words are (text, start_s, end_s). update() returns the newly committed prefix."""

    def __init__(self):
        self.prev = []

    def update(self, words):
        n = 0
        while n < len(self.prev) and n < len(words) and _norm(self.prev[n][0]) == _norm(words[n][0]):
            n += 1
        # the caller trims audio up to the last committed word, so the next hypothesis starts at words[n:]
        self.prev = list(words[n:])
        return list(words[:n])


def stream_end(frame_times, last_push_t):
    """Stream-clock time at which a segment's words end. Never fall back to a wall clock: a flush after every frame was already trimmed once stamped
    time.perf_counter() (about 117000 s of machine uptime) into a span that is read as stream time, which corrupted lag and drift in 4 of 9 saved runs."""
    if frame_times:
        return frame_times[-1]
    return last_push_t if last_push_t is not None else 0.0


class Fragmenter:
    def __init__(self, min_words=4, max_words=9, max_wait_s=1.6, language="en", phrase_guard=None):
        self.min_words, self.max_words, self.max_wait_s = min_words, max_words, max_wait_s
        self.phrase_guard = os.environ.get("BP_INCR_PHRASE_GUARD", "1") != "0" if phrase_guard is None else phrase_guard
        self.function_words = FUNCTION_WORDS.get(language, set())
        self._words, self._arrivals = [], []
        self.pending, self.since = [], None
        self.t0 = self.t1 = None      # stream-clock span of the pending words
        self.span = None              # span of the last popped fragment
        self.hold_s = None            # how long the oldest word of the last popped fragment waited in the buffer (the fragment policy's share of the lag)

    def push(self, words, now):
        for w in words:
            if w[0].strip():
                self.pending.append(w[0].strip())
                self._words.append(w)
                self._arrivals.append(now)
                self.t0 = w[1] if self.t0 is None else min(self.t0, w[1])
                self.t1 = w[2] if self.t1 is None else max(self.t1, w[2])
        if self.pending and self.since is None:
            self.since = now

    def pop(self, now, final=False, next_word=None):
        n = len(self.pending)
        if not n:
            return None
        expired = self.since is not None and now - self.since >= self.max_wait_s
        cut = n
        if not final and not expired:
            # Prefer the first complete clause, preserving the remainder and its clocks.
            boundaries = [i + 1 for i, w in enumerate(self.pending)
                          if (i + 1 >= 2 and w.endswith(SENTENCE_END))
                          or (i + 1 >= self.min_words and w.endswith(CLAUSE_END))]
            if self.phrase_guard and boundaries:
                cut = boundaries[0]
            elif not (boundaries and boundaries[-1] == n) and n < self.max_words:
                return None
            if self.phrase_guard and not self.pending[cut - 1].endswith(CLAUSE_END):
                following = self.pending[cut] if cut < n else (next_word or "")
                if (_norm(self.pending[cut - 1]) in self.function_words
                        or _norm(following) in self.function_words):
                    return None
        text = " ".join(self.pending[:cut])
        self.hold_s = (now - self.since) if self.since is not None else 0.0
        emitted = self._words[:cut]
        self.span = (min(w[1] for w in emitted), max(w[2] for w in emitted))
        del self.pending[:cut], self._words[:cut], self._arrivals[:cut]
        self.since = self._arrivals[0] if self._arrivals else None
        self.t0 = min((w[1] for w in self._words), default=None)
        self.t1 = max((w[2] for w in self._words), default=None)
        return text
