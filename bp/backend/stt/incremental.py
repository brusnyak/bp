"""Incremental (simultaneous) transcription logic, model-free so it can be unit-tested.

Local agreement (n=2): a word is committed once two consecutive hypotheses over the same growing
audio start with the same words. Committed words are grouped into short fragments (clause-sized)
that can be translated and voiced while the speaker is still talking. A fragment is flushed when it
ends a clause, reaches max_words, or has waited max_wait_s. Nothing here touches audio or models.
"""
import re

SENTENCE_END = (".", "!", "?", ";", ":")
CLAUSE_END = SENTENCE_END + (",",)


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


class Fragmenter:
    def __init__(self, min_words=4, max_words=9, max_wait_s=1.6):
        self.min_words, self.max_words, self.max_wait_s = min_words, max_words, max_wait_s
        self.pending, self.since = [], None
        self.t0 = self.t1 = None      # stream-clock span of the pending words
        self.span = None              # span of the last popped fragment

    def push(self, words, now):
        for w in words:
            if w[0].strip():
                self.pending.append(w[0].strip())
                self.t0 = w[1] if self.t0 is None else min(self.t0, w[1])
                self.t1 = w[2] if self.t1 is None else max(self.t1, w[2])
        if self.pending and self.since is None:
            self.since = now

    def pop(self, now, final=False):
        n = len(self.pending)
        if not n:
            return None
        last = self.pending[-1]
        ready = (final
                 or (n >= 2 and last.endswith(SENTENCE_END))
                 or (n >= self.min_words and last.endswith(CLAUSE_END))
                 or n >= self.max_words
                 or (n >= 2 and self.since is not None and now - self.since >= self.max_wait_s))
        if not ready:
            return None
        text = " ".join(self.pending)
        self.span = (self.t0, self.t1)
        self.pending, self.since, self.t0, self.t1 = [], None, None, None
        return text
