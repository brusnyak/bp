"""Output guards shared by every speech-recognition engine (applied in backend/main.py after recognition, and inside the MLX engine).

Live findings 2026-11-07 (Kaggle T4, real route, faster-whisper turbo): 10 of 34 recognised segments were stock phrases ("Thank you.") that Whisper
invents on silence or noise between sentences. Each was translated, spoken and padded, so the playback queue grew to an 8 s delay.
"""
from __future__ import annotations

import os
import re

GUARD = os.environ.get("BP_STT_HALLUCINATION_GUARD", "1") != "0"
# Whisper emits these on breaths, clicks and room noise (no VAD filter in MLX): seen 7x in a 60 s live replay of
# quiet speech on 2026-11-07. Dropped only when the audio chunk is short; real sentences are longer.
HALLUCINATIONS = {"thank you", "thanks", "thanks for watching", "thank you for watching", "okay", "ok", "oh", "you", "bye", "hmm", "uh", "um", "mm", "ah"}
SHORT_CHUNK_S = float(os.environ.get("BP_STT_STOCK_CHUNK_S", "2.5"))  # a stock phrase on a chunk shorter than this is dropped


import re

MIN_CHUNK_S = 0.4          # nothing intelligible fits in less (a 0.02 s tail produced "Oh." and a 35 s TTS cascade on 2026-11-07)
MAX_CHARS_PER_S = 30       # fast speech is ~15-18 chars/s; far above this is a decoding loop
LOOP_RE = re.compile(r"(\S{1,12})(?:[-\s,.]+\1){5,}")   # "1-1-1-1-1-1", "ha ha ha ha ha ha"
CREDIT_RE = re.compile(r"transcription by|subtitles? by|captions? by|castingwords|amara\.org|thanks for watching|please subscribe", re.I)


def is_garbage(text: str, chunk_s: float, compression_ratio: float = 0.0) -> bool:
    """True for output that cannot be real speech for this chunk: repetition loops, caption credits, impossible length."""
    t = text.strip()
    if not t:
        return False
    return bool(LOOP_RE.search(t) or CREDIT_RE.search(t) or compression_ratio > 2.4 or len(t) > MAX_CHARS_PER_S * max(chunk_s, 0.5) + 20)


def _norm(t: str) -> str:
    return "".join(c for c in t.lower() if c.isalnum() or c == " ").strip()




def should_drop(text: str, chunk_s: float, compression_ratio: float = 0.0) -> bool:
    """True when recognised text should not reach translation: loops, caption credits, impossible length, or a stock phrase on a short chunk."""
    if not GUARD or not text or not text.strip():
        return False
    return is_garbage(text, chunk_s, compression_ratio) or (chunk_s < SHORT_CHUNK_S and _norm(text) in HALLUCINATIONS)
