import re


def split_into_phrases(text: str) -> list:
    """T045: split text into clause-level phrases on punctuation, for chunked TTS streaming —
    each phrase gets synthesized (and sent) as its own audio chunk instead of waiting for the
    whole sentence. Deliberately simple (punctuation-based), not an NLP phrase-boundary model —
    matches this project's scope for this task."""
    parts = re.split(r'(?<=[.!?;:,])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_audio_by_duration(wav, sample_rate: int, chunk_ms: int = 150):
    """T045: slice one synthesized phrase's audio into ~chunk_ms pieces before it's sent over
    the wire. This doesn't shorten the underlying (blocking) synthesis call for that phrase —
    cancellation still can't preempt a call already in flight — but it does bound how much
    already-synthesized audio is queued waiting to be sent, which is what the barge-in flush
    target (<60ms, per documentation/simultaneous_translation_research_2026-08-22.md) actually
    needs: cancelling while awaiting the next queue chunk is fast (pure asyncio primitive),
    cancelling mid-way through sending one giant chunk is not."""
    n = len(wav)
    step = max(1, int(sample_rate * chunk_ms / 1000))
    for i in range(0, n, step):
        piece = wav[i:i + step]
        if len(piece) > 0:
            yield piece
