"""Model-free tests for the MLX Whisper output guards (backend/stt/mlx_whisper_stt.py)."""
from backend.stt.mlx_whisper_stt import is_garbage


def test_repetition_loop_is_garbage():
    assert is_garbage("that we can't just 1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-", 2.0)
    assert is_garbage("ha ha ha ha ha ha ha", 2.0)


def test_caption_credits_are_garbage():
    assert is_garbage("Transcription by CastingWords", 2.0)
    assert is_garbage("Thanks for watching", 3.0)


def test_impossible_length_and_compression_are_garbage():
    assert is_garbage("word " * 80, 1.0)
    assert is_garbage("fine text", 2.0, compression_ratio=3.1)


def test_real_speech_passes():
    assert not is_garbage("Good morning and welcome to this live demonstration of real time speech translation.", 6.0)
    assert not is_garbage("It is one of the first of the fifteen.", 2.5)
    assert not is_garbage("", 1.0)


def test_should_drop_stock_phrase_on_short_chunk_only():
    from backend.stt.guards import should_drop
    assert should_drop(" Thank you.", 2.0) and should_drop("Okay.", 1.0)
    assert not should_drop("Thank you for listening, and I look forward to your questions.", 2.0)   # a real sentence
    assert not should_drop(" Thank you.", 4.0)                                                       # long chunk: probably real
    assert not should_drop("", 1.0) and not should_drop("   ", 1.0)
