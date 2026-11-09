"""Model-free tests for the Voxtral token -> word assembly (backend/stt/voxtral_stt.py)."""
from backend.stt.voxtral_stt import _Stream


def test_words_hold_last_partial_and_join_punctuation():
    s = _Stream(None)
    assert s._words([" Good", " morning"]) == ["Good"]          # "morning" may still grow
    assert s._words([" and", " wel"]) == ["morning", "and"]
    assert s._words(["come", " back", "."]) == ["welcome"]       # subword pieces and the full stop join their word
    assert s._words([], final=True) == ["back."]                 # close flushes the held tail


def test_empty_and_whitespace_only():
    s = _Stream(None)
    assert s._words([]) == []
    assert s._words([" "]) == []
