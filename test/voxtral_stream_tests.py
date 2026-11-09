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


def test_engine_choice_falls_back_for_unsupported_source(monkeypatch):
    import backend.main as m
    monkeypatch.setattr(m.sys, "platform", "linux")
    assert m._effective_stt_engine({"stt_engine": "voxtral", "source_lang": "en"}) == "voxtral"
    assert m._effective_stt_engine({"stt_engine": "voxtral", "source_lang": "sk"}) == "faster_whisper"   # no Slovak in Voxtral
    monkeypatch.setattr(m.sys, "platform", "darwin")
    assert m._effective_stt_engine({"stt_engine": "voxtral", "source_lang": "sk"}) == "mlx_whisper"
