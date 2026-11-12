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


def test_nemotron_prompt_key_follows_the_models_own_spelling():
    from backend.stt.nemotron_stt import prompt_key
    d = {"en-US": 0, "sk-SK": 1, "de-DE": 2, "cs": 3}
    assert prompt_key(d, "sk") == "sk-SK" and prompt_key(d, "CS") == "cs" and prompt_key(d, "en") == "en-US"
    assert prompt_key(d, "ja") is None and prompt_key(d, None) is None and prompt_key(None, "sk") is None   # unknown: the model default, never a wrong prompt


def test_nemotron_covers_czech_not_slovak_and_voxtral_covers_neither(monkeypatch):
    """Slovak is routed away from Nemotron on purpose (41% word error on the saved sk->en route run, 2026-11-09 evidence)."""
    import backend.main as m
    monkeypatch.setattr(m.sys, "platform", "darwin")
    assert m._effective_stt_engine({"stt_engine": "nemotron", "source_lang": "cs"}) == "nemotron"
    assert m._effective_stt_engine({"stt_engine": "nemotron", "source_lang": "sk"}) == "mlx_whisper"
    assert m._effective_stt_engine({"stt_engine": "voxtral", "source_lang": "sk"}) == "mlx_whisper"
