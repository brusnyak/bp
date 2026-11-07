"""Model-free test for the token -> word merge used by the Parakeet streaming engine."""
from backend.stt.onnx_parakeet_stt import OnnxParakeetSTT


def test_tokens_merge_into_words_with_times():
    tokens = [' I', "'", 'm', ' spe', 'ak', 'ing', ' En', 'gl', 'ish']
    stamps = [0.24, 0.4, 0.56, 0.72, 0.88, 1.04, 1.2, 1.36, 1.52]
    w = OnnxParakeetSTT._words(tokens, stamps, 2.0)
    assert [x.word for x in w] == ["I'm", "speaking", "English"]
    assert w[0].start == 0.24 and w[1].start == 0.72 and w[2].start == 1.2
    assert all(a.end <= b.start + 1e-9 for a, b in zip(w, w[1:]))   # words never overlap
    assert w[-1].end <= 2.0                                          # clipped to the audio length


def test_empty_tokens():
    assert OnnxParakeetSTT._words([], [], 1.0) == []
