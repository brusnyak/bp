from backend.stt.incremental import Agreement, Fragmenter


def W(*t):
    return [(x, i * 0.3, i * 0.3 + 0.25) for i, x in enumerate(t)]


def test_agreement_commits_only_stable_prefix():
    a = Agreement()
    assert a.update(W("when", "the", "sun")) == []
    got = a.update(W("when", "the", "sunlight", "strikes"))
    assert [w[0] for w in got] == ["when", "the"]          # "sun" vs "sunlight" is not stable yet
    got = a.update(W("sunlight", "strikes", "rain"))          # caller trimmed: hypothesis restarts at "sunlight"
    assert [w[0] for w in got] == ["sunlight", "strikes"]


def test_agreement_ignores_case_and_punctuation():
    a = Agreement()
    a.update(W("Hello,", "world"))
    assert [w[0] for w in a.update(W("hello", "World."))] == ["hello", "World."]


def test_fragmenter_flushes_on_clause_length_wait_and_final():
    f = Fragmenter(min_words=4, max_words=9, max_wait_s=1.6)
    f.push(W("when", "the", "sun"), 0.0)
    assert f.pop(0.1) is None                                  # too short, no punctuation
    f.push(W("strikes,"), 0.4)
    assert f.pop(0.5) == "when the sun strikes,"               # clause end with >= min_words
    f.push(W("they", "act"), 1.0)
    assert f.pop(1.5) is None
    assert f.pop(2.7) == "they act"                            # waited max_wait_s
    f.push(W("a"), 3.0)
    assert f.pop(3.1) is None and f.pop(3.1, final=True) == "a"
    assert f.pop(9.9, final=True) is None                      # nothing pending
    f.push(W(*"one two three four five six seven eight nine".split()), 10.0)
    assert f.pop(10.0).count(" ") == 8                         # max_words forces a flush


def test_fragmenter_records_hold_of_oldest_word():
    from backend.stt.incremental import Fragmenter
    f = Fragmenter(min_words=2, max_words=9, max_wait_s=0.8)
    f.push([("Good", 0.0, 0.3)], now=10.0)
    f.push([("morning,", 0.3, 0.8)], now=10.5)
    assert f.pop(10.9) == "Good morning,"         # clause end with >= 2 words
    assert abs(f.hold_s - 0.9) < 1e-9             # waited since the first word was pushed


def test_stream_end_never_uses_a_wall_clock():
    from backend.stt.incremental import stream_end
    assert stream_end([1.0, 2.5], 9.0) == 2.5       # frames left: their newest time
    assert stream_end([], 7.25) == 7.25             # all trimmed: the last pushed stream time, not time.perf_counter()
    assert stream_end([], None) == 0.0


import pytest


@pytest.mark.parametrize("language,word", [("en", "the"), ("de", "mit"), ("cs", "na"), ("sk", "pre")])
def test_function_word_tail_waits_for_complement_but_not_past_cap(language, word):
    f = Fragmenter(max_words=3, max_wait_s=0.8, language=language)
    f.push(W("we", "work", word), 0)
    assert f.pop(0.1) is None
    f.push(W("people"), 0.2)
    assert f.pop(0.2) == f"we work {word} people"
    f.push(W("we", "work", word), 1)
    assert f.pop(1.8) == f"we work {word}"


@pytest.mark.parametrize("language,word", [("en", "of"), ("de", "und"), ("cs", "že"), ("sk", "alebo")])
def test_function_word_lookahead_delays_unpunctuated_cut(language, word):
    f = Fragmenter(max_words=2, language=language)
    f.push(W("more", "words"), 0)
    assert f.pop(0.1, next_word=word) is None
    assert f.pop(0.2, final=True) == "more words"


def test_clause_preferred_without_losing_remainder_or_timestamps():
    f = Fragmenter(min_words=2, max_words=4, max_wait_s=0.8)
    f.push([("Hello", 1, 2), ("world,", 2, 3)], 10)
    f.push([("from", 4, 5), ("the", 5, 6)], 10.2)
    assert f.pop(10.3, next_word="and") == "Hello world,"
    assert f.span == (1, 3)
    assert f.pending == ["from", "the"]
    assert f.pop(10.4) is None
    assert f.pop(11.01) == "from the"
    assert f.span == (4, 6)
    assert f.hold_s == pytest.approx(0.81)


def test_single_word_cannot_wait_forever():
    f = Fragmenter(max_wait_s=0.8)
    f.push(W("the"), 0)
    assert f.pop(0.8) == "the"


def test_phrase_guard_disabled_restores_length_flush(monkeypatch):
    monkeypatch.setenv("BP_INCR_PHRASE_GUARD", "0")
    f = Fragmenter(max_words=2)
    f.push(W("to", "the"), 0)
    assert f.pop(0.1, next_word="and") == "to the"
    f = Fragmenter(max_words=2, phrase_guard=True)
    f.push(W("to", "the"), 0)
    assert f.pop(0.1) is None
