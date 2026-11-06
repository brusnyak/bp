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
