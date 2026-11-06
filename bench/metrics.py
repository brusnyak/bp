"""WER / CER with a frozen normalisation. Run `python bench/metrics.py` for the self-test."""
import re


def normalise(s: str) -> str:
    """lowercase, strip punctuation (anything not word char/space), collapse whitespace. NO diacritic folding."""
    return " ".join(re.sub(r"[^\w\s]", "", s.lower()).split())


def _edit(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p = d[:]
        d[0] = i
        for j in range(1, len(h) + 1):
            d[j] = min(p[j] + 1, d[j - 1] + 1, p[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)]


def wer(ref: str, hyp: str) -> float:
    r, h = normalise(ref).split(), normalise(hyp).split()
    return _edit(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r, h = list(normalise(ref)), list(normalise(hyp))  # spaces count as characters
    return _edit(r, h) / len(r)


if __name__ == "__main__":
    assert normalise("  Dobrý, DEŇ!  Ako   sa máš? ") == "dobrý deň ako sa máš"
    assert normalise("deň") != normalise("den")  # no diacritic folding
    assert wer("a b c d", "a b c d") == 0.0
    assert wer("a b c d", "a x c d") == 0.25
    assert wer("a b c d", "a c d") == 0.25          # deletion
    assert wer("a b", "a b c d") == 1.0             # insertions can exceed/hit 1
    assert wer("Ahoj, svet!", "ahoj svet") == 0.0   # punctuation/case ignored
    assert wer("deň", "den") == 1.0                 # diacritics matter
    assert cer("abcd", "abxd") == 0.25
    assert cer("ab cd", "ab cd") == 0.0
    print("metrics self-test PASS")
