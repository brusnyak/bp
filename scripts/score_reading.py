#!/usr/bin/env python3
"""Score a recording of documentation/reading_script_multilingual.md: word error per section and pause statistics.

    python scripts/score_reading.py ~/Downloads/de_script_1.m4a de            # engine from BP_SCORE_ENGINE (mlx | faster), default mlx on macOS else faster
    BP_SCORE_MODEL=large-v3-turbo python scripts/score_reading.py rec.m4a sk

The reference is the script itself (section C with its pause marks removed). Sections are found by timing: the recording is transcribed once and the
words are aligned to the concatenated reference by edit distance, so a skipped sentence costs deletions, not a shifted score.
"""
import os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANG_HEAD = {"en": "## English (EN)", "sk": "## Slovak (SK)", "de": "## German (DE)", "cs": "## Czech (CS)"}


def reference(lang):
    text = (ROOT / "documentation" / "reading_script_multilingual.md").read_text(encoding="utf-8")
    start = text.index(LANG_HEAD[lang]); nxt = [text.index(h, start + 5) for h in LANG_HEAD.values() if h in text and text.index(h) > start]
    block = text[start: min(nxt) if nxt else len(text)]
    out = {}
    for line in block.splitlines():
        m = re.match(r"^([A-D])(\d+)\.\s+(.*)$", line)
        if m: out.setdefault(m.group(1), []).append(re.sub(r"\s*/+\s*", " ", m.group(3)))
    return out


norm = lambda s: re.findall(r"[^\W_]+", s.lower())


def align(ref_words, hyp_words):
    """Edit distance with backtrace: returns per-reference-word 'correct' flags."""
    n, m = len(ref_words), len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (ref_words[i-1] != hyp_words[j-1]))
    ok, i, j = [False] * n, n, m
    while i > 0 and j > 0:
        if d[i][j] == d[i-1][j-1] + (ref_words[i-1] != hyp_words[j-1]):
            ok[i-1] = ref_words[i-1] == hyp_words[j-1]; i -= 1; j -= 1
        elif d[i][j] == d[i-1][j] + 1: i -= 1
        else: j -= 1
    return ok


def transcribe(path, lang):
    engine = os.environ.get("BP_SCORE_ENGINE") or ("mlx" if sys.platform == "darwin" else "faster")
    model = os.environ.get("BP_SCORE_MODEL", "large-v3-turbo")
    if engine == "mlx":
        import mlx_whisper
        return mlx_whisper.transcribe(path, path_or_hf_repo=f"mlx-community/whisper-{model}", language=lang, condition_on_previous_text=False)["text"]
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="auto", compute_type="auto")
    segs, _ = m.transcribe(path, language=lang, beam_size=5, vad_filter=True, condition_on_previous_text=False)
    return " ".join(s.text for s in segs)


def pauses(path):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "silencedetect=noise=-45dB:d=0.12", "-f", "null", "-"], capture_output=True, text=True)
    return sorted(float(x) for x in re.findall(r"silence_duration: ([\d.]+)", p.stderr))


def peak_db(path):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"max_volume: (-?[\d.]+) dB", p.stderr); return float(m.group(1)) if m else None


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in LANG_HEAD: sys.exit(__doc__)
    path, lang = sys.argv[1], sys.argv[2]
    ref = reference(lang); hyp = norm(transcribe(path, lang))
    order = [(sec, norm(s)) for sec in "ABCD" for s in ref.get(sec, [])]
    ref_words = [w for _, ws in order for w in ws]; sec_of = [sec for sec, ws in order for _ in ws]
    ok = align(ref_words, hyp)
    print(f"{path} [{lang}] reference {len(ref_words)} words, recognised {len(hyp)} words, peak level {peak_db(path)} dB (aim for -6 to -3)")
    for sec, name in (("A", "plain sentences"), ("B", "numbers, names, terms"), ("C", "pause passage"), ("D", "short questions")):
        idx = [i for i, s in enumerate(sec_of) if s == sec]
        if idx: print(f"  {sec} {name:24s} word error {100 * (1 - sum(ok[i] for i in idx) / len(idx)):5.1f}%  ({len(idx)} words)")
    print(f"  all sections            word error {100 * (1 - sum(ok) / len(ok)):5.1f}%")
    ps = pauses(path)
    if ps:
        q = lambda f: ps[min(len(ps) - 1, int(f * len(ps)))]
        long_marked = sum(sentence.count("//") for sentence in ref.get("C", []))
        print(f"  pauses >= 0.12 s: {len(ps)}; median {q(.5):.2f} s, p90 {q(.9):.2f} s, longest {ps[-1]:.2f} s; long (>= 0.8 s): {sum(p >= 0.8 for p in ps)}")


if __name__ == "__main__":
    main()
