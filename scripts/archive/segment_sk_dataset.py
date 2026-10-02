#!/usr/bin/env python3
"""Segment the SK script reading into 18 sentence clips aligned to ground truth.

Method (purely acoustic, no STT): ffmpeg silencedetect -> inter-sentence
pauses (>=1.0s) become cut points at silence midpoints. Segment count is
snapped to the 18 script sentences by merging across the weakest pause (or
splitting the longest segment). Clips get the script text verbatim.

Out: OUT_DIR/wav/NN.wav + OUT_DIR/speaker_voices.json + printed QC table.
Run: venv/bin/python scripts/segment_sk_dataset.py
"""

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = "/tmp/piper_finetune_work/voices_sk_seg"
SCRIPT_MD = os.path.join(REPO_ROOT, "documentation", "reading_script_bilingual.md")
SK_WAV = "/tmp/piper_finetune_work/voices_sk/sk_script_reading.wav"
MIN_PAUSE = 1.0
N_SENT = 18


def script_sentences():
    sents = []
    with open(SCRIPT_MD) as f:
        for line in f:
            if line.startswith("|") and not line.startswith("| #") and "---" not in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) >= 3 and cells[0].isdigit():
                    sents.append(cells[2])
    return sents


def silences():
    p = subprocess.run(
        ["ffmpeg", "-i", SK_WAV, "-af",
         "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts: list = []
    for line in p.stderr.splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            starts.append(float(m.group(1)))
    bounds_raw = []
    si = 0
    for line in p.stderr.splitlines():
        m = re.search(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", line)
        if m:
            bounds_raw.append((starts[si], float(m.group(1)), float(m.group(2))))
            si += 1
    return bounds_raw


def main():
    import librosa
    import soundfile as sf

    sents = script_sentences()
    assert len(sents) == N_SENT, f"expected {N_SENT}, got {len(sents)}"
    wav, _ = librosa.load(SK_WAV, sr=22050, mono=True)
    dur = len(wav) / 22050

    pauses = [(s, e, d) for s, e, d in silences()
              if d >= MIN_PAUSE and s > 0.5 and e < dur - 0.5]
    print(f"pauses >= {MIN_PAUSE}s (excl. edges): {len(pauses)}")
    cuts = sorted((s + e) / 2 for s, e, d in pauses)  # midpoint of each pause

    # snap segment count to N_SENT
    while len(cuts) + 1 > N_SENT:
        # merge across the weakest (shortest) pause
        idx = min(range(len(cuts)),
                  key=lambda i: next(d for s, e, d in pauses if abs((s + e) / 2 - cuts[i]) < 0.01))
        cuts.pop(idx)
    bounds = [0.0] + cuts + [dur]
    while len(bounds) - 1 < N_SENT:
        # split the longest segment in half
        i = max(range(len(bounds) - 1), key=lambda i: bounds[i + 1] - bounds[i])
        bounds.insert(i + 1, (bounds[i] + bounds[i + 1]) / 2)
    assert len(bounds) - 1 == N_SENT

    os.makedirs(f"{OUT_DIR}/wav", exist_ok=True)
    table = []
    for i, sent in enumerate(sents):
        t0, t1 = bounds[i], bounds[i + 1]
        assert t1 > t0, f"clip {i + 1}: empty range"
        name = f"{i + 1:02d}.wav"
        sf.write(f"{OUT_DIR}/wav/{name}", wav[int(t0 * 22050):int(t1 * 22050)], 22050)
        table.append({"n": i + 1, "t0": round(t0, 2), "t1": round(t1, 2),
                      "dur": round(t1 - t0, 2), "text": sent, "file": name})
    with open(f"{OUT_DIR}/speaker_voices.json", "w") as f:
        json.dump(
            [{"id": f"sk_seg_{t['n']:02d}", "language": "sk", "name": f"sk_seg_{t['n']:02d}",
              "path": t["file"], "transcript_source": "recording script row",
              "transcribed_text": t["text"]} for t in table],
            f, indent=2, ensure_ascii=False,
        )
    print(f"{'n':>2} {'t0':>7} {'t1':>7} {'dur':>5}  text head")
    for t in table:
        print(f"{t['n']:>2} {t['t0']:>7} {t['t1']:>7} {t['dur']:>5}  {t['text'][:52]}")
    durs = [t["dur"] for t in table]
    print(f"clips {len(table)}, min {min(durs)}s max {max(durs)}s total {sum(durs):.1f}s / {dur:.1f}s")


if __name__ == "__main__":
    main()
