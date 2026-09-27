#!/usr/bin/env python3
"""Generic sentence segmenter: audio + transcript text -> clips + speaker_voices.json.

Silence-midpoint cuts snapped to len(sentences) (cf. segment_sk_dataset.py).
Usage: venv/bin/python scripts/segment_generic.py <audio> <out_dir> <id_prefix> <lang> --text "..."
Writes out_dir/NN.wav + out_dir/speaker_voices.json, prints QC table.
"""

import argparse
import json
import os
import re
import subprocess
import sys


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", text.strip()) if s.strip()]


def bounds_for(audio, n):
    import librosa

    p = subprocess.run(
        ["ffmpeg", "-i", audio, "-af", "silencedetect=noise=-35dB:d=0.4",
         "-f", "null", "-"], capture_output=True, text=True,
    )
    starts, raw = [], []
    for line in p.stderr.splitlines():
        m = re.search(r"silence_start: ([\d.]+)", line)
        if m:
            starts.append(float(m.group(1)))
    si = 0
    for line in p.stderr.splitlines():
        m = re.search(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", line)
        if m:
            raw.append((starts[si], float(m.group(1)), float(m.group(2))))
            si += 1
    wav, _ = librosa.load(audio, sr=22050, mono=True)
    dur = len(wav) / 22050
    pauses = [(s, e, d) for s, e, d in raw if d >= 0.5 and s > 0.5 and e < dur - 0.5]
    cuts = sorted((s + e) / 2 for s, e, d in pauses)
    while len(cuts) + 1 > n:
        idx = min(range(len(cuts)),
                  key=lambda i: next(d for s, e, d in pauses if abs((s + e) / 2 - cuts[i]) < 0.01))
        cuts.pop(idx)
    segs = [0.0] + cuts + [dur]
    while len(segs) - 1 < n:
        i = max(range(len(segs) - 1), key=lambda i: segs[i + 1] - segs[i])
        segs.insert(i + 1, (segs[i] + segs[i + 1]) / 2)
    return wav, [(segs[i], segs[i + 1]) for i in range(n)], pauses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out_dir")
    ap.add_argument("id_prefix")
    ap.add_argument("lang")
    ap.add_argument("--text", required=True)
    a = ap.parse_args()

    import soundfile as sf

    sents = split_sentences(a.text)
    print(f"sentences: {len(sents)}")
    wav, segs, pauses = bounds_for(a.audio, len(sents))
    print(f"pauses>=0.5s: {len(pauses)}, clips: {len(segs)}")
    os.makedirs(a.out_dir, exist_ok=True)
    table = []
    for i, (sent, (t0, t1)) in enumerate(zip(sents, segs)):
        name = f"{i + 1:02d}.wav"
        sf.write(os.path.join(a.out_dir, name), wav[int(t0 * 22050):int(t1 * 22050)], 22050)
        table.append({"n": i + 1, "t0": round(t0, 2), "t1": round(t1, 2),
                      "dur": round(t1 - t0, 2), "text": sent, "file": name})
    with open(os.path.join(a.out_dir, "speaker_voices.json"), "w") as f:
        json.dump(
            [{"id": f"{a.id_prefix}_{t['n']:02d}", "language": a.lang, "name": f"{a.id_prefix}_{t['n']:02d}",
              "path": t["file"], "transcript_source": "recording script v2",
              "transcribed_text": t["text"]} for t in table],
            f, indent=2, ensure_ascii=False,
        )
    durs = [t["dur"] for t in table]
    print(f"clips {len(table)}, min {min(durs)}s max {max(durs)}s total {sum(durs):.1f}s")
    for t in table:
        print(f"{t['n']:>2} {t['t0']:>7} {t['t1']:>7} {t['dur']:>5}  {t['text'][:48]}")


if __name__ == "__main__":
    main()
