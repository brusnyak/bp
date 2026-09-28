#!/usr/bin/env python3
"""Record the sentences of a recording set one by one (16 kHz mono WAV, sk_00.wav, sk_01.wav, ...).

    python scripts/record_reading.py                     # resumes where you stopped
    python scripts/record_reading.py --list-devices      # find your microphone number
    python scripts/record_reading.py --device 3 --redo 12   # re-record sentence 12

Per sentence: press Enter to START, Enter again to STOP. Then: Enter = keep, r = redo, s = skip, q = quit.
Needs: pip install sounddevice   (included in requirements-dev.txt)
"""
import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 16000


def record(sd, device):
    frames, stop = [], threading.Event()

    def cb(indata, n, t, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=device, callback=cb):
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait()
    return np.concatenate(frames)[:, 0] if frames else np.zeros(0, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(ROOT / "eval_data" / "recording_set_v2"))
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--redo", type=int, default=None, help="re-record just this sentence id")
    a = ap.parse_args()
    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices()); return
    d = Path(a.set)
    items = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    for it in items:
        path = d / f"sk_{it['id']:02d}.wav"
        if a.redo is not None and it["id"] != a.redo:
            continue
        if path.exists() and a.redo is None:
            continue
        while True:
            print(f"\n[{it['id'] + 1}/{len(items)}]  {it['sk']}\n(en: {it['en']})")
            input("  Enter = start recording ")
            print("  ● recording... Enter = stop")
            audio = record(sd, a.device)
            if len(audio) < SR // 2:
                print("  too short, try again"); continue
            peak = 20 * np.log10(max(float(np.abs(audio).max()), 1e-9))
            warn = "  TOO QUIET: move closer / raise input volume" if peak < -30 else ("  CLIPPING: lower input volume" if peak > -1 else "")
            print(f"  {len(audio) / SR:.1f}s, peak {peak:.0f} dBFS{warn}")
            ans = input("  Enter = keep, r = redo, s = skip, q = quit > ").strip().lower()
            if ans == "r":
                continue
            if ans == "q":
                print("stopped; run again to resume"); return
            if ans != "s":
                sf.write(path, audio, SR, subtype="PCM_16")
            break
    print("\nDone. Files:", len(list(d.glob('sk_*.wav'))), "in", d)


if __name__ == "__main__":
    sys.exit(main())
