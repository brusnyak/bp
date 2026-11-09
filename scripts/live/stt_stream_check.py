"""Isolated streaming-recogniser check (no server, MT or TTS): feeds one WAV to the Nemotron engine in 0.32 s pushes as fast as it accepts them and prints
compute RTF, slowest push, first-output audio time and, with REF=<testaudio json>, word error.
  REF=processed/testaudio/sk_paced.json HEAVY_QOS=normal ~/.claude/scripts/heavy -- .venv/bin/python -I scripts/live/stt_stream_check.py processed/testaudio/sk_paced.wav sk 560
(args: wav, language, delay ms). Not real-time pacing: lag needs route_check.py."""
import os, sys, time, json
sys.path.insert(0, os.getcwd())
import numpy as np, soundfile as sf, scipy.signal as ss
path, lang, delay = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["BP_NEMOTRON_DELAY_MS"] = delay
a, sr = sf.read(path, dtype="float32")
if a.ndim > 1: a = a.mean(1)
if sr != 16000: a = ss.resample_poly(a, 16000, sr).astype(np.float32)
a = a[: 16000 * 45]
from backend.stt.nemotron_stt import NemotronSTT
t0 = time.time(); e = NemotronSTT(); load_s = time.time() - t0
st = e.open_stream(lang)
step = int(16000 * float(os.environ.get("STEP", "0.32")))
pushes, out = [], []
for i in range(0, len(a), step):
    last = i + step >= len(a)
    t = time.time(); w = e.stream_feed(st, a[i:i + step], final=last); dt = time.time() - t
    pushes.append(dt)
    if w: out.append((round((i + step) / 16000, 2), w))
p = np.array(pushes)
print(json.dumps({"delay_ms": delay, "lang": lang, "audio_s": round(len(a) / 16000, 1), "load_s": round(load_s, 1), "total_compute_s": round(p.sum(), 1),
                  "rtf": round(p.sum() / (len(a) / 16000), 3), "push_median_s": round(float(np.median(p)), 3), "push_max_s": round(float(p.max()), 3),
                  "first_output_at_audio_s": out[0][0] if out else None}))
hyp = " ".join(" ".join(x[1]) for x in out)
print(hyp[:900])
ref_path = os.environ.get("REF")
if ref_path:
    import re, unicodedata
    ref = json.load(open(ref_path, encoding="utf-8"))["text"]
    def norm(s, fold):
        s = s.lower()
        if fold: s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return re.findall(r"\w+", s)
    def wer(r, h):
        d = list(range(len(h) + 1))
        for i, rw in enumerate(r, 1):
            p, d[0] = d[0], i
            for j, hw in enumerate(h, 1):
                p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (rw != hw))
        return d[-1] / len(r)
    print("WER strict %.1f%%  ascii-folded %.1f%%  (ref words %d)" % (100 * wer(norm(ref, 0), norm(hyp, 0)), 100 * wer(norm(ref, 1), norm(hyp, 1)), len(norm(ref, 0))))
