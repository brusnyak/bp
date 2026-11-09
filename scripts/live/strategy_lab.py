#!/usr/bin/env python3
"""Offline segmentation-strategy lab: replay ONE full-stream recording (processed/sessions/<stem>_full.wav, written when BP_LOG_SESSIONS=1)
through every strategy and score recognition error and delay on the identical audio. Recognition only: translation (~0.1 s) and voice (~0.2-0.5 s) are not in these numbers.

  PYTHONPATH=. .venv/bin/python scripts/live/strategy_lab.py processed/sessions/X_full.wav --lang sk --engine mlx \
      [--sk-model ct2_models/whisper-large-v3-turbo-sk] [--strategies pause chunk2 whole agree1.0] [--out lab.json]

Strategies
  whole      decode the entire recording once (ceiling for quality; useless for live)
  clause     pause of 0.2 s or a 4 s cap (UI option 'Clause')
  pause      the 2025-12 demo: cut at a pause of >= 0.3 s (webrtcvad) or 15 s, decode each phrase whole
  chunk2     independent 2.0 s windows (the live route before 2026-11-08)
  agreeS     decode the growing buffer every S s, commit words two consecutive decodes agree on (backend/stt/incremental.py Agreement)
Error = word error against documentation/reading_script_multilingual.md (the script read twice is allowed; unread parts are free).
Delay: single worker, computation-aware clock. emit = max(trigger time, worker free) + measured decode time.
  end_lag   emit minus the end of the phrase/window that was decoded (segment strategies) or minus the end of the committed word (agree)
  word_lag  median over words of emit minus the word's estimated end; segment words are spread evenly over the segment (approximation)
"""
import argparse, json, re, statistics as st, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SR = 16000


def load(path):
    p = Path(path)
    if p.suffix == ".f32":
        return np.fromfile(p, dtype=np.float32)
    import soundfile as sf
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if sr != SR:
        import scipy.signal as ss
        a = ss.resample_poly(a, SR, sr).astype(np.float32)
    return a


def make_engine(kind, sk_model, lang):
    if kind == "mlx":
        from backend.stt.mlx_whisper_stt import MLXWhisperSTT
        return MLXWhisperSTT()
    from faster_whisper import WhisperModel
    path = sk_model if lang == "sk" and sk_model else "large-v3-turbo"

    class FW:
        def __init__(self): self.m = WhisperModel(path, device="cpu", compute_type="int8")
        def transcribe_audio(self, a, sr, language=None, vad_filter=False, initial_prompt=None, word_timestamps=False):
            t = time.time(); segs, _ = self.m.transcribe(a, language=language, vad_filter=False, word_timestamps=word_timestamps, beam_size=5, condition_on_previous_text=False)
            segs = list(segs); return segs, time.time() - t, language
    return FW()


def words_of(segs, offset_s):
    out = []
    for s in segs:
        ws = getattr(s, "words", None)
        if ws:
            out += [(w.word.strip(), offset_s + w.start, offset_s + w.end) for w in ws if w.word.strip()]
        else:
            toks = s.text.split()
            for i, w in enumerate(toks):
                a = offset_s + s.start + (s.end - s.start) * i / max(1, len(toks)); b = offset_s + s.start + (s.end - s.start) * (i + 1) / max(1, len(toks))
                out.append((w, a, b))
    return out


def decode(eng, audio, lang, offset_s, wt=False):
    segs, dt, _ = eng.transcribe_audio(np.ascontiguousarray(audio), SR, language=lang, vad_filter=False, word_timestamps=wt)
    return words_of(segs, offset_s), dt


# ---- strategies: each returns [(emit_time_s, [(word, start, end)...], decode_s)] ----
def s_whole(eng, a, lang):
    w, dt = decode(eng, a, lang, 0.0); return [(len(a) / SR + dt, w, dt)]


def s_chunk(eng, a, lang, size=2.0):
    free, out, n = 0.0, [], int(size * SR)
    for i in range(0, len(a) - SR // 2, n):
        end = min(len(a), i + n) / SR; w, dt = decode(eng, a[i:i + n], lang, i / SR); start = max(end, free); free = start + dt; out.append((free, w, dt))
    return out


def vad_segments(a, silence=0.3, max_s=15.0, aggr=3):
    import webrtcvad
    v = webrtcvad.Vad(aggr); fr = SR // 50; segs, cur, last, t0 = [], [], None, None
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    for k in range(len(pcm) // fr):
        f = pcm[k * fr:(k + 1) * fr]; t = k * 0.02
        if v.is_speech(f.tobytes(), SR):
            if t0 is None: t0 = t
            last = t
            if last - t0 > max_s: segs.append((t0, last)); t0 = None; last = None
        elif t0 is not None and last is not None and t - last > silence:
            segs.append((t0, last + 0.1)); t0 = None; last = None
    if t0 is not None and last is not None: segs.append((t0, last + 0.1))
    return segs


def s_pause(eng, a, lang, silence=0.3, max_s=15.0):
    free, out = 0.0, []
    for (t0, t1) in vad_segments(a, silence=silence, max_s=max_s):
        t0 = max(0.0, t0 - 0.2); seg = a[int(t0 * SR):int(t1 * SR)]
        if len(seg) < SR: continue
        w, dt = decode(eng, seg, lang, t0); trig = t1 + silence; start = max(trig, free); free = start + dt; out.append((free, w, dt))
    return out


def s_agree(eng, a, lang, step):
    from backend.stt.incremental import Agreement
    ag, buf0, free, out, t = Agreement(), 0, 0.0, [], step
    while t <= len(a) / SR + step:
        now = min(t, len(a) / SR); start = max(now, free)
        w, dt = decode(eng, a[buf0:int(now * SR)], lang, buf0 / SR, wt=True)
        free = start + dt
        com = ag.update(w)
        if com: out.append((free, com, dt)); buf0 = int(com[-1][2] * SR)
        t += step
    rest = ag.prev
    if rest: out.append((free, rest, 0.0))
    return out


# ---- scoring ----
norm = lambda s: re.findall(r"[^\W_]+", s.lower())


def reference(lang, lines=None):
    """lines: e.g. ["A1","A2","A3"] = the script lines actually read (default: the whole script, unread parts free at the ends only)."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from score_reading import reference as ref
    r = ref(lang); words = []
    for sec in "ABCD":
        for i, line in enumerate(r.get(sec, []), 1):
            if not lines or f"{sec}{i}" in lines: words += norm(line)
    return words


def wer_free(ref_words, hyp, passes=2):
    ref_words = ref_words * passes; n, m = len(ref_words), len(hyp)
    if not m: return 100.0
    d = [[0] * (m + 1) for _ in range(n + 1)]   # d[i][0] = 0: the reading may start anywhere in the script
    for j in range(1, m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (ref_words[i-1] != hyp[j-1]))
    return round(100.0 * min(d[i][m] for i in range(n + 1)) / m, 1)   # per hypothesis word, unread reference at the start and end is free


PASSES = 2


def report(name, items, audio_s, ref_words):
    hyp_words, lags, ends = [], [], []
    for emit, ws, dt in items:
        for (w, s0, e0) in ws: hyp_words += norm(w); lags.append(emit - e0)
        if ws: ends.append(emit - max(e for _, _, e in ws))
    text = " ".join(w for _, ws, _ in items for (w, _, _) in ws)
    return {"strategy": name, "hyp_words": len(hyp_words), "error_pct": wer_free(ref_words, hyp_words, PASSES),
            "word_lag_median_s": round(st.median(lags), 2) if lags else None, "end_lag_median_s": round(st.median(ends), 2) if ends else None,
            "end_lag_max_s": round(max(ends), 2) if ends else None, "decodes": len(items), "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("--lang", required=True); ap.add_argument("--engine", default="mlx", choices=["mlx", "faster"])
    ap.add_argument("--sk-model", default=None); ap.add_argument("--strategies", nargs="+", default=["whole", "pause", "chunk2", "agree1.0"]); ap.add_argument("--out"); ap.add_argument("--lines", nargs="+", help="script lines actually read, e.g. A1 A2 A3 A4"); ap.add_argument("--ref-json", help="reference text from make_test_audio.py (passes=1)"); ap.add_argument("--passes", type=int, default=2, help="times the reading was repeated")
    ap.add_argument("--rescore", help="recompute error for a saved --out json with --lines/--passes, no decoding")
    a = ap.parse_args(); global PASSES; PASSES = a.passes
    if a.rescore:
        d = json.loads(Path(a.rescore).read_text()); ref = reference(d["lang"], a.lines)
        for r in d["results"]: print(r["strategy"], "error %", wer_free(ref, norm(r["text"]), PASSES), "| word_lag", r["word_lag_median_s"], "end_lag", r["end_lag_median_s"], "decodes", r["decodes"])
        return
    if a.ref_json: a.passes = 1; PASSES = 1
    audio = load(a.audio); eng = make_engine(a.engine, a.sk_model, a.lang); ref = norm(json.loads(Path(a.ref_json).read_text())["text"]) if a.ref_json else reference(a.lang, a.lines)
    print(f"{a.audio}: {len(audio) / SR:.1f} s, engine={a.engine}, lang={a.lang}", flush=True); res = []
    for s in a.strategies:
        t = time.time()
        items = (s_whole(eng, audio, a.lang) if s == "whole" else s_pause(eng, audio, a.lang) if s == "pause" else s_pause(eng, audio, a.lang, 0.2, 4.0) if s == "clause" else s_chunk(eng, audio, a.lang)
                 if s == "chunk2" else s_agree(eng, audio, a.lang, float(s[5:])))
        r = report(s, items, len(audio) / SR, ref); r["wall_s"] = round(time.time() - t, 1); res.append(r)
        print({k: v for k, v in r.items() if k != "text"}, flush=True); print("   ", r["text"][:420], flush=True)
    if a.out: Path(a.out).write_text(json.dumps({"audio": a.audio, "engine": a.engine, "lang": a.lang, "results": res}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
