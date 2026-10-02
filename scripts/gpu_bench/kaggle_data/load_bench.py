#!/usr/bin/env python3
"""Full-cascade load bench: STT -> MT -> TTS(OmniVoice clone) on one GPU, three experiments.

  sweep     speech length 2/5/10/20/40 s, EN->SK and SK->EN, n=1: per-stage time, time-to-first-audio (ttfa), RTF
  ramp      N concurrent speaker streams (open loop: each stream "speaks" in real time, so a slow server falls behind
            instead of being throttled). duty=1.0 = everyone talks all the time (worst case), duty=0.25 = realistic meeting.
  timeline  3-minute presentation on a virtual clock: segment i is spoken at t_i, the single worker processes serially,
            report lag behind the speaker and whether the backlog stays bounded.

Shared models, one process, TTS serialised by a lock (like the app's single TTS worker). STT/MT use the same libraries as the
app (faster-whisper, CTranslate2 Opus-MT). Inputs: --inputs dir with lc_*.wav + lc_*.txt (+ refs en/sk .wav/.txt for the clone).
Assumed constants (not measured, recorded in the JSON): VAD hangover 0.5 s, pause between segments 0.4 s.

  python load_bench.py --models-dir ct2 --inputs inputs --out results --tts omni      # GPU
  python load_bench.py --models-dir ../../ct2_models --inputs ... --tts stub --device cpu --levels 1,2 --duration 20   # smoke
"""
from __future__ import annotations

import argparse, glob, json, os, random, re, statistics, subprocess, threading, time

import numpy as np
import soundfile as sf

SR = 16000
VAD_HANGOVER_S, SEG_PAUSE_S = 0.5, 0.4
LANG = {"en": "sk", "sk": "en"}  # source -> target
OPUS = {"en": "Helsinki-NLP--opus-mt-en-sk", "sk": "Helsinki-NLP--opus-mt-sk-en"}


# ---------------------------------------------------------------- models
class Cascade:
    def __init__(self, a):
        import ctranslate2
        from faster_whisper import WhisperModel
        from transformers import AutoTokenizer
        self.dev = a.device
        ct = "float16" if a.device == "cuda" else "int8"
        md = a.models_dir
        en_dir = os.path.join(md, "whisper-base")
        self.stt = {"en": WhisperModel(en_dir if os.path.exists(en_dir) else "base", device=a.device, compute_type=ct),
                    "sk": WhisperModel(os.path.join(md, "whisper-small-sk"), device=a.device, compute_type=ct)}
        self.mt, self.tok = {}, {}
        for src, name in OPUS.items():
            d = os.path.join(md, name)
            self.mt[src] = ctranslate2.Translator(d, device=a.device, compute_type=ct)
            self.tok[src] = AutoTokenizer.from_pretrained(d)
        self.lock = threading.Lock()  # TTS worker
        self.tts_kind = a.tts
        self.refs = {}
        if a.tts == "omni":
            import torch
            from omnivoice import OmniVoice
            self.torch = torch
            dtype = torch.float16 if a.device == "cuda" else torch.float32
            self.omni = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=a.device, dtype=dtype)
            self.sr = self.omni.sampling_rate
            for tgt in ("en", "sk"):  # the clone speaks the TARGET language
                txt = open(os.path.join(a.inputs, f"{tgt}.txt"), encoding="utf-8").read().strip()
                self.refs[tgt] = self.omni.create_voice_clone_prompt(os.path.join(a.inputs, f"{tgt}.wav"), txt)
        else:
            self.sr = 24000

    def run_stt(self, audio, src):
        t0 = time.perf_counter()
        segs, _ = self.stt[src].transcribe(audio, language=src, beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs)
        return text, time.perf_counter() - t0

    def run_mt(self, sents, src):
        if not sents:
            return [], 0.0
        t0 = time.perf_counter()
        toks = [self.tok[src].convert_ids_to_tokens(self.tok[src].encode(s, add_special_tokens=True)) for s in sents]
        res = self.mt[src].translate_batch(toks, beam_size=4, max_batch_size=16)
        out = [self.tok[src].decode(self.tok[src].convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True) for r in res]
        return out, time.perf_counter() - t0

    def run_tts(self, text, tgt):
        """returns (audio_seconds, wait_s, run_s)"""
        tw = time.perf_counter()
        with self.lock:
            t0 = time.perf_counter()
            if self.tts_kind == "omni":
                audio = self.omni.generate(text=text, language=tgt, voice_clone_prompt=self.refs[tgt], num_step=16)[0]
                if self.dev == "cuda":
                    self.torch.cuda.synchronize()
                dur = len(audio) / self.sr
            else:
                dur = 0.07 * len(text)
            return dur, t0 - tw, time.perf_counter() - t0

    def utterance(self, audio, src, ref_text=None):
        """full cascade for one VAD-endpointed utterance, sentence-chunked TTS (as the live app does)."""
        t0 = time.perf_counter()
        text, stt_s = self.run_stt(audio, src)
        sents = [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s] or [text or "."]
        out, mt_s = self.run_mt(sents, src)
        tgt = LANG[src]
        tts_run = tts_wait = out_audio = 0.0
        ttfa = None
        for s in out:
            dur, w, r = self.run_tts(s, tgt)
            tts_wait += w; tts_run += r; out_audio += dur
            if ttfa is None:
                ttfa = time.perf_counter() - t0
        total = time.perf_counter() - t0
        row = {"src": src, "in_s": round(len(audio) / SR, 2), "stt_s": round(stt_s, 3), "mt_s": round(mt_s, 3),
               "tts_run_s": round(tts_run, 3), "tts_wait_s": round(tts_wait, 3), "ttfa_s": round(ttfa or total, 3),
               "total_s": round(total, 3), "out_s": round(out_audio, 2), "n_sent": len(out), "text": text}
        if ref_text:
            import jiwer
            norm = lambda t: re.sub(r"[^\w\s]", "", t.lower())
            row["wer_vs_idle"] = round(jiwer.wer(norm(ref_text), norm(text)), 3)
        return row


# ---------------------------------------------------------------- inputs
def load_clips(d):
    clips = {"en": [], "sk": [], "en_s": [], "sk_s": []}  # *_s = ~2-3 s clips for the shortest sweep point
    for w in sorted(glob.glob(os.path.join(d, "lc_*.wav"))) + sorted(glob.glob(os.path.join(d, "ls_*.wav"))):
        lang = os.path.basename(w).split("_")[1] + ("_s" if os.path.basename(w).startswith("ls_") else "")
        audio, sr = sf.read(w)
        audio = audio.mean(1) if audio.ndim > 1 else audio
        if sr != SR:
            import librosa
            audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=SR)
        clips[lang].append([audio.astype("float32"), None])  # ref text filled by idle_refs()
    return clips


def idle_refs(c, clips):
    """no ground-truth scripts per segment: the reference is the idle (n=1, unloaded) transcript of the same clip, so every
    'wer' below is drift versus idle, not absolute accuracy (absolute WER lives in processed/stt_baseline.json)."""
    for key, pool in clips.items():
        for item in pool:
            item[1] = c.run_stt(item[0], key[:2])[0]


def compose(clips, lang, target_s, rng=None):
    """concatenate clips (cycling) until >= target_s; returns (audio, reference_text)."""
    pool, a, t, total = clips[lang + ("_s" if target_s <= 3 else "")], [], [], 0.0
    i = 0
    while total < target_s - 0.5:
        au, tx = pool[i % len(pool)]
        a += [au, np.zeros(int(0.3 * SR), "float32")]; t.append(tx); total += len(au) / SR + 0.3; i += 1
        if len(pool) == 1 and total >= target_s:
            break
    return np.concatenate(a), " ".join(t)


def pctl(x, p):
    x = sorted(x)
    return round(x[min(len(x) - 1, int(round(p / 100 * (len(x) - 1))))], 3) if x else None


# ---------------------------------------------------------------- GPU sampler
class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.rows = []; self.stop = False
    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.rows.append((time.time(), float(o[0]), float(o[1])))
            except Exception:
                return
            time.sleep(0.5)
    def summary(self, t0, t1):
        r = [x for x in self.rows if t0 <= x[0] <= t1]
        return {"vram_max_mb": max((x[1] for x in r), default=None),
                "gpu_util_mean": round(statistics.mean(x[2] for x in r), 1) if r else None,
                "gpu_util_max": max((x[2] for x in r), default=None)}


# ---------------------------------------------------------------- experiments
def exp_sweep(c, clips, a):
    rows = []
    for src in ("en", "sk"):
        for target in (2, 5, 10, 20, 40):
            audio, ref = compose(clips, src, target)
            for rep in range(a.reps):
                r = c.utterance(audio, src, ref); r["target_s"] = target; r["rep"] = rep
                r["rtf"] = round(r["total_s"] / r["in_s"], 3)
                rows.append(r); print("sweep", src, target, r["in_s"], "ttfa", r["ttfa_s"], "total", r["total_s"], "rtf", r["rtf"], "wer_vs_idle", r.get("wer_vs_idle"), flush=True)
    return rows


def exp_ramp(c, clips, a, sampler):
    levels = [int(x) for x in a.levels.split(",")]
    out = []
    for duty in (1.0, 0.25):
        for n in levels:
            seg = {s: compose(clips, s, 8.0) for s in ("en", "sk")}
            results, lock = [], threading.Lock()
            t_start = time.time(); end = time.perf_counter() + a.duration
            def stream(i):
                src = "en" if i % 2 == 0 else "sk"
                audio, ref = seg[src]
                dur = len(audio) / SR
                period = dur / duty
                nxt = time.perf_counter() + random.uniform(0, period)
                while nxt < end:
                    time.sleep(max(0.0, nxt - time.perf_counter()))
                    r = c.utterance(audio, src, ref)
                    r["lag_s"] = round(time.perf_counter() - nxt, 3)  # finish - scheduled arrival (includes catch-up)
                    r["lag_before_s"] = round(time.perf_counter() - nxt - r["total_s"], 3)  # how late work even started
                    with lock:
                        results.append(r)
                    nxt += period
            ts = [threading.Thread(target=stream, args=(i,)) for i in range(n)]
            [t.start() for t in ts]; [t.join() for t in ts]
            t_end = time.time()
            ok = results
            row = {"duty": duty, "streams": n, "n_utts": len(ok),
                   "ttfa_p50": pctl([r["ttfa_s"] + r["lag_before_s"] for r in ok], 50),
                   "ttfa_p95": pctl([r["ttfa_s"] + r["lag_before_s"] for r in ok], 95),
                   "lag_p50": pctl([r["lag_s"] for r in ok], 50), "lag_p95": pctl([r["lag_s"] for r in ok], 95),
                   "lag_max": max((r["lag_s"] for r in ok), default=None),
                   "work_rtf": round(sum(r["total_s"] for r in ok) / max(1e-9, sum(r["in_s"] for r in ok)), 3),
                   "tts_wait_mean": round(statistics.mean(r["tts_wait_s"] for r in ok), 3) if ok else None,
                   "stt_s_mean": round(statistics.mean(r["stt_s"] for r in ok), 3) if ok else None,
                   "tts_run_mean": round(statistics.mean(r["tts_run_s"] for r in ok), 3) if ok else None,
                   "wer_mean": round(statistics.mean(r["wer_vs_idle"] for r in ok if "wer_vs_idle" in r), 3) if ok else None,
                   "out_audio_ratio": round(sum(r["out_s"] for r in ok) / max(1e-9, sum(r["in_s"] for r in ok)), 2),
                   "wall_s": round(t_end - t_start, 1), **sampler.summary(t_start, t_end)}
            row["keeps_up"] = bool(row["lag_p95"] is not None and row["lag_p95"] < 6.0 and row["lag_max"] < 15.0)
            out.append(row); print("ramp", row, flush=True)
            if not row["keeps_up"] and row["lag_p95"] and row["lag_p95"] > 40:
                print("  overloaded hard; skipping higher levels for duty", duty); break
    return out


def exp_timeline(c, clips, a):
    """virtual-clock presentation: one monologue EN->SK (A) and an alternating 2-speaker meeting (B)."""
    scen = {}
    for name, langs in (("monologue_en", ["en"] * 22), ("alternating_en_sk", ["en", "sk"] * 11)):
        pools = {"en": 0, "sk": 0}
        segs, t = [], 0.0
        for lang in langs:
            au, tx = clips[lang][pools[lang] % len(clips[lang])]; pools[lang] += 1
            speech_end = t + len(au) / SR
            segs.append({"lang": lang, "audio": au, "ref": tx, "t_start": round(t, 2), "t_end": round(speech_end, 2)})
            t = speech_end + SEG_PAUSE_S
        free, play_end, rows = 0.0, 0.0, []
        for i, s in enumerate(segs):
            avail = s["t_end"] + VAD_HANGOVER_S
            start = max(avail, free)
            r = c.utterance(s["audio"], s["lang"], s["ref"])  # measured on the real GPU, run for real
            ready_first, ready_all = start + r["ttfa_s"], start + r["total_s"]
            free = ready_all
            play_start = max(ready_first, play_end)
            play_end = max(play_end, ready_first) + r["out_s"]
            rows.append({"i": i, "lang": s["lang"], "t_start": s["t_start"], "t_end": s["t_end"], "in_s": r["in_s"],
                         "queue_wait_s": round(start - avail, 2), "ttfa_s": r["ttfa_s"], "total_s": r["total_s"],
                         "first_audio_at": round(ready_first, 2), "listener_delay_s": round(play_start - s["t_end"], 2),
                         "out_s": r["out_s"], "stt_s": r["stt_s"], "mt_s": r["mt_s"], "tts_run_s": r["tts_run_s"], "wer_vs_idle": r.get("wer_vs_idle")})
            print("timeline", name, i, "listener_delay", rows[-1]["listener_delay_s"], "queue", rows[-1]["queue_wait_s"], flush=True)
        d = [r["listener_delay_s"] for r in rows]
        scen[name] = {"segments": rows, "speech_total_s": round(segs[-1]["t_end"], 1),
                      "listener_delay_p50": pctl(d, 50), "listener_delay_p95": pctl(d, 95), "listener_delay_max": max(d),
                      "queue_wait_max": max(r["queue_wait_s"] for r in rows),
                      "bounded": bool(rows[-1]["listener_delay_s"] <= 2 * statistics.median(d) + 3)}
    return scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="ct2"); ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--out", default="results"); ap.add_argument("--tts", default="omni", choices=["omni", "stub"])
    ap.add_argument("--device", default="cuda"); ap.add_argument("--levels", default="1,2,4,8,12,16,24,32")
    ap.add_argument("--duration", type=float, default=90.0); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="sweep,ramp,timeline")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips = load_clips(a.inputs)
    c = Cascade(a)
    # warm-up (CUDA kernels, first-call allocs) - discarded
    for src in ("en", "sk"):
        c.utterance(clips[src][0][0], src)
    idle_refs(c, clips)
    sampler = Sampler(); sampler.start()
    machine = {"date": time.strftime("%Y-%m-%d %H:%M"), "device": a.device, "tts": a.tts,
               "assumed": {"vad_hangover_s": VAD_HANGOVER_S, "segment_pause_s": SEG_PAUSE_S}}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0); machine.update({"gpu": p.name, "vram_gb": round(p.total_memory / 1e9, 1)})
    except Exception:
        pass
    res = {"machine": machine}
    for name, fn in (("sweep", lambda: exp_sweep(c, clips, a)), ("ramp", lambda: exp_ramp(c, clips, a, sampler)),
                     ("timeline", lambda: exp_timeline(c, clips, a))):
        if name in a.only.split(","):
            res[name] = fn()
            json.dump(res, open(os.path.join(a.out, "load_bench.json"), "w"), indent=1, ensure_ascii=False)
    sampler.stop = True
    print("wrote", os.path.join(a.out, "load_bench.json"))


if __name__ == "__main__":
    main()
