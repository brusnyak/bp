#!/usr/bin/env python
"""Variant runner (T1 prompt / T2 VAD + anti-hallucination flags). Reuses run_stt.py helpers. New file; run_stt.py untouched.
  .venv/bin/python bench/run_stt_vad.py --model ct2_models/whisper-small-sk --tag naive --mode whole --variants base prompt vad_default ...
"""
import argparse, datetime, json, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE)
import numpy as np, librosa
from metrics import wer, cer
from run_stt import segment, SR, CLIPS

PROMPT = json.load(open(os.path.join(REPO, "languages/packs/sk-en.json")))["initial_prompt"] if False else None
def _prompt():
    d = json.load(open(os.path.join(REPO, "languages/packs/sk-en.json")))
    def find(o):
        if isinstance(o, dict):
            if "initial_prompt" in o: return o["initial_prompt"]
            for v in o.values():
                r = find(v)
                if r: return r
    return find(d)
AH = dict(no_speech_threshold=0.6, compression_ratio_threshold=2.4, log_prob_threshold=-1.0)
VADD = dict(vad_filter=True)
VADB = dict(vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400))
VARIANTS = {
    "base": {}, "prompt": {"initial_prompt": "@P"},
    "vad_default": VADD, "vad_b": VADB, "nocond": {"condition_on_previous_text": False}, "antihall": AH,
    "combo_vadb_nocond_ah": {**VADB, "condition_on_previous_text": False, **AH},
    "combo_vadd_nocond_ah": {**VADD, "condition_on_previous_text": False, **AH},
    "combo_vadb_ah": {**VADB, **AH}, "combo_vadb_nocond": {**VADB, "condition_on_previous_text": False},
}
def loops(words):
    """count n-gram (n=1..3) immediately repeated >=3 times in a row; each run counts once"""
    n_runs = 0
    for n in (1, 2, 3):
        i = 0
        while i + 3 * n <= len(words):
            g = words[i:i + n]; k = 1
            while words[i + k * n:i + (k + 1) * n] == g: k += 1
            if k >= 3: n_runs += 1; i += k * n
            else: i += 1
    return n_runs
def czechlike(words): return sum(1 for w in words if re.search(r"[ěřů]", w))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["whole", "live"], default="whole"); ap.add_argument("--cap", type=float, default=6.0)
    ap.add_argument("--variants", nargs="+", required=True); ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--clips", nargs="+", default=CLIPS)
    a = ap.parse_args()
    import faster_whisper
    from faster_whisper import WhisperModel
    mp = a.model if os.path.isabs(a.model) else os.path.join(REPO, a.model)
    model = WhisperModel(mp, device="cpu", compute_type="int8", local_files_only=True)
    audio = {}; segs = {}
    for c in a.clips:
        w, _ = librosa.load(f"{REPO}/speaker_voices/{c}.m4a", sr=SR, mono=True); audio[c] = w.astype(np.float32)
        if a.mode == "live": segs[c] = segment(audio[c], a.cap)
    for v in a.variants:
        opts = {k: (_prompt() if x == "@P" else x) for k, x in VARIANTS[v].items()}
        out = []
        for c in a.clips:
            ref = open(os.path.join(REPO, f"bench/refs/{c}.txt")).read().strip(); kept = 0.0; tot = 0.0
            def tr(x):
                nonlocal kept, tot
                kw = dict(language="sk", beam_size=a.beam, temperature=0.0, vad_filter=False); kw.update(opts)
                if a.mode == "live" and "no_speech_threshold" not in opts: kw["no_speech_threshold"] = None
                ss, info = model.transcribe(x, **kw); txt = " ".join(s.text.strip() for s in list(ss)).strip()
                tot += info.duration; kept += info.duration_after_vad if kw.get("vad_filter") else info.duration
                return txt
            t0 = time.perf_counter()
            hyp = tr(audio[c]) if a.mode == "whole" else " ".join(tr(s[2]) for s in segs[c]).strip()
            wall = time.perf_counter() - t0; dur = len(audio[c]) / SR
            hw = re.sub(r"[^\w\s]", "", hyp.lower()).split()
            out.append(dict(clip=c, wer=round(wer(ref, hyp), 4), cer=round(cer(ref, hyp), 4), hypothesis=hyp, audio_seconds=round(dur, 2),
                            wall_seconds=round(wall, 2), rtf=round(wall / dur, 3), vad_kept_seconds=round(kept, 2), vad_total_seconds=round(tot, 2),
                            hyp_words=len(hw), loops=loops(hw), czech_like_words=czechlike(hw), n_segments=len(segs[c]) if a.mode == "live" else None))
            print(a.tag, v, a.mode, c, "WER", out[-1]["wer"], "CER", out[-1]["cer"], "RTF", out[-1]["rtf"], "kept", out[-1]["vad_kept_seconds"], "/", out[-1]["vad_total_seconds"], "loops", out[-1]["loops"], "cz", out[-1]["czech_like_words"], flush=True)
        shown = {k: ("<initial_prompt from languages/packs/sk-en.json>" if x == "@P" else x) for k, x in VARIANTS[v].items()}
        res = dict(model=a.model, model_tag=f"{a.tag}+{v}", mode=a.mode, cap=a.cap if a.mode == "live" else None, beam=a.beam, temperature=0.0, language="sk",
                   compute_type="int8", device="cpu", variant=v, transcribe_options=shown, faster_whisper=faster_whisper.__version__,
                   utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), clips=out)
        name = f"{datetime.date.today().isoformat()}_{a.tag}+{v}_{a.mode}" + (f"{int(a.cap)}" if a.mode == "live" else "") + ".json"
        p = os.path.join(HERE, "results", name); k = 1
        while os.path.exists(p): k += 1; p = os.path.join(HERE, "results", name[:-5] + f"_run{k}.json")
        json.dump(res, open(p, "w"), ensure_ascii=False, indent=1); print("wrote", p, flush=True)
if __name__ == "__main__": main()
