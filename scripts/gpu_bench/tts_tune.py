#!/usr/bin/env python3
"""OmniVoice latency/pace tuning on a GPU: what makes the first translated audio faster and the clone's pace match the speaker?

Uses load_bench.Cascade (same STT/MT/TTS objects) on the owner's real segments. For N translated sentences per direction:
  A. grid  num_step {32,16,12,8} x speed {1.0,1.2,1.35}: gen time, audio length, chars/s, round-trip WER (STT of the output)
  B. first-chunk: time to synthesize only the first clause (<=6 words) vs the whole first sentence
  C. duration matching: duration = source speech seconds allocated by characters per sentence (isochrony), ratio out/in + WER
  D. voice-prompt cost: create_voice_clone_prompt cold vs warm (what "prep at enrollment" would save)
Round-trip WER = jiwer(sentence text, STT(output audio)): quality guard against the speed/steps cliff.
Out: results/tts_tune.json
"""
from __future__ import annotations

import argparse, json, os, re, statistics as st, time

import numpy as np

import load_bench as lb


def norm(t):
    return re.sub(r"[^\w\s]", "", t.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="ct2"); ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--out", default="results"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--tts", default="omni"); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import jiwer, torch
    clips = lb.load_clips(a.inputs)
    c = lb.Cascade(a)
    om = c.omni
    res = {"machine": {"date": time.strftime("%Y-%m-%d %H:%M"), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else a.device}}

    # D. prompt cost (cold = first call after load, warm = repeat)
    D = []
    for tgt in ("en", "sk"):
        txt = open(os.path.join(a.inputs, f"{tgt}.txt"), encoding="utf-8").read().strip()
        for tag in ("first", "second", "third"):
            t0 = time.perf_counter()
            om.create_voice_clone_prompt(os.path.join(a.inputs, f"{tgt}.wav"), txt)
            if a.device == "cuda":
                torch.cuda.synchronize()
            D.append({"lang": tgt, "call": tag, "s": round(time.perf_counter() - t0, 3)})
    res["prompt_cost"] = D
    print("prompt", D, flush=True)

    # test material: real segments -> STT -> MT. items: dict(dir, src_s, sentences(tgt text), src_text)
    items = {"en": [], "sk": []}
    for src in ("en", "sk"):
        for audio, _ in clips[src][: a.n]:
            text, _ = c.run_stt(audio, src)
            sents = [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s] or [text]
            out, _ = c.run_mt(sents, src)
            items[src].append({"src_s": len(audio) / lb.SR, "sents": out, "src_sents": sents})
    # warm
    for src in ("en", "sk"):
        tgt = lb.LANG[src]
        om.generate(text=items[src][0]["sents"][0], language=tgt, voice_clone_prompt=c.refs[tgt], num_step=16)

    def gen(text, tgt, **kw):
        if a.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        wav = om.generate(text=text, language=tgt, voice_clone_prompt=c.refs[tgt], **kw)[0]
        if a.device == "cuda":
            torch.cuda.synchronize()
        return wav, time.perf_counter() - t0

    def rt_wer(wav, text, tgt):
        a16 = np.asarray(wav, dtype="float32")
        if om.sampling_rate != lb.SR:
            import librosa
            a16 = librosa.resample(a16, orig_sr=om.sampling_rate, target_sr=lb.SR)
        hyp, _ = c.run_stt(a16, tgt)
        return jiwer.wer(norm(text), norm(hyp)) if norm(text) else 0.0

    # A. grid (first sentence of each item)
    A = []
    for steps in (32, 16, 12, 8):
        for speed in (1.0, 1.2, 1.35):
            for src in ("en", "sk"):
                tgt = lb.LANG[src]
                rows = []
                for it in items[src]:
                    s = it["sents"][0]
                    wav, dt = gen(s, tgt, num_step=steps, speed=speed)
                    dur = len(wav) / om.sampling_rate
                    rows.append({"gen_s": dt, "audio_s": dur, "chars_s": len(s) / dur if dur else 0, "wer": rt_wer(wav, s, tgt), "chars": len(s)})
                g = lambda k: round(st.mean(r[k] for r in rows), 3)
                A.append({"dir": f"{src}->{tgt}", "num_step": steps, "speed": speed, "n": len(rows), "gen_s": g("gen_s"), "audio_s": g("audio_s"),
                          "chars_per_s": g("chars_s"), "rtf": round(sum(r["gen_s"] for r in rows) / sum(r["audio_s"] for r in rows), 3),
                          "wer": g("wer"), "mean_chars": g("chars")})
                print("grid", A[-1], flush=True)
    res["grid"] = A

    # B. first clause vs first sentence
    B = []
    for src in ("en", "sk"):
        tgt = lb.LANG[src]
        for it in items[src]:
            s = it["sents"][0]
            words = s.split()
            first = " ".join(words[:6]) if len(words) > 8 else s
            _, t_full = gen(s, tgt, num_step=16)
            _, t_first = gen(first, tgt, num_step=16)
            B.append({"dir": f"{src}->{tgt}", "words": len(words), "full_s": round(t_full, 3), "first6_s": round(t_first, 3)})
    res["first_chunk"] = B
    print("first_chunk mean", {d: (round(st.mean(b["full_s"] for b in B if b["dir"] == d), 2), round(st.mean(b["first6_s"] for b in B if b["dir"] == d), 2)) for d in ("en->sk", "sk->en")}, flush=True)

    # C. duration matching: allocate source seconds over the translated sentences by character share
    C = []
    for src in ("en", "sk"):
        tgt = lb.LANG[src]
        for it in items[src]:
            chars = [max(1, len(s)) for s in it["sents"]]
            tot = sum(chars)
            for mode in ("free", "match1.0", "match0.85"):
                out_s, gen_s, wers = 0.0, 0.0, []
                for s, ch in zip(it["sents"], chars):
                    kw = {"num_step": 16}
                    if mode != "free":
                        kw["duration"] = it["src_s"] * (ch / tot) * (1.0 if mode == "match1.0" else 0.85)
                    wav, dt = gen(s, tgt, **kw)
                    out_s += len(wav) / om.sampling_rate; gen_s += dt
                    wers.append(rt_wer(wav, s, tgt))
                C.append({"dir": f"{src}->{tgt}", "mode": mode, "src_s": round(it["src_s"], 2), "out_s": round(out_s, 2), "ratio": round(out_s / it["src_s"], 3),
                          "gen_s": round(gen_s, 2), "wer": round(st.mean(wers), 3)})
    res["duration_match"] = C
    for d in ("en->sk", "sk->en"):
        for m in ("free", "match1.0", "match0.85"):
            r = [x for x in C if x["dir"] == d and x["mode"] == m]
            print("dur", d, m, "ratio", round(st.mean(x["ratio"] for x in r), 3), "gen", round(st.mean(x["gen_s"] for x in r), 2), "wer", round(st.mean(x["wer"] for x in r), 3), flush=True)

    json.dump(res, open(os.path.join(a.out, "tts_tune.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote tts_tune.json")


if __name__ == "__main__":
    main()
