#!/usr/bin/env python
"""Frozen SK->EN STT benchmark runner (STT stage only). See bench/README.md.

  python bench/run_stt.py --model ct2_models/whisper-small-sk --mode live --cap 6
Segmenter in `live` mode is copied from /Users/yegor/bp-review-2026-11-02/replay.py (replay of the main.py VAD loop in
audio time: webrtcvad aggr 3, 20 ms frames, SILENCE_TIMEOUT 0.3 s, pre-VAD RMS gate, cap on speech frames).
"""
import argparse, datetime, json, os, platform, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np, librosa
from metrics import wer, cer

SR = 16000; FR = int(SR * 0.02); PRE = int(SR * 0.5); RMS_T = 0.02; SIL_FR = int(round(0.3 / 0.02)); CHUNK = int(SR * 0.1)
CLIPS = ["me_sk_script", "me_sk_trhove", "me_sk_trhove_b"]


def segment(wav, cap):
    import webrtcvad
    vad = webrtcvad.Vad(3); segs = []; pre = []; q = []; met = False; speech = []; inseg = False; sil = 0
    def close():
        nonlocal speech, inseg
        a = np.concatenate([f for f, _ in speech]); segs.append((speech[0][1] / SR, (speech[-1][1] + FR) / SR, a)); speech = []; inseg = False
    for c0 in range(0, len(wav), CHUNK):
        ch = wav[c0:c0 + CHUNK]
        if not met:
            pre.append((ch, c0))
            if sum(len(x) for x, _ in pre) >= PRE:
                buf = np.concatenate([x for x, _ in pre])
                if np.sqrt(np.mean(buf ** 2)) > RMS_T:
                    met = True; q = [(buf, pre[0][1])]
                pre = []
        else:
            q.append((ch, c0))
        if met:
            buf = np.concatenate([x for x, _ in q]); base = q[0][1]; q = []; i = 0
            while i + FR <= len(buf):
                f = buf[i:i + FR]; t = base + i
                sp = vad.is_speech((np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes(), SR)
                if sp:
                    speech.append((f, t)); inseg = True; sil = 0
                    if len(speech) * 0.02 > cap:
                        close(); met = False; pre = []; buf = buf[:0]; i = 0; break
                elif inseg:
                    sil += 1
                    if sil > SIL_FR:
                        close(); met = False; pre = []; buf = buf[:0]; i = 0; sil = 0; break
                i += FR
            else:
                q = [(buf[i:], base + i)] if i < len(buf) else []
    if speech: close()
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="CT2 dir (relative to repo ok) or faster-whisper size name")
    ap.add_argument("--mode", choices=["whole", "live"], required=True)
    ap.add_argument("--cap", type=float, default=6.0, help="live: MAX_SEGMENT_S")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--clips", nargs="+", default=CLIPS)
    ap.add_argument("--language", default="sk")
    ap.add_argument("--local-only", action="store_true", default=True, help="never download (default)")
    ap.add_argument("--tag", default=None, help="name used in the output file (default: model basename)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    import faster_whisper, ctranslate2
    from faster_whisper import WhisperModel
    mp = a.model if os.path.isabs(a.model) else os.path.join(REPO, a.model)
    src = mp if os.path.isdir(mp) else a.model
    model = WhisperModel(src, device="cpu", compute_type="int8", local_files_only=True)
    def tr(audio):
        kw = dict(language=a.language, beam_size=a.beam, temperature=a.temperature, vad_filter=False)
        if a.mode == "live": kw["no_speech_threshold"] = None  # same as backend wrapper when external VAD is used
        segs, _ = model.transcribe(audio, **kw)
        return " ".join(s.text.strip() for s in list(segs)).strip()
    out = []
    for clip in a.clips:
        wav, _ = librosa.load(f"{REPO}/speaker_voices/{clip}.m4a", sr=SR, mono=True); wav = wav.astype(np.float32)
        ref_path = f"bench/refs/{clip}.txt"; ref = open(os.path.join(REPO, ref_path)).read().strip()
        t0 = time.perf_counter(); n = None; segs_info = None
        if a.mode == "whole":
            hyp = tr(wav)
        else:
            segs = segment(wav, a.cap); n = len(segs)
            hyp = " ".join(tr(x[2]) for x in segs).strip()
            segs_info = [[round(s, 2), round(e, 2)] for s, e, _ in segs]
        wall = time.perf_counter() - t0; dur = len(wav) / SR
        out.append(dict(clip=clip, wer=round(wer(ref, hyp), 4), cer=round(cer(ref, hyp), 4), hypothesis=hyp, reference_path=ref_path,
                        audio_seconds=round(dur, 2), wall_seconds=round(wall, 2), rtf=round(wall / dur, 3), n_segments=n, segment_bounds=segs_info))
        print(clip, a.mode, "WER", out[-1]["wer"], "CER", out[-1]["cer"], "RTF", out[-1]["rtf"], flush=True)
    try: sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception: sha = None
    tag = a.tag or os.path.basename(os.path.normpath(a.model))
    res = dict(model=a.model, model_tag=tag, mode=a.mode, cap=a.cap if a.mode == "live" else None, beam=a.beam, temperature=a.temperature,
               language=a.language, compute_type="int8", device="cpu",
               machine=f"{platform.machine()} CPU int8", python=platform.python_version(), faster_whisper=faster_whisper.__version__,
               ctranslate2=ctranslate2.__version__, git_head=sha, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               clips=out)
    os.makedirs(a.out_dir, exist_ok=True)
    name = f"{datetime.date.today().isoformat()}_{tag}_{a.mode}" + (f"{int(a.cap)}" if a.mode == "live" else "") + ".json"
    p = os.path.join(a.out_dir, name); k = 1
    while os.path.exists(p):  # never overwrite: repeat runs get _run2, _run3
        k += 1; p = os.path.join(a.out_dir, name[:-5] + f"_run{k}.json")
    json.dump(res, open(p, "w"), ensure_ascii=False, indent=1); print("wrote", p)

if __name__ == "__main__":
    main()
