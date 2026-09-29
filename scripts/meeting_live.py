#!/usr/bin/env python3
"""Bidirectional meeting through the REAL live pipeline (/ws), not a simulation.

For each scripted turn: synthesize the speaker clip (personal Piper voices),
stream it through `live_direction_probe.py` (20ms chunks, real VAD/STT/MT/TTS),
capture timestamped events + TTS audio, and lay all turns on one meeting clock.

Needs `make run` (wss://localhost:8000). Out: processed/meeting/ (same schema
as meeting_sim.py plus "live": true and per-utterance event times).

Run: .venv/bin/python scripts/meeting_live.py [--gap 0.8]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
OUT_DIR = os.path.join(REPO_ROOT, "processed", "meeting")

from scripts.meeting_sim import TURNS, SPEAKER_VOICE  # noqa: E402


def wav_duration(path: str) -> float:
    import soundfile as sf
    return len(sf.read(path)[0]) / sf.info(path).samplerate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.8)
    ap.add_argument("--tts-sk", default="piper_sk_personal",
                    help="explicit engine for SK output (no remap ambiguity)")
    ap.add_argument("--tts-en", default="piper_personal_v2",
                    help="explicit engine for EN output (no remap ambiguity)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the throwaway warmup turn (cold model load then taxes turn 0)")
    ap.add_argument("--tag", default="",
                    help="filename tag for this run (e.g. generic), keeps runs side by side")
    ap.add_argument("--speak-sk", default=None, help="speaker voice for SK turns (default: personal)")
    ap.add_argument("--speak-en", default=None, help="speaker voice for EN turns (default: personal)")
    args = ap.parse_args()
    TAG = (args.tag + "_" if args.tag else "")
    os.makedirs(OUT_DIR, exist_ok=True)
    # Fresh run: stale dumps/raws from an aborted run must never contaminate timing.
    import glob
    for stale in glob.glob(os.path.join(OUT_DIR, f"live_dump_{TAG}*.json")) + \
            glob.glob(os.path.join(OUT_DIR, f"live_tts_{TAG}*.raw.wav")):
        os.remove(stale)

    import numpy as np
    import soundfile as sf
    from backend.tts.piper_tts import PiperTTS

    tts = {"en": PiperTTS(model_id=args.speak_en or SPEAKER_VOICE["en"]),
           "sk": PiperTTS(model_id=args.speak_sk or SPEAKER_VOICE["sk"])}

    if not args.no_warmup:
        # Throwaway turn so cold model loads don't tax turn 0 (same as the demo
        # runbook's pre-warm step). Results are discarded.
        warm_path = os.path.join(OUT_DIR, f"live_warmup_{TAG}en.wav")
        warm, wsr, _ = tts["en"].synthesize("Warmup sentence, please ignore.", language="en")
        import soundfile as sf_w
        sf_w.write(warm_path, warm, wsr)
        print("warmup: streaming throwaway turn …", flush=True)
        subprocess.run(
            [os.path.join(REPO_ROOT, ".venv", "bin", "python"),
             os.path.join(REPO_ROOT, "scripts", "live_direction_probe.py"),
             "--source", "en", "--target", "sk", "--clip", warm_path,
             "--tts", args.tts_sk, "--settle", "10", "--trail", "2.0"],
            capture_output=True, cwd=REPO_ROOT)
        print("warmup done (discarded)", flush=True)

    clock, timeline, mix_parts, chapters = 0.0, [], [], []
    for i, turn in enumerate(TURNS):
        src, tgt = turn["lang"], ("sk" if turn["lang"] == "en" else "en")
        speak_path = os.path.join(OUT_DIR, f"live_speak_{TAG}{i:02d}_{src}.wav")
        speech, sr, _ = tts[src].synthesize(turn["text"], language=src)
        sf.write(speak_path, speech, sr)
        speech_s = len(speech) / sr

        dump = os.path.join(OUT_DIR, f"live_dump_{TAG}{i:02d}_{src}-{tgt}.json")
        tts_raw = os.path.join(OUT_DIR, f"live_tts_{TAG}{i:02d}_{tgt}.raw.wav")
        tts_choice = args.tts_sk if tgt == "sk" else args.tts_en
        cmd = [os.path.join(REPO_ROOT, ".venv", "bin", "python"),
               os.path.join(REPO_ROOT, "scripts", "live_direction_probe.py"),
               "--source", src, "--target", tgt, "--clip", speak_path,
               "--tts", tts_choice, "--settle", "14", "--trail", "3.0",
               "--dump", dump, "--tts-out", tts_raw]
        print(f"turn {i} {src}->{tgt}: streaming {speech_s:.1f}s through /ws …", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        print(r.stdout[-1500:] if r.stdout else "", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"probe failed for turn {i}:\n{r.stderr[-2000:]}\n"
                               "Is `make run` serving wss://localhost:8000?")
        d = json.load(open(dump, encoding="utf-8"))

        # Decode concatenated WAV chunks -> one playback clip (tolerant: TTS may
        # legitimately be absent if the turn produced no translation).
        pcm_parts, prate = [], None
        blob = b""
        if os.path.isfile(tts_raw):
            with open(tts_raw, "rb") as f:
                blob = f.read()
        off = 0
        # WAV chunks each carry a 44-byte header (PCM16); split on 'RIFF'.
        import re
        starts = [m.start() for m in re.finditer(b"RIFF", blob)] or ([0] if blob else [])
        for k, s0 in enumerate(starts):
            s1 = starts[k + 1] if k + 1 < len(starts) else len(blob)
            try:
                w, rate = sf.read(io.BytesIO(blob[s0:s1]))
            except Exception:
                continue
            pcm_parts.append(np.asarray(w, dtype=np.float32))
            prate = rate
        if pcm_parts:
            playback = np.concatenate(pcm_parts)
            play_s = len(playback) / prate
        else:
            playback, prate, play_s = np.zeros(0, dtype=np.float32), 22050, 0.0
        play_path = os.path.join(OUT_DIR, f"live_turn_{TAG}{i:02d}_{src}-{tgt}.wav")
        sf.write(play_path, playback, prate)

        t0 = clock
        tr_at = [e["at"] for e in d["transcripts"]]
        tl_at = [e["at"] for e in d["translations"]]
        first_tts = d.get("first_tts_at")
        mt_first = (tl_at[0] - tr_at[0]) if tr_at and tl_at else None
        entry = {
            "n": i, "speaker": turn["speaker"], "direction": f"{src}->{tgt}", "live": True,
            "script": turn["text"],
            "transcript": " / ".join(e["text"] for e in d["transcripts"]),
            "translation": " / ".join(e["text"] for e in d["translations"]),
            "t_start_s": round(t0, 3), "speech_s": round(speech_s, 3),
            "utterances": len(tr_at),
            "mt_first_word_s": round(mt_first, 4) if mt_first is not None else None,
            "tts_first_chunk_s": (round(first_tts - tl_at[0], 4)
                                  if first_tts is not None and tl_at else None),
            "playback_start_s": round(t0 + first_tts, 3) if first_tts is not None else None,
            "playback_s": round(play_s, 3),
            "partials": len(d.get("partials", [])),
            "events": {"transcript_at": [round(t0 + t, 3) for t in tr_at],
                       "translation_at": [round(t0 + t, 3) for t in tl_at]},
            "playback_wav": os.path.relpath(play_path, REPO_ROOT),
            "speech_wav": os.path.relpath(speak_path, REPO_ROOT),
        }
        timeline.append(entry)
        mix_parts += [(t0, np.asarray(speech, dtype=np.float32), sr, "speech")]
        if len(playback):
            mix_parts.append((t0 + (first_tts or 0.0), playback, prate, "playback"))
        chapters.append({"turn": i, "speaker": turn["speaker"], "direction": entry["direction"],
                         "speech_at": round(t0, 2),
                         "playback_at": round(t0 + (first_tts or 0.0), 2)})
        last_end = max([t0 + speech_s] + [t0 + t for t in tl_at] +
                       ([t0 + first_tts + play_s] if first_tts is not None else []))
        clock = last_end + args.gap
        print(f"  -> utterances={len(tr_at)} partials={len(d.get('partials', []))} "
              f"first_tts={first_tts}s playback={play_s:.1f}s", flush=True)

    mix_sr = 22050
    total_n = int(round(clock * mix_sr)) + mix_sr
    mix = np.zeros(total_n, dtype=np.float32)
    for at, wav, wsr, _label in mix_parts:
        import librosa
        if wsr != mix_sr:
            wav = np.asarray(librosa.resample(wav, orig_sr=wsr, target_sr=mix_sr), dtype=np.float32)
        n0 = int(round(at * mix_sr))
        n1 = min(n0 + len(wav), total_n)
        if n1 > n0:
            mix[n0:n1] += wav[:n1 - n0]
    mix_path = os.path.join(OUT_DIR, "meeting_mix.wav")
    sf.write(mix_path, mix, mix_sr)
    with open(os.path.join(OUT_DIR, "meeting_timeline.json"), "w", encoding="utf-8") as f:
        json.dump({"gap_s": args.gap, "total_s": round(clock, 3), "turns": timeline,
                   "timing_model": "live-ws-measured", "tts_choice": {"sk": args.tts_sk, "en": args.tts_en},
                   "assumptions": {"note": "All event times measured on the real /ws pipeline; "
                                           "turns aligned sequentially with a fixed gap"},
                   "mix_wav": os.path.relpath(mix_path, REPO_ROOT),
                   "chapters": chapters}, f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.relpath(mix_path, REPO_ROOT)} ({clock:.1f}s total, live)")


if __name__ == "__main__":
    main()
