#!/usr/bin/env python3
"""Bulk HQ generation: OmniVoice zero-shot clones of the owner's v2b SK voice.

15 varied ~30s Slovak passages (meeting-style + transcript continuations) for the
future Piper training corpus. Slow one-time job: run ALONE in the background
(.venv-omni, MPS) — never alongside STT tests (16GB RAM box).

Out: processed/bulk_hq/hq_XX.wav + manifest.json
Run: .venv-omni/bin/python scripts/bulk_hq.py [--count N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "processed", "bulk_hq")
SK_REF = os.path.join(REPO_ROOT, "processed", "omnivoice", "ref_sk_trhove_head.wav")
SK_REF_TEXT = ("Včera ráno som išiel na trh kúpiť čerstvý chlieb a mlieko. "
               "Stretol som tam starého priateľa Ľuba, ktorý predával med a syry.")

# Meeting-style SK continuations (owner's register, new sentences for corpus variety).
NEW = [
    "Poďme si to zhrnúť. Prvý bod je hotový a druhý si necháme na budúci týždeň. Ak bude treba, zavolám ešte dnes poobede a dohodneme podrobnosti.",
    "Technika dnes poslúcha celkom dobre. Obraz je ostrý a zvuk čistý, takže sa môžeme venovať obsahu. Začnem krátkym úvodom a potom otvorím diskusiu.",
    "Cestou sem som stretol suseda a chvíľu sme sa rozprávali o záhrade. Sľúbil, že na jar prinesie sadenice paradajok a papriky.",
    "Káva je už hotová a vonia po celej kuchyni. Sadneme si k stolu a prejdeme dnešný program bod po bode, aby nám nič neuniklo.",
    "Vlak meškal skoro dvadsať minút, takže som si na stanici kúpil noviny a rožok. Nakoniec som prišiel včas a stihol aj rannú poradu.",
    "Deti sa hrali na dvore s loptou a smiali sa na celé kolo. Prizeral som sa im z okna a spomenul som si na vlastné detstvo.",
    "Večer si pozriem správy a potom si prečítam zopár strán z novej knihy. Ráno vstanem skôr, aby som stihol prechádzku so psom.",
    "Na trhu mali dnes čerstvé jahody a voňavý chlieb priamo z pece. Kúpil som aj syr a maslo, lebo cez víkend čakáme návštevu.",
]


def build_passages(sv_text: str, count: int, skip_sents: int = 0) -> list[str]:
    import re
    sents = [s for s in re.split(r"(?<=[.?!])\s+", sv_text.strip()) if s]
    sents = sents[skip_sents:] + sents[:skip_sents]  # rotate for round 2+ variety
    passages = []
    # 1) transcript windows (~350 chars at sentence boundaries)
    buf = ""
    for s in sents:
        buf += (" " if buf else "") + s
        if len(buf) >= 330:
            passages.append(buf)
            buf = ""
    # 2) meeting-style sentences grouped to ~350 chars
    buf = ""
    for s in NEW:
        buf += (" " if buf else "") + s
        if len(buf) >= 330:
            passages.append(buf)
            buf = ""
    if buf:
        passages.append(buf)
    # 3) pad by cycling transcript windows with different offsets if short
    k = 0
    while len(passages) < count:
        w = " ".join(sents[k % len(sents):k % len(sents) + 4])
        if w not in passages:
            passages.append(w)
        k += 1
        if k > 200:  # pragma: no cover - safety
            break
    return passages[:count]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=15)
    ap.add_argument("--start", type=int, default=0,
                    help="first clip index (round 2 continues numbering, never overwrites)")
    ap.add_argument("--skip-sents", type=int, default=0,
                    help="rotate transcript sentences for fresh passage windows")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    import torch
    import soundfile as sf
    from omnivoice import OmniVoice

    meta = json.load(open(os.path.join(REPO_ROOT, "speaker_voices", "speaker_voices.json"),
                          encoding="utf-8"))
    sv_text = next(e["transcribed_text"] for e in meta if "sk_trhove_rano_v2b" in e.get("path", ""))
    passages = build_passages(sv_text, args.count, args.skip_sents)
    print(f"{len(passages)} passages, chars: {[len(p) for p in passages]}", flush=True)

    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="mps", dtype=torch.float32)
    print(f"load {time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(SK_REF, SK_REF_TEXT)
    print(f"prompt {time.perf_counter() - t0:.1f}s (reused for all clips)", flush=True)

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f).get("clips", [])
    except (OSError, ValueError):
        manifest = []
    manifest = [c for c in manifest if c.get("n", -1) < args.start]
    for j, text in enumerate(passages):
        n = args.start + j
        t0 = time.perf_counter()
        audio = model.generate(text=text, language="sk", voice_clone_prompt=prompt, num_step=16)[0]
        syn_s = time.perf_counter() - t0
        audio_s = len(audio) / model.sampling_rate
        path = os.path.join(OUT_DIR, f"hq_{n:02d}.wav")
        sf.write(path, audio, model.sampling_rate)
        manifest.append({"n": n, "chars": len(text), "text": text,
                         "syn_s": round(syn_s, 2), "audio_s": round(audio_s, 2),
                         "rtf": round(syn_s / audio_s, 3), "wav": path})
        print(f"hq_{n:02d}: {audio_s:.1f}s audio in {syn_s:.1f}s (rtf {syn_s / audio_s:.2f})", flush=True)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"reference": SK_REF, "reference_text": SK_REF_TEXT,
                   "device": "mps", "num_step": 16, "clips": manifest},
                  f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.relpath(OUT_DIR, REPO_ROOT)}/manifest.json")


if __name__ == "__main__":
    main()
