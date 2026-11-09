#!/usr/bin/env python3
"""Generate clean reference speech from the reading script with the generic Piper voices (no owner voice), so segmentation strategies can be
tested without anyone reading aloud. Writes processed/testaudio/<lang>_<scenario>.wav (16 kHz) and a .json with the exact reference text.

  PYTHONPATH=. .venv/bin/python scripts/live/make_test_audio.py [--langs sk en de]

Scenarios (gap = silence between sentences):
  paced       lines A1-A8, 0.9 s gaps      a speaker who pauses between sentences (the demo cadence)
  continuous  lines A1-A8, 0.25 s gaps     reading without pauses (what a live reader actually does)
  long        script line C1 pause marks as 0.5 s gaps inside long sentences, numbers/names lines B1-B3 at 0.4 s gaps
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
VOICE = {"sk": "sk_SK-lili-medium", "en": "en_US-john-medium", "de": "de_DE-thorsten-medium"}
SR = 16000


def lines(lang):
    from score_reading import reference
    return reference(lang)   # {"A": [...], "B": [...], "C": [...]}; "/" pause marks already turned into spaces


def raw_c1(lang):
    head = {"sk": "## Slovak (SK)", "en": "## English (EN)", "de": "## German (DE)"}[lang]
    t = (ROOT / "documentation" / "reading_script_multilingual.md").read_text(encoding="utf-8")
    block = t[t.index(head):]; nxt = [block.find(h, 5) for h in ("## Slovak", "## English", "## German") if block.find(h, 5) > 0]
    block = block[: min(nxt)] if nxt else block
    m = re.search(r"^C1\.\s+(.*)$", block, re.M)
    return m.group(1) if m else ""


def scenario(lang, name):
    L = lines(lang)
    if name == "paced": return [(s, 0.9) for s in L["A"][:8]]
    if name == "continuous": return [(s, 0.25) for s in L["A"][:8]]
    if name == "long":
        out = []
        for part in re.split(r"\s*//\s*", raw_c1(lang)):
            for sub in re.split(r"\s*/\s*", part): out.append((sub.strip(), 0.5))
        return out + [(s, 0.4) for s in L["B"][:3]]
    raise SystemExit(name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--langs", nargs="+", default=["sk", "en", "de"]); ap.add_argument("--scenarios", nargs="+", default=["paced", "continuous", "long"])
    a = ap.parse_args(); import soundfile as sf, scipy.signal as ss
    from backend.tts.piper_tts import PiperTTS
    out = ROOT / "processed" / "testaudio"; out.mkdir(parents=True, exist_ok=True)
    for lang in a.langs:
        tts = PiperTTS(model_id=VOICE[lang])
        for sc in a.scenarios:
            parts, text, times, t = [], [], [], 0.0
            for sent, gap in scenario(lang, sc):
                if not sent: continue
                w, sr, _ = tts.synthesize(sent, language=lang); w = np.asarray(w, dtype=np.float32)
                if int(sr) != SR: w = ss.resample_poly(w, SR, int(sr)).astype(np.float32)
                times.append({"text": sent, "start": round(t, 2), "end": round(t + len(w) / SR, 2)}); t += len(w) / SR + gap
                parts += [w, np.zeros(int(SR * gap), dtype=np.float32)]; text.append(sent)
            audio = np.concatenate(parts); sf.write(out / f"{lang}_{sc}.wav", audio, SR)
            (out / f"{lang}_{sc}.json").write_text(json.dumps({"lang": lang, "scenario": sc, "voice": VOICE[lang], "seconds": round(len(audio) / SR, 1), "text": " ".join(text), "sentences": times}, ensure_ascii=False, indent=1))
            print(f"{lang}_{sc}: {len(audio) / SR:.1f} s, {len(text)} sentences", flush=True)


if __name__ == "__main__":
    main()
