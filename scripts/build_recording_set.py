#!/usr/bin/env python3
"""Build a Slovak reading script with English references for recording an evaluation/demo set.

    python scripts/build_recording_set.py            # writes eval_data/recording_set_v2/manifest.json
                                                     #        documentation/recording_script_sk_v2.md
Then record it:  python scripts/record_reading.py

Sentences: 15 conference phrases + 60 human-written news sentences from NTREX-128 (Microsoft; parallel
Slovak/English, downloaded from GitHub on first use). News sentences are filtered to be easy to read aloud:
7-14 words, no digits, quotes or brackets, no mid-sentence capitals (names/acronyms).
"""
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NTREX = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/"
N_NEWS = 60

# (Slovak, English reference) - short natural conference talk. Fix anything that does not sound natural to you
# BEFORE recording and keep the English in sync.
CONFERENCE = [
    ("Dobrý deň, vitajte na našej konferencii.", "Good afternoon, welcome to our conference."),
    ("Môžete sa, prosím, predstaviť?", "Could you introduce yourself, please?"),
    ("Počujete ma dobre, alebo mám hovoriť hlasnejšie?", "Can you hear me well, or should I speak louder?"),
    ("Ďakujem za otázku, dovoľte mi na ňu odpovedať.", "Thank you for the question, let me answer it."),
    ("Ukážem vám, ako funguje živý preklad reči.", "I will show you how live speech translation works."),
    ("Preklad sa zobrazí na obrazovke a zároveň ho počujete.", "The translation appears on the screen and you hear it at the same time."),
    ("Zdržanie medzi mojou vetou a prekladom je niekoľko sekúnd.", "The delay between my sentence and the translation is a few seconds."),
    ("Môžeme prejsť k ďalšiemu bodu programu?", "Can we move on to the next item on the agenda?"),
    ("Naše výsledky sme zhrnuli v krátkej správe.", "We summarized our results in a short report."),
    ("Rozumiem vašej otázke, ale potrebujem viac času.", "I understand your question, but I need more time."),
    ("Prestávka bude trvať pätnásť minút.", "The break will last fifteen minutes."),
    ("Prosím, vypnite si mikrofóny, keď nehovoríte.", "Please turn off your microphones when you are not speaking."),
    ("Ďakujem vám za pozornosť a teším sa na diskusiu.", "Thank you for your attention and I look forward to the discussion."),
    ("Tento projekt sa zaoberá prekladom reči v reálnom čase.", "This project deals with real-time speech translation."),
    ("S tým súhlasím, ale mám k tomu jednu poznámku.", "I agree with that, but I have one comment."),
]


def fetch(name):
    cache = ROOT / "eval_data" / "ntrex" / name
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NTREX + name, cache)
    return [l.rstrip("\n") for l in cache.read_text(encoding="utf-8").splitlines()]


def readable(s):
    words = s.split()
    if not 7 <= len(words) <= 14 or re.search(r"[\d()\[\]\"'„“”‚‘’:;/%&@#]", s):
        return False
    return not any(w[:1].isupper() for w in words[1:])  # no names / acronyms mid-sentence


def build():
    sk = fetch("newstest2019-ref.slk.txt"); en = fetch("newstest2019-src.eng.txt")
    pool = [(a, b) for a, b in zip(sk, en) if readable(a)]
    step = max(len(pool) // N_NEWS, 1)
    news = pool[::step][:N_NEWS]
    items = [{"id": i, "kind": "conference" if i < len(CONFERENCE) else "news", "sk": a, "en": b}
             for i, (a, b) in enumerate(CONFERENCE + news)]
    return items


def main():
    items = build()
    out = ROOT / "eval_data" / "recording_set_v2"; out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Slovak recording script v2", "",
             f"{len(items)} sentences ({len(CONFERENCE)} conference phrases + {len(items) - len(CONFERENCE)} news sentences). ",
             "Record with `python scripts/record_reading.py` (one file per sentence, saved as `sk_00.wav` ...).", "",
             "## How to record", "",
             "- Quiet room, mic 15-20 cm from your mouth, no music/fan. Same position for the whole session.",
             "- Speak at your **normal conversational pace**, not slower and not \"announcer\" style. One natural take per sentence.",
             "- Read what is written. If you stumble or say a different word, redo the sentence (the recorder has `r`).",
             "- Leave about half a second of silence before and after each sentence.",
             "- Do not repeat a sentence you already read well; avoid over-articulating word endings.",
             "- If a phrase feels unnatural Slovak to you, tell me and I will replace it before you record it.", "",
             "| # | Read this (Slovak) | English reference |", "| --- | --- | --- |"]
    lines += [f"| {it['id']:02d} | {it['sk']} | {it['en']} |" for it in items]
    doc = ROOT / "documentation" / "recording_script_sk_v2.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(items)} sentences -> {out / 'manifest.json'} and {doc}")


if __name__ == "__main__":
    main()
