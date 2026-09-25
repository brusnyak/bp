#!/usr/bin/env python3
"""Smart recording script: EN originals (project-owned, no manual writing needed for SK)
translated with the project's OWN local MT (Helsinki Opus-MT en-sk, CTranslate2 int8).

Run:  venv/bin/python scripts/make_reading_script.py
Out:  documentation/reading_script_bilingual.md  (numbered EN + SK, print-and-read)

Human step after: read the SK side aloud, fix any awkward machine phrasing BEFORE
recording (a corrected script beats a literal one — you are training YOUR voice,
not the translator). Then record per documentation/voice_and_app_direction_2026-09.md §3.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Original EN sentences: phonetically varied, declarative, TTS-training friendly.
EN_SENTENCES = [
    "Good morning, and welcome to this live demonstration of real time speech translation.",
    "I am speaking English, and you will hear my own voice speaking Slovak within seconds.",
    "The system listens through the microphone and detects when speech begins and ends.",
    "First the speech is transcribed into text, then translated, then spoken again.",
    "Every step runs locally on this computer, with no cloud and no waiting room.",
    "Latency is the time between my words and the translated voice you hear.",
    "Our goal is a delay of only a few seconds, short enough for natural conversation.",
    "Voice cloning means the Slovak output sounds like me, not like a stranger.",
    "To teach the system my voice, I record a few minutes of clean speech.",
    "A quiet room matters more than an expensive microphone.",
    "Please speak at a steady pace, with short pauses between sentences.",
    "Numbers, names, and technical terms need extra care in both languages.",
    "The subtitle line shows partial results while I am still speaking.",
    "You can pop the subtitles out into a floating window above any application.",
    "The translated voice is routed into the meeting as a virtual microphone.",
    "Other participants hear Slovak, while I keep speaking English.",
    "Questions from the audience will be translated in the opposite direction.",
    "Thank you for listening, and I look forward to your questions.",
]

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "documentation",
    "reading_script_bilingual.md",
)


def main():
    from backend.mt.ctranslate2_mt import CTranslate2MT

    mt = CTranslate2MT("Helsinki-NLP/opus-mt-en-sk")
    lines = [
        "# Bilingual recording script (EN originals + local-MT SK draft)",
        "",
        "Generated with the project's own Opus-MT en-sk model — no external service.",
        "**Proofread the SK column before recording.** Fix anything that sounds machine-made;",
        "the recording teaches YOUR voice, the text must be worth saying.",
        "",
        "| # | EN (read for the EN session) | SK (correct, then read for the SK session) |",
        "| --- | --- | --- |",
    ]
    for i, en in enumerate(EN_SENTENCES, 1):
        sk, _lat = mt.translate(en, "en", "sk")
        lines.append(f"| {i} | {en} | {sk.strip()} |")
    lines += [
        "",
        "Recording spec: 2–3 min per language, quiet room, steady pace, short pauses.",
        "Stage files in the Voice Lab upload box first, listen, then register.",
    ]
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT} ({len(EN_SENTENCES)} pairs)")


if __name__ == "__main__":
    main()
