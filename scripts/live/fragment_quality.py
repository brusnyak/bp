#!/usr/bin/env python3
"""Does cutting speech into short segments/fragments damage the translation?
Takes paced_replay JSONs, joins the source texts of all fragments and translates that ONE text in a
single MT call (the no-fragmentation reference for the same STT words), then scores the fragment
translations joined together against it with chrF++. 100 = identical to unfragmented MT of the same
words. Isolates fragmentation damage from STT errors; says nothing about absolute translation quality.
usage: fragment_quality.py <ct2 model dir> <src> <tgt> processed/paced_a.json ..."""
import json, sys
import sacrebleu
sys.path.insert(0, ".")
from backend.mt.ctranslate2_mt import CTranslate2MT

model, src, tgt, *files = sys.argv[1:]
mt = CTranslate2MT(model_path=model, device="cpu")
for f in files:
    ev = json.load(open(f))["events"]
    source = " ".join(e["text"].strip() for e in ev if e["type"] == "transcription_result" and e.get("text", "").strip())
    frag = " ".join(e["text"].strip() for e in ev if e["type"] == "translation_result" and e.get("text", "").strip())
    ref, _ = mt.translate(source, src, tgt)
    score = sacrebleu.corpus_chrf([frag], [[ref]], word_order=2).score
    print(f"{f}: chrF++ vs unfragmented MT = {score:.1f}  (source words {len(source.split())}, fragments {sum(e['type']=='translation_result' for e in ev)})")
