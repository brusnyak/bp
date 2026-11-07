#!/usr/bin/env python3
"""Generate documentation/site/index.html (the project's public page) from the evidence files, so its numbers cannot drift.

    python3 scripts/build_site.py            # writes documentation/site/index.html
Publish by copying it to docs/index.html on the gh-pages branch (the Pages source is gh-pages, /docs). Publishing is a separate, deliberate step.
"""
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "documentation/evidence/2026-11-07"
m = json.loads((EV / "lang_matrix.json").read_text())
e2e = json.loads((EV / "e2e_chain.json").read_text())["chain"]
sim = json.loads((EV / "stream_sim.json").read_text())["cells"]
sk = json.loads((EV / "sk_compare.json").read_text())["cells"]
route = json.loads((EV / "route_test.json").read_text())["runs"]

def w(model, lang): return m["stt_t4"][f"{model}|{lang}"]["WER"] * 100
LANGS = [("German", "de"), ("English", "en"), ("French", "fr"), ("Polish", "pl"), ("Czech", "cs"), ("Slovak", "sk")]
rows = "".join(
    f"<tr><td>{n}</td><td>{w('whisper-large-v3-turbo', c):.1f}%</td><td>{w('parakeet-tdt-v3', c):.1f}%</td><td>{w('whisper-small', c):.1f}%</td><td>{w('whisper-base', c):.1f}%</td></tr>"
    for n, c in LANGS)
chain = "".join(f"<tr><td>{escape(k)}</td><td>{v['stt_s']:.2f}</td><td>{v['mt_s']:.2f}</td><td>{v['tts_s']:.2f}</td><td><b>{v['chain_s_median']:.2f}</b></td></tr>" for k, v in sorted(e2e.items()))
def s(k): c = sim[k]; return f"{c['latency_median_s']:.2f} s, {c['WER'] * 100:.1f}%"
NAMES = {"en-de|chunk2": "English to German, 2 s chunks", "en-de|agree10": "English to German, agreement every 1.0 s (recommended on a GPU)", "en-de|agree05": "English to German, agreement every 0.5 s (collapses)",
         "en-sk|chunk2": "English to Slovak, 2 s chunks", "en-sk|agree10": "English to Slovak, agreement every 1.0 s"}
route_rows = "".join(f"<tr><td>{NAMES[k]}</td><td>{route[k]['lag_median_s']:.2f} s / {route[k]['lag_p90_s']:.2f} s</td><td>{route[k]['first_word_lag_median_s']:.2f} s</td><td>{route[k]['transcript_WER_prefix'] * 100:.1f}%</td></tr>" for k in NAMES if k in route)
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Speech Translation: status and evidence</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#111;background:#fff}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}td,th{{border-bottom:1px solid #ddd;padding:.35rem .5rem;text-align:left}}th{{font-weight:600}}
code{{background:#f2f2f2;padding:0 .25rem}}small{{color:#555}}@media(prefers-color-scheme:dark){{body{{background:#111;color:#eee}}td,th{{border-color:#333}}code{{background:#222}}small{{color:#aaa}}}}</style></head><body>
<h1>Live Speech Translation</h1>
<p>Real-time speech translation that runs on your own machine: speech recognition, translation and a synthetic voice, with live captions. A bachelor's thesis project (STU FEI). Measured 2026-11-07; the numbers on this page are generated from the saved receipts in the repository.</p>
<p><a href="https://github.com/brusnyak/bp">Source and install</a> · <a href="https://github.com/brusnyak/bp/blob/main/documentation/STATUS.md">Status</a> · <a href="https://github.com/brusnyak/bp/blob/main/documentation/LANGUAGES.md">Languages</a> · <a href="https://github.com/brusnyak/bp/blob/main/documentation/BENCHMARKS.md">Benchmarks</a></p>
<h2>Speech recognition by language <small>(mean word error, Kaggle T4, FLEURS read speech, 60 clips each)</small></h2>
<table><tr><th>Language</th><th>Whisper large-v3-turbo</th><th>Parakeet v3</th><th>Whisper small</th><th>Whisper base</th></tr>{rows}</table>
<p>Slovak with the kinit fine-tune: {sk['kinit/whisper-large-v3-turbo-sk|fleurs-sk']['WER'] * 100:.1f}% (plain turbo {sk['large-v3-turbo (plain)|fleurs-sk']['WER'] * 100:.1f}%). On a non-native speaker's own reading: {sk['kinit/whisper-large-v3-turbo-sk|owner-sk']['WER'] * 100:.1f}%.</p>
<h2>Whole chain for a 10 s utterance <small>(seconds, median of 25 clips, T4; the voice runs on the CPU)</small></h2>
<table><tr><th>Direction</th><th>Recognition</th><th>Translation</th><th>Voice</th><th>Total</th></tr>{chain}</table>
<h2>Streaming: delay versus accuracy <small>(T4 simulation, word-commit delay and word error)</small></h2>
<table><tr><th>Strategy (Whisper turbo)</th><th>English reading</th><th>German speech</th></tr>
<tr><td>Independent 2 s chunks (current live route)</td><td>{s('en-owner|turbo|chunk-2.0')}</td><td>{s('de-fleurs|turbo|chunk-2.0')}</td></tr>
<tr><td>Growing buffer every 1.0 s, commit agreed words</td><td>{s('en-owner|turbo|agree-1.0')}</td><td>{s('de-fleurs|turbo|agree-1.0')}</td></tr>
<tr><td>Same, every 0.5 s</td><td>{s('en-owner|turbo|agree-0.5')}</td><td>{s('de-fleurs|turbo|agree-0.5')}</td></tr></table>
<h2>Real server end to end <small>(Kaggle T4, your 60 s English reading replayed through the live route)</small></h2>
<table><tr><th>Route</th><th>Delay median / p90</th><th>First-word delay</th><th>Transcript error</th></tr>{route_rows}</table>
<p><b>Honest status.</b> One speaker, one direction runs live on a laptop; German and Slovak are validated through the real server on a GPU. The room mode (many listeners, each with an earpiece and a language) is designed, not built. See the status page for the full list.</p>
</body></html>"""
out = ROOT / "documentation/site/index.html"; out.write_text(html, encoding="utf-8"); print("wrote", out, len(html), "bytes")
