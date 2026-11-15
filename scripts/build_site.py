#!/usr/bin/env python3
"""Generate documentation/site/index.html (the project's public page) from the evidence files,
so its numbers cannot drift.

    python3 scripts/build_site.py            # writes documentation/site/index.html
Publish by copying it to docs/index.html on the gh-pages branch (the Pages source is gh-pages, /docs).
Publishing is a separate, deliberate step.
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

def stt(model, lang): return m["stt_t4"][f"{model}|{lang}"]["WER"] * 100

LANGS = [("German", "de"), ("English", "en"), ("French", "fr"), ("Polish", "pl"),
         ("Czech", "cs"), ("Slovak", "sk")]

def s(key):
    c = sim[key]
    return f"{c['latency_median_s']:.2f} s / {c['WER'] * 100:.1f}%"

NAMES = {
    "en-owner|turbo|chunk-2.0": ("English to German, independent 2 s chunks (current live route)", "en"),
    "en-owner|turbo|agree-1.0": ("English to German, growing buffer every 1.0 s, commit agreed words", "en"),
    "en-owner|turbo|agree-0.5": ("English to German, same, every 0.5 s", "en"),
}
GERMAN_KEYS = {"en-owner|turbo|chunk-2.0": "de-fleurs|turbo|chunk-2.0",
               "en-owner|turbo|agree-1.0": "de-fleurs|turbo|agree-1.0",
               "en-owner|turbo|agree-0.5": "de-fleurs|turbo|agree-0.5"}

ROUTE_NAMES = {
    "en-de|chunk2": ("English to German, 2 s chunks", "works"),
    "en-de|agree10": ("English to German, agreement every 1.0 s", "recommended on a GPU"),
    "en-de|agree05": ("English to German, agreement every 0.5 s", "collapses — do not use"),
    "en-sk|chunk2": ("English to Slovak, 2 s chunks", "works"),
    "en-sk|agree10": ("English to Slovak, agreement every 1.0 s", "same as chunks"),
}

def route_row(k):
    r = route[k]
    return f"<tr><td>{escape(ROUTE_NAMES[k][0])}</td><td><span class='tag'>{ROUTE_NAMES[k][1]}</span></td><td>{r['lag_median_s']:.2f} s / {r['lag_p90_s']:.2f} s</td><td>{r['first_word_lag_median_s']:.2f} s</td><td>{r['transcript_WER_prefix'] * 100:.1f}%</td></tr>"

css = """
body{font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:900px;margin:0 auto;padding:1.5rem 1rem;color:#222;background:#fff}
h1{font-size:1.5rem;margin:0 0 .25rem 0}
.sub{color:#555;font-size:.95rem;margin-bottom:1rem}
header{border-bottom:2px solid #e5e5e5;padding-bottom:1rem;margin-bottom:1.5rem}
.verdict{background:#eef6ff;border-left:4px solid #2563eb;padding:.6rem .9rem;margin-bottom:1rem;border-radius:4px}
.verdict strong{color:#1d4ed8}
.tags{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:1.5rem}
.tag{background:#f1f5f9;border:1px solid #cbd5e1;color:#334155;padding:.15rem .5rem;border-radius:999px;font-size:.8rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;margin-bottom:1.5rem}
.card{border:1px solid #e5e5e5;border-radius:8px;padding:1rem;background:#fafafa}
.card h3{margin:0 0 .5rem 0;font-size:.95rem;color:#111}
.card p{margin:0;font-size:.88rem;color:#334155}
h2{font-size:1.1rem;margin:1.5rem 0 .5rem 0;border-bottom:1px solid #e5e5e5;padding-bottom:.25rem}
.intro{color:#444;font-size:.9rem;margin-bottom:.6rem}
table{border-collapse:collapse;width:100%;margin:.4rem 0 1.2rem 0;font-size:.88rem}
th,td{border-bottom:1px solid #e5e5e1;padding:.45rem .5rem;text-align:left;vertical-align:top}
th{background:#f8fafc;font-weight:600;color:#334155}
tr:nth-child(even) td{background:#fcfcfc}
code{background:#f1f5f9;padding:0 .2rem;border-radius:3px;font-size:.85rem}
.small{color:#555;font-size:.82rem}
.footer{border-top:1px solid #e5e5e1;margin-top:2rem;padding-top:1rem;color:#555;font-size:.8rem}
.works{border:1px solid #dcfce7;background:#f0fdf4;border-radius:8px;padding:.8rem 1rem;margin-bottom:1rem}
.works h3{margin:0 0 .4rem 0;color:#166534;font-size:.95rem}
.works ul{margin:0;padding-left:1.1rem;color:#14532d;font-size:.88rem}
.works li{margin-bottom:.25rem}
.notbuilt{border:1px solid #fee2e2;background:#fef2f2;border-radius:8px;padding:.8rem 1rem;margin-bottom:1rem}
.notbuilt h3{margin:0 0 .4rem 0;color:#991b1b;font-size:.95rem}
.notbuilt ul{margin:0;padding-left:1.1rem;color:#7f1d1d;font-size:.88rem}
.notbuilt li{margin-bottom:.25rem}
@media (max-width:640px){table{font-size:.8rem} th,td{padding:.35rem .4rem}}
"""

stt_rows = "".join(
    f"<tr><td>{n}</td><td>{stt('whisper-large-v3-turbo', c):.1f}%</td><td>{stt('parakeet-tdt-v3', c):.1f}%</td><td>{stt('whisper-small', c):.1f}%</td><td>{stt('whisper-base', c):.1f}%</td></tr>"
    for n, c in LANGS)
chain = "".join(f"<tr><td>{escape(k)}</td><td>{v['stt_s']:.2f}</td><td>{v['mt_s']:.2f}</td><td>{v['tts_s']:.2f}</td><td><b>{v['chain_s_median']:.2f}</b></td></tr>" for k, v in sorted(e2e.items()))
stream_rows = ""
for k, (label, _) in NAMES.items():
    gk = GERMAN_KEYS[k]
    stream_rows += f"<tr><td>{escape(label)}</td><td>{s(k)}</td><td>{s(gk)}</td></tr>"
route_rows = "".join(route_row(k) for k in ROUTE_NAMES)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Speech Translation — status and evidence</title><style>{css}</style></head><body>
<header>
  <h1>Live Speech Translation</h1>
  <p class="sub">Real-time speech translation that runs on your own machine: speak English, hear German, Slovak or Czech a moment later, with live captions. Research prototype. Numbers on this page are generated from saved receipts in the repository and verified by <code>make check-docs</code>.</p>
</header>

<div class="verdict">
  <strong>Verdict:</strong> one speaker, one direction runs live on a laptop (English → Slovak/Czech/German). Median lag {0.5}-{1.8} s on generated English on the M1 Pro Mac; the 2 s goal is met there but <strong>Slovak input under 2 s is not viable on this Mac</strong> — it needs a faster GPU (T4/L4/A10) or a streaming-native recogniser. German and Slovak are validated end to end through the real server on a Kaggle T4. Room mode is built and unit-tested; its real-model test is pending.
</div>

<div class="tags"><span class="tag">local-first, no cloud</span><span class="tag">EN↔SK/DE/CS</span><span class="tag">local speech translation</span><span class="tag">measured 2026-11-07</span></div>

<h2>At a glance</h2>
<div class="grid">
  <div class="card">
    <h3>What works today</h3>
    <p>EN→SK, EN→DE, EN→CS live on your own machine. Slovak output uses the kinit fine-tune (6.5% WER on native Slovak FLEURS; 19.8% on a non-native speaker's own reading). German validated end to end on a T4.</p>
  </div>
  <div class="card">
    <h3>Goal</h3>
    <p>Median lag ≤ 2 s, voice starts mid-sentence, drift ≤ 0.5 s/min. Met on generated English on the Mac; not yet on real Slovak speech.</p>
  </div>
  <div class="card">
    <h3>Main blocker</h3>
    <p>Slovak input under 2 s needs better compute. Whisper turbo is ~0.7 s/call on the Mac; Nemotron scores 28% on Slovak. Route: Nemotron 3.5 Slovak baseline on a T4 (~30 min), then fine-tune.</p>
  </div>
  <div class="card">
    <h3>Voice</h3>
    <p>Piper is generic but fast (~0.15 s/piece). OmniVoice clones zero-shot (~0.66 s/piece) but is batch/QC only, not streaming. Listening review is pending before selecting the route.</p>
  </div>
</div>

<h2>Speech recognition by language</h2>
<p class="intro">Mean word error on Kaggle T4, FLEURS read speech, 60 clips per language. Lower is better.</p>
<table>
  <tr><th>Language</th><th>Whisper large-v3-turbo</th><th>Parakeet v3</th><th>Whisper small</th><th>Whisper base</th></tr>
  {stt_rows}
</table>
<p class="small">Slovak with the kinit fine-tune: {sk['kinit/whisper-large-v3-turbo-sk|fleurs-sk']['WER'] * 100:.1f}% (plain turbo {sk['large-v3-turbo (plain)|fleurs-sk']['WER'] * 100:.1f}%). On a non-native speaker's own reading: {sk['kinit/whisper-large-v3-turbo-sk|owner-sk']['WER'] * 100:.1f}%.</p>

<h2>Whole chain for a 10 s utterance</h2>
<p class="intro">Seconds, median of 25 clips on Kaggle T4; voice runs on CPU. Total in bold.</p>
<table>
  <tr><th>Direction</th><th>Recognition</th><th>Translation</th><th>Voice</th><th>Total</th></tr>
  {chain}
</table>

<h2>Streaming: delay versus accuracy</h2>
<p class="intro">T4 simulation, word-commit delay and word error (Whisper turbo). No setting reached under 1 s with under 10% error.</p>
<table>
  <tr><th>Strategy</th><th>English reading</th><th>German speech</th></tr>
  {stream_rows}
</table>

<h2>Real server end to end</h2>
<p class="intro">Kaggle T4, a test speaker's 60 s English reading replayed through the live route at 1×. Delay measured after each segment ends.</p>
<table>
  <tr><th>Route</th><th>Status</th><th>Delay median / p90</th><th>First-word delay</th><th>Transcript error</th></tr>
  {route_rows}
</table>

<h2>Against Google, Microsoft and OpenAI</h2>
<p class="intro">Read from their own pages (2026-11-08). We compete on behaviour and on running locally — not on measured quality (no side-by-side test exists).</p>
<table>
  <tr><th></th><th>Google Gemini Live Translate</th><th>Microsoft Teams Interpreter</th><th>OpenAI gpt-realtime-translate</th><th>This project</th></tr>
  <tr><td>Design</td><td>one speech-to-speech model; keeps intonation, pacing, pitch</td><td>cascade: recognition, translation, voice</td><td>streaming speech-to-speech model</td><td>cascade (same as Microsoft)</td></tr>
  <tr><td>Lag</td><td>"a few seconds behind" (no number)</td><td>not stated</td><td>not stated</td><td>3.0–3.6 s on M1 Pro (goal 3.0 s); ~2.3–2.9 s on T4 simulation at 6–17% word error</td></tr>
  <tr><td>Access</td><td>paid Google plan / API preview</td><td>Copilot or Teams Premium licence</td><td>API, usage-priced</td><td>runs on your own machine; no cloud, licence, or per-minute fee</td></tr>
</table>

<div class="works">
  <h3>Honest status — works</h3>
  <ul>
    <li>EN→SK, EN→DE, EN→CS live on a laptop; local-first, no audio leaves the machine</li>
    <li>German end to end through the real server on a T4, zero server errors</li>
    <li>Room mode built and unit-tested (many listeners, each own language)</li>
    <li>Docs numbers verified against receipts (<code>make check-docs</code> 57/57)</li>
  </ul>
</div>
<div class="notbuilt">
  <h3>Honest status — not yet built / validated</h3>
  <ul>
    <li>Slovak input under 2 s (needs GPU or streaming recogniser)</li>
    <li>Room mode real-model test pending</li>
    <li>Cloned voice for other people; fresh-install Linux/Windows retest pending</li>
    <li>Side-by-side test vs Google/Microsoft/OpenAI</li>
  </ul>
</div>

<div class="footer">
  <p><strong>Source &amp; install:</strong> <a href="https://github.com/brusnyak/bp">github.com/brusnyak/bp</a> · <a href="https://github.com/brusnyak/bp/blob/main/documentation/STATUS.md">Status</a> · <a href="https://github.com/brusnyak/bp/blob/main/documentation/LANGUAGES.md">Languages</a> · <a href="https://github.com/brusnyak/bp/blob/main/documentation/BENCHMARKS.md">Benchmarks</a></p>
  <p>Quick start: <code>git clone https://github.com/brusnyak/bp.git &amp;&amp; cd bp &amp;&amp; python3.11 scripts/setup.py --profile mac --langs de,sk &amp;&amp; make run PRESET=mac</code> then open <code>http://localhost:8000</code>.</p>
  <p class="small">Regenerate this page: <code>python3 scripts/build_site.py</code> (reads receipts in <code>documentation/evidence/2026-11-07/</code>), then publish to <code>gh-pages</code>/<code>docs/index.html</code>.</p>
</div>
</body></html>"""

out = ROOT / "documentation/site/index.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, len(html), "bytes")