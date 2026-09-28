#!/usr/bin/env python3
"""Two-sided conversation demo through the real STT -> MT -> TTS backends, with a timing report.

Person A speaks English (Slovak is played to B); Person B answers in Slovak (English is played to A).
Each turn is timed per stage. Several Slovak recognizers can be compared in one self-contained HTML page
(tabs, SVG timeline chart, per-turn table with the translated audio) plus a JSON file.

    python scripts/demo_conversation.py                                   # synthetic speech, default recognizer
    python scripts/demo_conversation.py --sk-stt "turbo=large-v3-turbo,parakeet,small-sk=ct2_models/whisper-small-sk"
    python scripts/demo_conversation.py --inputs both --en-dir eval_data/en_clips --sk-dir eval_data/sk_clips

Recognizer spec: `parakeet` (NVIDIA Parakeet-TDT v3 via onnx-asr) or a faster-whisper model name / CT2 directory,
optionally prefixed `label=`. Output: processed/demo/conversation_demo.{html,json} (processed/ is git-ignored).
Only the translated (synthetic) audio is embedded in the report, never your input recordings.
"""
import argparse
import base64
import html
import io
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import soundfile as sf

# (speaker, text) - a natural exchange used to synthesize the input speech when no recordings are given
DIALOGUE = [
    ("A", "Good morning, can you hear me clearly?"),
    ("B", "Dobré ráno, počujem vás veľmi dobre."),
    ("A", "Great. Today I would like to show you how the live translation works."),
    ("B", "Výborne, som zvedavý, ako rýchlo to dokáže preložiť moju odpoveď."),
    ("A", "The delay between the end of my sentence and the translated voice is what we measure."),
    ("B", "Rozumiem. Ďakujem za ukážku a teším sa na ďalšie zlepšenia."),
]
GAP_S = 0.6  # pause between turns on the timeline
COLORS = {"speech": "#94a3b8", "stt": "#f59e0b", "mt": "#10b981", "tts": "#8b5cf6", "play": "#0ea5e9"}


def resample16k(audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def wav_b64(audio, sr):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


class Parakeet:
    """Slovak recognizer: NVIDIA Parakeet-TDT-0.6B-v3 via onnx-asr (int8, CPU, no PyTorch)."""
    def __init__(self):
        import onnx_asr
        self.m = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")

    def transcribe(self, wav, lang):
        t = time.perf_counter()
        return str(self.m.recognize(wav, sample_rate=16000)).strip(), time.perf_counter() - t


class Whisper:
    def __init__(self, size):
        from backend.stt.faster_whisper_stt import FasterWhisperSTT
        self.m = FasterWhisperSTT(model_size=size)

    def transcribe(self, wav, lang):
        segs, dt, _ = self.m.transcribe_audio(wav, 16000, language=lang)
        return " ".join(s.text for s in segs).strip(), dt


def make_stt(spec):
    label, _, s = spec.partition("=")
    if not s:
        label, s = spec, spec
    return label, (Parakeet() if s == "parakeet" else Whisper(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-dir"), ap.add_argument("--sk-dir")
    ap.add_argument("--inputs", choices=["synthetic", "recordings", "both"], default=None)
    ap.add_argument("--en-stt", default="base")
    local_sk = ROOT / "ct2_models" / "whisper-small-sk"  # same order as backend/main.py::_pick_stt_model
    default_sk = os.environ.get("BP_SK_STT_MODEL") or (str(local_sk) if (local_sk / "model.bin").exists() else "large-v3-turbo")
    ap.add_argument("--sk-stt", default=default_sk, help="comma separated recognizer specs")
    ap.add_argument("--out", default="processed/demo")
    ap.add_argument("--docs-dir", help="also write documentation assets here: audio-free HTML + standalone SVG charts")
    ap.add_argument("--from-json", help="skip the run; render --docs-dir from a saved conversation_demo.json")
    a = ap.parse_args()
    if a.from_json:
        export_docs(json.loads(Path(a.from_json).read_text(encoding="utf-8")), Path(a.docs_dir or "documentation/demo"))
        return
    modes = {"synthetic": ["synthetic"], "recordings": ["recordings"], "both": ["synthetic", "recordings"]}[
        a.inputs or ("recordings" if a.en_dir and a.sk_dir else "synthetic")]
    if "recordings" in modes and not (a.en_dir and a.sk_dir):
        sys.exit("--inputs recordings needs --en-dir and --sk-dir (files en_00.wav ... / sk_00.wav ...)")

    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.tts.piper_tts import PiperTTS

    t0 = time.perf_counter()
    en_label, stt_en = make_stt(a.en_stt)
    mt = {"en": CTranslate2MT("Helsinki-NLP/opus-mt-en-sk"), "sk": CTranslate2MT("Helsinki-NLP/opus-mt-sk-en")}
    tts = {"sk": PiperTTS(model_id="sk_SK-lili-medium"), "en": PiperTTS(model_id="en_US-ryan-medium")}  # spoken to B / to A
    sk_engines = [make_stt(s) for s in a.sk_stt.split(",")]
    for lang, eng in [("en", stt_en)] + [("sk", e) for _, e in sk_engines]:  # warm-up: first inference is much slower
        w, sr, _ = tts[lang].synthesize("Test." if lang == "en" else "Skúška.", language=lang)
        eng.transcribe(resample16k(w, sr), lang)
    load_s = time.perf_counter() - t0

    runs = []
    for mode in modes:
        for label, stt_sk in sk_engines:
            counters = {"A": 0, "B": 0}
            turns, clock = [], 0.0
            for speaker, text in DIALOGUE:
                src, tgt = ("en", "sk") if speaker == "A" else ("sk", "en")
                if mode == "recordings":
                    i = counters[speaker]; counters[speaker] += 1
                    audio, sr = sf.read(Path(a.en_dir if src == "en" else a.sk_dir) / f"{src}_{i:02d}.wav", dtype="float32")
                    wav, ref = resample16k(audio, sr), None
                else:
                    audio, sr, _ = tts[src].synthesize(text, language=src)
                    wav, ref = resample16k(audio, sr), text
                speech_s = len(wav) / 16000
                hyp, stt_s = (stt_en if src == "en" else stt_sk).transcribe(wav, src)
                translated, mt_s = mt[src].translate(hyp if hyp.strip() else ".", src, tgt)
                t = time.perf_counter()
                out, osr, _ = tts[tgt].synthesize(translated[:600], language=tgt)
                tts_s = time.perf_counter() - t
                out = np.asarray(out, dtype=np.float32)
                turn = {"speaker": speaker, "src": src, "tgt": tgt, "start": round(clock, 3), "speech_s": round(speech_s, 2),
                        "stt_s": round(stt_s, 3), "mt_s": round(mt_s, 3), "tts_s": round(tts_s, 3), "play_s": round(len(out) / osr, 2),
                        "latency_s": round(stt_s + mt_s + tts_s, 3), "reference": ref, "heard": hyp, "translation": translated,
                        "audio_b64": wav_b64(out, osr)}
                if ref:
                    import jiwer
                    turn["wer"] = round(jiwer.wer(norm(ref), norm(hyp)), 3) if hyp.strip() else 1.0
                turns.append(turn)
                clock += speech_s + turn["latency_s"] + turn["play_s"] + GAP_S
                print(f"[{label}/{mode}] turn {len(turns)} {speaker} {src}->{tgt}: STT {stt_s:.2f} MT {mt_s:.2f} TTS {tts_s:.2f} = {turn['latency_s']:.2f}s | {hyp!r} -> {translated!r}", flush=True)
            runs.append({"label": label, "input": mode, "turns": turns, "total_s": round(clock, 1)})

    report = {"machine": os.environ.get("BP_DEMO_MACHINE", "this machine"), "load_s": round(load_s, 1), "en_stt": en_label, "runs": runs}
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    slim = {**report, "runs": [{**r, "turns": [{k: v for k, v in t.items() if k != "audio_b64"} for t in r["turns"]]} for r in runs]}
    (outdir / "conversation_demo.json").write_text(json.dumps(slim, indent=1, ensure_ascii=False), encoding="utf-8")
    (outdir / "conversation_demo.html").write_text(render(report), encoding="utf-8")
    print(f"wrote {outdir / 'conversation_demo.html'}")
    if a.docs_dir:
        export_docs(slim, Path(a.docs_dir))


SVG_STYLE = ('<style>.lane{font:12px sans-serif;fill:#0f172a}.axis{stroke:#e2e8f0}.tick,.turn{font:10px sans-serif;fill:#64748b}'
             '.lat{font:600 11px sans-serif;fill:#0f172a}.bar{font:11px sans-serif;fill:#fff}</style><rect width="100%" height="100%" fill="#fff"/>')


def export_docs(report, docdir):
    """Audio-free assets for the repository documentation (GitHub renders the SVGs inline)."""
    docdir.mkdir(parents=True, exist_ok=True)
    (docdir / "conversation_demo.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (docdir / "conversation_demo_noaudio.html").write_text(render(report), encoding="utf-8")
    for run in report["runs"]:
        slug = re.sub(r"[^a-z0-9]+", "-", f'{run["label"].split(" (")[0]}-{run["input"]}'.lower()).strip("-")
        for name, fn in (("timeline", timeline_svg), ("breakdown", breakdown_svg)):
            svg = fn(run)
            head, rest = svg.split(">", 1)
            (docdir / f"{name}_{slug}.svg").write_text(head + ">" + SVG_STYLE + rest, encoding="utf-8")
    print(f"wrote documentation assets to {docdir}")


# ---------------------------------------------------------------- report ----
def med(run, direction, key="latency_s"):
    v = [t[key] for t in run["turns"] if t["src"] == direction]
    return statistics.median(v) if v else float("nan")


def timeline_svg(run):
    T = run["total_s"]; W, LEFT = 1000, 150
    sx = lambda s: LEFT + (W - LEFT - 20) * s / T
    lane_y = {"A": 40, "B": 110}
    svg = [f'<svg viewBox="0 0 {W} 190" role="img" aria-label="Conversation timeline" xmlns="http://www.w3.org/2000/svg">']
    for lane, label in (("A", "Person A (English)"), ("B", "Person B (Slovak)")):
        y = lane_y[lane]
        svg.append(f'<text x="8" y="{y+22}" class="lane">{label}</text><line x1="{LEFT}" x2="{W-20}" y1="{y+30}" y2="{y+30}" class="axis"/>')
    for i, t in enumerate(run["turns"], 1):
        me, other = lane_y[t["speaker"]], lane_y["B" if t["speaker"] == "A" else "A"]
        x = t["start"]; a = x + t["speech_s"]
        segs = [("speech", x, t["speech_s"], me, "speaking"), ("stt", a, t["stt_s"], me, "speech-to-text"),
                ("mt", a + t["stt_s"], t["mt_s"], me, "translation"), ("tts", a + t["stt_s"] + t["mt_s"], t["tts_s"], me, "text-to-speech"),
                ("play", a + t["latency_s"], t["play_s"], other, "translated voice heard")]
        for kind, s0, dur, y, name in segs:
            svg.append(f'<rect x="{sx(s0):.1f}" y="{y+4}" width="{max(sx(s0+dur)-sx(s0),1.5):.1f}" height="22" rx="2" fill="{COLORS[kind]}"><title>turn {i} {name}: {dur:.2f}s</title></rect>')
        svg.append(f'<text x="{sx(x):.1f}" y="{me}" class="turn">#{i}</text><text x="{sx(a):.1f}" y="{me+44}" class="lat">{t["latency_s"]:.1f}s</text>')
    for s in range(0, int(T) + 1, 5):
        svg.append(f'<text x="{sx(s):.1f}" y="184" class="tick">{s}s</text>')
    return "".join(svg) + "</svg>"


def breakdown_svg(run):
    rows = []
    for d, label in (("en", "EN → SK"), ("sk", "SK → EN")):
        rows.append((label, [med(run, d, k) for k in ("stt_s", "mt_s", "tts_s")]))
    scale = 880 / max(max(sum(v) for _, v in rows), 0.5)
    svg = ['<svg viewBox="0 0 1000 96" role="img" aria-label="Where the time goes" xmlns="http://www.w3.org/2000/svg">']
    for i, (label, vals) in enumerate(rows):
        y = 8 + i * 44; x = 100
        svg.append(f'<text x="4" y="{y+19}" class="lane">{label}</text>')
        for v, kind in zip(vals, ("stt", "mt", "tts")):
            w = max(v * scale, 1)
            svg.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="28" rx="2" fill="{COLORS[kind]}"><title>{kind}: {v:.2f}s</title></rect>')
            if w > 34:
                svg.append(f'<text x="{x+w/2:.1f}" y="{y+19}" class="bar" text-anchor="middle">{v:.2f}s</text>')
            x += w
        svg.append(f'<text x="{x+6:.1f}" y="{y+19}" class="lat">{sum(vals):.2f}s</text>')
    return "".join(svg) + "</svg>"


def render(r):
    runs = r["runs"]
    legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c in
                     (("speaking", COLORS["speech"]), ("speech-to-text", COLORS["stt"]), ("translation", COLORS["mt"]),
                      ("text-to-speech", COLORS["tts"]), ("translated voice heard", COLORS["play"])))
    summary = []
    for run in runs:
        wers = [t["wer"] for t in run["turns"] if t.get("wer") is not None and t["src"] == "sk"]
        summary.append(f'<tr><td>{html.escape(run["label"])}</td><td>{run["input"]}</td><td class="n">{med(run,"en"):.2f}</td>'
                       f'<td class="n"><b>{med(run,"sk"):.2f}</b></td><td class="n">{max(t["latency_s"] for t in run["turns"]):.1f}</td>'
                       f'<td class="n">{(f"{statistics.mean(wers):.2f}" if wers else "–")}</td></tr>')
    tabs, panels, css = [], [], []
    for k, run in enumerate(runs):
        tabs.append(f'<input type="radio" name="tab" id="t{k}" {"checked" if k == 0 else ""}><label for="t{k}">{html.escape(run["label"])} · {run["input"]}</label>')
        css.append(f'#t{k}:checked ~ #p{k}{{display:block}}#t{k}:checked + label{{border-bottom-color:#0ea5e9;font-weight:600}}')
        rows = []
        for i, t in enumerate(run["turns"], 1):
            d = "EN → SK" if t["src"] == "en" else "SK → EN"
            wer = f' <small>(WER {t["wer"]:.2f})</small>' if t.get("wer") is not None else ""
            rows.append(f'<tr><td>{i}</td><td>{d}</td><td>{html.escape(t["heard"])}{wer}</td><td>{html.escape(t["translation"])}</td>'
                        f'<td class="n">{t["stt_s"]:.2f}</td><td class="n">{t["mt_s"]:.2f}</td><td class="n">{t["tts_s"]:.2f}</td><td class="n"><b>{t["latency_s"]:.2f}</b></td>'
                        + (f'<td><audio controls preload="none" src="data:audio/wav;base64,{t["audio_b64"]}"></audio></td></tr>'
                           if t.get("audio_b64") else '<td>–</td></tr>'))
        panels.append(f'<section class="panel" id="p{k}"><h3>Where the time goes (median per stage)</h3>{breakdown_svg(run)}'
                      f'<h3>Conversation timeline</h3>{timeline_svg(run)}<div class="legend">{legend}</div>'
                      f'<table><thead><tr><th>#</th><th>Dir.</th><th>Recognized</th><th>Translated</th><th>STT s</th><th>MT s</th><th>TTS s</th><th>Total s</th><th>Voice</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live translation demo</title><style>
:root{{--bg:#fff;--fg:#0f172a;--mut:#64748b;--line:#e2e8f0}}@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0b1220;--fg:#e2e8f0;--mut:#94a3b8;--line:#1e293b;color-scheme:dark}}}}:root[data-theme="dark"]{{--bg:#0b1220;--fg:#e2e8f0;--mut:#94a3b8;--line:#1e293b;color-scheme:dark}}
body{{margin:0 auto;max-width:1060px;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:17px;margin:26px 0 6px}}h3{{font-size:14px;margin:18px 0 4px;color:var(--mut);font-weight:600}}p.sub,li{{color:var(--mut)}}
svg{{width:100%;height:auto}}.lane{{font-size:12px;fill:var(--fg)}}.axis{{stroke:var(--line)}}.tick,.turn{{font-size:10px;fill:var(--mut)}}.lat{{font-size:11px;fill:var(--fg);font-weight:600}}.bar{{font-size:11px;fill:#fff}}
.legend span{{margin-right:14px;font-size:13px}}.legend i{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:5px;vertical-align:-1px}}
table{{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px}}th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}td.n{{text-align:right;font-variant-numeric:tabular-nums}}
audio{{height:30px;width:170px}}input[type=radio]{{position:absolute;opacity:0;pointer-events:none}}input[type=radio]:focus-visible + label{{outline:2px solid #0ea5e9;outline-offset:2px}}label{{display:inline-block;padding:8px 12px;cursor:pointer;border-bottom:3px solid transparent;color:var(--mut);font-size:14px}}
.panel{{display:none;border-top:1px solid var(--line);padding-top:6px}}{"".join(css)}.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}.cols div{{border:1px solid var(--line);border-radius:8px;padding:10px 14px}}
@media(max-width:700px){{table{{display:block;overflow-x:auto}}}}</style></head><body>
<h1>Live speech translation: two people, two languages, measured</h1>
<p class="sub">Person A speaks English, Person B answers in Slovak; each sentence goes speech → text → translation → synthetic voice on a CPU-only laptop ({html.escape(r["machine"])}). Models load once in {r["load_s"]} s.</p>
<div class="cols"><div><b>What it is</b><br>A local pipeline: voice-activity detection, Whisper/Parakeet speech recognition, Opus-MT translation (CTranslate2 int8), Piper voice. No cloud, no GPU.</div>
<div><b>What you see</b><br>For each turn: time spent speaking, then recognition, translation and voice synthesis, then the other person hearing the result. "Total" = end of speech → translated audio ready.</div>
<div><b>Status</b><br>English → Slovak is about 1 s. Slovak → English depends on the recognizer: compare the rows below.</div></div>
<h2>Recognizers compared (same conversation)</h2>
<table><thead><tr><th>Slovak recognizer</th><th>Input</th><th>EN→SK total s (median)</th><th>SK→EN total s (median)</th><th>Slowest turn s</th><th>SK WER</th></tr></thead><tbody>{"".join(summary)}</tbody></table>
<h2>Detail per run</h2><div>{"".join(tabs)}{"".join(panels)}</div>
<p class="sub">Total is speech end → complete translated audio; TTS is not streamed here, so it is an upper bound on time-to-first-sound. WER only for synthetic input (known text). Generated by scripts/demo_conversation.py.</p>
</body></html>"""


if __name__ == "__main__":
    main()
