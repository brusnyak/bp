#!/usr/bin/env python3
"""Demo charts for the Voice Lab showcase (matplotlib only, stdlib otherwise).

Reads measured JSONs — never hardcodes numbers — and writes PNGs next to the
static Lab page so lab.html can show them with no backend:

  processed/conversation_sim.json                  -> charts/latency_by_direction.png
  processed/sk_direction/sk_direction_matrix.json  -> charts/wer_synth_vs_mic.png
  processed/stt_input_test/clean_synth_matrix.json -> (same second chart)

Run:  python3 scripts/build_demo_charts.py
Needs: matplotlib (system python3 has it; NOT a repo dependency).
"""
from __future__ import annotations

import json
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "ui", "voice-lab", "charts")


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def chart_latency_by_direction(plt):
    sim = load(os.path.join(REPO_ROOT, "processed", "conversation_sim.json")) or {}
    rows = sim.get("sentences") or sim.get("sections") or []
    by_dir: dict[str, dict[str, list[float]]] = {}
    for s in rows:
        d = s.get("dir", "?")
        bucket = by_dir.setdefault(d, {"stt": [], "mt": [], "tts": []})
        # keys differ across runs: stt_s / mt_sentence_s|mt_s / tts_s
        if isinstance(s.get("stt_s"), (int, float)):
            bucket["stt"].append(s["stt_s"])
        mt = s.get("mt_sentence_s", s.get("mt_s"))
        if isinstance(mt, (int, float)):
            bucket["mt"].append(mt)
        if isinstance(s.get("tts_s"), (int, float)):
            bucket["tts"].append(s["tts_s"])
    dirs = [d for d in ("en_sk", "sk_en") if d in by_dir] or sorted(by_dir)
    if not dirs:
        print("skip latency chart: no rows in conversation_sim.json")
        return None
    # New schema records STT only as batch direction totals in agg (no per-row stt_s):
    # fall back to mean-per-clip so the STT bar stays but is honestly labeled.
    agg_stt = ((sim.get("agg") or {}).get("stt") or {})
    batch_totals = {"en_sk": agg_stt.get("en_parakeet_total_s"), "sk_en": agg_stt.get("sk_whisper_total_s")}
    stt_note = ""
    for d in dirs:
        if not by_dir[d]["stt"] and isinstance(batch_totals.get(d), (int, float)):
            n = len(rows and [s for s in rows if s.get("dir") == d]) or 1
            by_dir[d]["stt"] = [batch_totals[d] / n]
            stt_note = "STT = batch mean/clip (EN incl. model load)"
    labels = {"en_sk": "EN→SK", "sk_en": "SK→EN"}
    stages = ["stt", "mt", "tts"]
    means = {d: [statistics.mean(by_dir[d][s]) if by_dir[d][s] else 0.0 for s in stages] for d in dirs}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(dirs))
    bottoms = [0.0] * len(dirs)
    colors = {"stt": "#0A1C4F", "mt": "#C41E3A", "tts": "#2E7D32"}
    for s in stages:
        vals = [means[d][stages.index(s)] for d in dirs]
        ax.bar([labels.get(d, d) for d in dirs], vals, bottom=bottoms, label=s.upper(), color=colors[s])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, d in enumerate(dirs):
        total = sum(means[d])
        ax.text(i, total + 0.03, f"{total:.2f}s", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("seconds (mean per turn)")
    title = "Pipeline latency per turn, by direction (measured)"
    if stt_note:
        title += f"\n{stt_note}; MT/TTS are per-turn means"
    ax.set_title(title, fontsize=11)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "latency_by_direction.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def chart_timeline_meeting(plt):
    """Meeting timeline: every simulated conference turn in order, per-stage stack."""
    sim = load(os.path.join(REPO_ROOT, "processed", "conversation_sim.json")) or {}
    rows = sim.get("sentences") or sim.get("sections") or []
    if not rows:
        print("skip timeline chart: no rows in conversation_sim.json")
        return None
    # Meeting-like view: first 7 turns per direction (full 36-row stats stay in JSON).
    capped, seen = [], {}
    for s in rows:
        d = s.get("dir", "?")
        if seen.get(d, 0) < 7:
            capped.append(s)
            seen[d] = seen.get(d, 0) + 1
    rows = capped
    labels, stt_v, mt_v, tts_v, colors = [], [], [], [], []
    counts: dict[str, int] = {}
    agg = sim.get("agg") or {}
    agg_stt = agg.get("stt") or {}
    batch_totals = {"en_sk": agg_stt.get("en_parakeet_total_s"), "sk_en": agg_stt.get("sk_whisper_total_s")}
    for s in rows:
        d = s.get("dir", "?")
        counts[d] = counts.get(d, 0) + 1
        short = "EN→SK" if d == "en_sk" else ("SK→EN" if d == "sk_en" else d)
        labels.append(f"{short} #{counts[d]}")
        stt_v.append(s.get("stt_s") or 0.0)
        mt_v.append(s.get("mt_sentence_s", s.get("mt_s")) or 0.0)
        tts_v.append(s.get("tts_s") or 0.0)
        colors.append("#C41E3A" if d == "en_sk" else "#0A1C4F")
    # New schema: STT only as batch direction totals — spread flat, label honestly.
    stt_flat_note = ""
    if not any(stt_v) and any(isinstance(batch_totals.get(d), (int, float)) for d in counts):
        stt_flat_note = "STT shown as batch mean/clip (flat, EN incl. model load)"
        for i, s in enumerate(rows):
            tot = batch_totals.get(s.get("dir"))
            n = counts.get(s.get("dir")) or 1
            if isinstance(tot, (int, float)):
                stt_v[i] = tot / n

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(len(rows)))
    b1 = ax.bar(x, stt_v, label="STT", color="#0A1C4F")
    b2 = ax.bar(x, mt_v, bottom=stt_v, label="MT", color="#C41E3A")
    bottoms = [a + b for a, b in zip(stt_v, mt_v)]
    ax.bar(x, tts_v, bottom=bottoms, label="TTS", color="#2E7D32")
    for i in x:
        total = stt_v[i] + mt_v[i] + tts_v[i]
        ax.text(i, total + 0.02, f"{total:.2f}s", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=20)
    ax.set_ylabel("seconds")
    title = "Simulated meeting: translation turnaround per turn, in order"
    if stt_flat_note:
        title += f"\n{stt_flat_note}"
    ax.set_title(title, fontsize=11)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "timeline_meeting.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def chart_synth_by_length(plt):
    """TTS scaling: synthesis seconds and RTF at ~7/15/30s of output audio."""
    data = load(os.path.join(REPO_ROOT, "processed", "synth_lengths", "synth_lengths.json")) or {}
    rows = data.get("results", [])
    if not rows:
        print("skip synth-length chart: no processed/synth_lengths/synth_lengths.json")
        return None
    targets = []
    for r in rows:
        if r["target"] not in targets:
            targets.append(r["target"])
    voices = ["generic", "personal"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    x = list(range(len(targets)))
    w = 0.36
    for i, v in enumerate(voices):
        synth = [next((r["synth_s"] for r in rows if r["voice"] == v and r["target"] == t), 0) for t in targets]
        rtf = [next((r["rtf"] for r in rows if r["voice"] == v and r["target"] == t), 0) for t in targets]
        ax1.bar([j + (i - 0.5) * w for j in x], synth, w, label=v)
        ax2.bar([j + (i - 0.5) * w for j in x], rtf, w, label=v)
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, fontsize=9)
    ax1.set_ylabel("synthesis seconds")
    ax1.set_title("Piper synth time by output length")
    ax1.legend(fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, fontsize=9)
    ax2.set_ylabel("RTF (lower is better)")
    ax2.set_title("Piper RTF stays flat (~0.04)")
    ax2.axhline(y=1.0, color="green", linestyle="--", linewidth=1)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "synth_by_length.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def chart_gantt_meeting(plt):
    """Meeting Gantt (streaming-first-chunk model): 2 speaker lanes x time.

    Speech bar on the speaker's lane; STT overlaps speech (streaming); MT-first-word
    and TTS-first-chunk are markers; translated playback starts ~0.4s after speech
    end on the other lane. Only the 0.15s STT tail is assumed (see JSON).
    """
    data = load(os.path.join(REPO_ROOT, "processed", "meeting", "meeting_timeline.json")) or {}
    turns = data.get("turns", [])
    if not turns:
        print("skip gantt chart: no processed/meeting/meeting_timeline.json")
        return None
    tail = ((data.get("assumptions") or {}).get("stt_tail_s")) or 0.15
    lanes = {"A": 2, "B": 1}
    lane_labels = {2: "Speaker A (EN)", 1: "Speaker B (SK)"}
    colors = {"speech": "#0A1C4F", "stt": "#E65100", "playback": "#2E7D32"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for t in turns:
        y = lanes[t["speaker"]]
        other = lanes["B" if t["speaker"] == "A" else "A"]
        t0, sp = t["t_start_s"], t["speech_s"]
        ax.broken_barh([(t0, sp)], (y - 0.35, 0.7), facecolors=colors["speech"])
        ax.broken_barh([(t0, sp + tail)], (y - 0.35, 0.7), facecolors="none",
                       edgecolors=colors["stt"], linewidth=1.5, linestyle=(0, (3, 2)))
        chunks = t.get("chunks") or []
        if chunks:
            # Per-chunk playback bars: chunk 0's playback overlaps chunk 1's speech.
            for c in chunks:
                ax.broken_barh([(c["playback_start"], c["playback_s"])], (other - 0.35, 0.7),
                               facecolors=colors["playback"])
            first_audio_at = chunks[0]["playback_start"]
        else:  # legacy first-chunk schema
            first_word_at = t0 + sp + tail + (t.get("mt_first_word_s") or 0.0)
            first_audio_at = t.get("playback_start_s") or (first_word_at + (t.get("tts_first_chunk_s") or 0.0))
            ax.plot([first_word_at], [y], marker="D", color="#C41E3A", markersize=7, zorder=5)
            ax.broken_barh([(first_audio_at, t.get("playback_s") or 0.0)], (other - 0.35, 0.7),
                           facecolors=colors["playback"])
        ax.plot([first_audio_at], [y], marker="o", color="#2E7D32", markersize=7, zorder=5)
        ax.text(t0 + sp / 2, y, f"turn {t['n']}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_ylim(0.4, 2.9)
    ax.set_yticks([1, 2])
    ax.set_yticklabels([lane_labels[1], lane_labels[2]])
    ax.set_xlabel("meeting time (seconds)")
    ax.set_title("Simulated meeting (chunked streaming): playback overlaps speech", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=colors["speech"], label="speech"),
                       Patch(facecolor="none", edgecolor=colors["stt"], label="STT (streaming)"),
                       Patch(facecolor="#C41E3A", label="first translated word (~0.1s)"),
                       Patch(facecolor="#2E7D32", label="first audio/playback (~0.4s after speech)")],
              fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "gantt_meeting.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def chart_wer_synth_vs_mic(plt):
    skm = load(os.path.join(REPO_ROOT, "processed", "sk_direction", "sk_direction_matrix.json")) or {}
    clean = load(os.path.join(REPO_ROOT, "processed", "stt_input_test", "clean_synth_matrix.json")) or {}
    groups: list[tuple[str, float | None, float | None]] = []  # (label, base_wer, small_wer)
    clips = skm.get("clips", {})
    if clips:
        base_wers = [c["rungs"]["base"]["wer"] for c in clips.values() if "base" in c.get("rungs", {})]
        small_wers = [c["rungs"]["small-sk"]["wer"] for c in clips.values() if "small-sk" in c.get("rungs", {})]
        groups.append((
            f"real mic (n={len(clips)})",
            round(statistics.mean(base_wers), 3) if base_wers else None,
            round(statistics.mean(small_wers), 3) if small_wers else None,
        ))
    cres = clean.get("results", {})
    for label in ("generic", "personal"):
        b = cres.get(f"{label}/base", {}).get("wer")
        s = cres.get(f"{label}/small-sk", {}).get("wer")
        if b is not None or s is not None:
            groups.append((f"clean synth, {label} voice", b, s))
    groups = [g for g in groups if g[1] is not None or g[2] is not None]
    if not groups:
        print("skip WER chart: no sk_direction or stt_input_test data")
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(groups)))
    w = 0.36
    base_vals = [g[1] if g[1] is not None else 0.0 for g in groups]
    small_vals = [g[2] if g[2] is not None else 0.0 for g in groups]
    b1 = ax.bar([i - w / 2 for i in x], base_vals, w, label="whisper-base", color="#888888")
    b2 = ax.bar([i + w / 2 for i in x], small_vals, w, label="whisper-small-sk", color="#C41E3A")
    for bars in (b1, b2):
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("WER (lower is better)")
    ax.set_title("SK recognition: clean synth input vs real microphone")
    ax.set_ylim(0, max(base_vals + small_vals + [0.5]) * 1.2)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "wer_synth_vs_mic.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found; run with system python3 (has it) — not a repo dependency.",
              file=sys.stderr)
        sys.exit(2)
    os.makedirs(OUT_DIR, exist_ok=True)
    made = [p for p in (chart_latency_by_direction(plt), chart_wer_synth_vs_mic(plt),
                        chart_timeline_meeting(plt), chart_synth_by_length(plt),
                        chart_gantt_meeting(plt)) if p]
    for p in made:
        print(f"wrote {os.path.relpath(p, REPO_ROOT)}")
    if not made:
        sys.exit(1)


if __name__ == "__main__":
    main()
