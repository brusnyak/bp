#!/usr/bin/env python3
"""Build ui/voice-lab/library.json: the static manifest behind the Voice Lab page.

Scans (stdlib only — run with any python3):
  speaker_voices/*.{wav,m4a,mp3}  (+ transcript from speaker_voices.json)
  processed/voice_qc/*.wav        (+ engine/latency from candidates.json,
                                   + similarity from scores.json if present)
  test/*.wav                      (+ *_transcript.txt sidecar if present)

Writes ui/voice-lab/library.json with relative paths. The page (lab.html) reads
only this file — no backend needed. Re-run after every new recording or QC run:

    python3 scripts/update_voice_lab_library.py
"""

import argparse
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_EXTS = (".wav", ".m4a", ".mp3", ".ogg", ".flac")


def audio_files(directory):
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [n for n in names if n.lower().endswith(AUDIO_EXTS)]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true",
                    help="skip the legacy test/ clips section (test fixtures, not voice material)")
    args = ap.parse_args()
    library = {"sections": []}

    # 1. Speaker voices (reference recordings + transcripts).
    sv_dir = os.path.join(REPO_ROOT, "speaker_voices")
    meta = load_json(os.path.join(sv_dir, "speaker_voices.json")) or []
    meta_by_file = {os.path.basename(m.get("path", "")): m for m in meta if m.get("path")}
    # Featured first: the three new v2b takes (owner review 2026-09-29); the rest
    # stay available under a collapsed toggle in the page.
    FEATURED_VOICES = {"cs_staromestske_v2.m4a", "en_rainbow_v2b.m4a", "sk_trhove_rano_v2b.m4a"}
    voices = []
    for name in audio_files(sv_dir):
        m = meta_by_file.get(name, {})
        voices.append(
            {
                "name": m.get("name", os.path.splitext(name)[0]),
                "file": "../../speaker_voices/" + name,
                "language": m.get("language", "?"),
                "transcript": m.get("transcribed_text", "").strip(),
                "in_registry": name in meta_by_file,
                "featured": name in FEATURED_VOICES,
            }
        )
    library["sections"].append(
        {"id": "voices", "title": "Speaker voices (reference recordings)", "items": voices}
    )

    # 2. QC candidates (synth engine outputs + measurements).
    qc_dir = os.path.join(REPO_ROOT, "processed", "voice_qc")
    manifest = load_json(os.path.join(qc_dir, "candidates.json")) or {}
    scores = {
        r.get("label"): r.get("cosine_similarity")
        for r in (load_json(os.path.join(qc_dir, "scores.json")) or {}).get("results", [])
    }
    by_label = {c.get("label"): c for c in manifest.get("candidates", [])}
    # Showcase curation (owner review 2026-09-29): the A/B pair + the two decent
    # fine-tunes stay visible; default-voice renders and superseded fine-tunes stay
    # on disk (and in candidates.json numbers) but are hidden from the showcase.
    QC_HIDDEN = {
        "male_last_base", "male_last_ns05", "piper_male_sk_alldata",
        "sandbox_cs_CZ-jirka-medium_cs", "sandbox_en_US-personal-medium_en",
        "sandbox_en_US-personal-v2_en", "sandbox_en_US-ryan-medium_en",
    }
    qc_items = []
    for name in audio_files(qc_dir):
        label = os.path.splitext(name)[0]
        if label in QC_HIDDEN:
            continue
        c = by_label.get(label, {})
        qc_items.append(
            {
                "name": label,
                "file": "../../processed/voice_qc/" + name,
                "engine": c.get("engine", "?"),
                "language": c.get("language", "?"),
                "sample_rate": c.get("sample_rate"),
                "synthesis_latency_s": c.get("synthesis_latency_s"),
                "similarity": scores.get(label),
            }
        )
    library["sections"].append(
        {"id": "qc", "title": "QC candidates (engine outputs)", "items": qc_items}
    )

    # 3. Test clips (pipeline inputs + ground-truth transcripts).
    # Skipped with --no-test: test/*.wav are old unit-test fixtures, not voice material.
    if not args.no_test:
        test_dir = os.path.join(REPO_ROOT, "test")
        test_items = []
        for name in audio_files(test_dir):
            stem = os.path.splitext(name)[0]
            transcript = ""
            for cand in (stem + "_transcript.txt", stem + " transcript.txt"):
                p = os.path.join(test_dir, cand)
                if os.path.exists(p):
                    with open(p) as f:
                        transcript = f.read().strip()
                    break
            test_items.append(
                {"name": stem, "file": "../../test/" + name, "transcript": transcript}
            )
        library["sections"].append(
            {"id": "test", "title": "Test clips (pipeline inputs)", "items": test_items}
        )

    # 4. Conversation sim (end-to-end sentence outputs + per-stage latencies).
    conv_dir = os.path.join(REPO_ROOT, "processed", "conversation")
    sim = load_json(os.path.join(REPO_ROOT, "processed", "conversation_sim.json")) or {}
    sim_by_key = {(s.get("dir"), s.get("n")): s for s in sim.get("sections", []) or sim.get("sentences", [])}
    conv_items = []
    for name in audio_files(conv_dir):
        stem = os.path.splitext(name)[0]  # e.g. en_sk_00
        m = re.match(r"(.+)_(\d+)$", stem)
        meta = dict(sim_by_key.get((m.group(1), int(m.group(2))), {})) if m else {}
        meta.pop("dir", None)
        meta.pop("n", None)
        conv_items.append({"name": stem, "file": "../../processed/conversation/" + name,
                           "meta": meta})
    library["sections"].append(
        {"id": "conversation", "title": "Conversation sim (live-pipeline sentence outputs)", "items": conv_items}
    )

    # 5. SK->EN Direction Matrix (STT WER/CER + MT chrF + latencies).
    sk_matrix_file = os.path.join(REPO_ROOT, "processed", "sk_direction", "sk_direction_matrix.json")
    sk_matrix = load_json(sk_matrix_file)
    if sk_matrix and "clips" in sk_matrix:
        matrix_items = []
        for clip_id, cdata in sk_matrix["clips"].items():
            audio_path = os.path.join("..", "..", "speaker_voices", f"{clip_id}.m4a")
            matrix_items.append({
                "name": clip_id,
                "file": audio_path,
                "audio_s": cdata.get("audio_s"),
                "ref_chars": cdata.get("ref_chars"),
                "en_ref_chars": cdata.get("en_ref_chars"),
                "rungs": cdata.get("rungs", {}),
            })
        library["sections"].append({
            "id": "sk_direction",
            "title": "SK→EN Pipeline Matrix (STT Rungs & MT chrF)",
            "items": matrix_items,
        })

    # 6. Demo charts (PNGs rendered by scripts/build_demo_charts.py from measured JSONs).
    charts_dir = os.path.join(REPO_ROOT, "ui", "voice-lab", "charts")
    captions = {
        "latency_by_direction.png": "Mean STT/MT/TTS seconds per turn, EN→SK vs SK→EN (processed/conversation_sim.json)",
        "wer_synth_vs_mic.png": "SK WER: clean Piper synth input vs real microphone, base vs small-sk (processed/stt_input_test + sk_direction)",
        "timeline_meeting.png": "Every simulated conference turn in order, per-stage stack (processed/conversation_sim.json)",
        "synth_by_length.png": "Piper synth seconds + RTF at ~7/15/30s output, generic vs personal (processed/synth_lengths)",
        "gantt_meeting.png": "Two speaker lanes over meeting time: speech, STT/MT/TTS, translated playback (processed/meeting/meeting_timeline.json)",
    }
    chart_items = [
        {"name": os.path.splitext(n)[0], "file": "charts/" + n,
         "caption": captions.get(n, "")}
        for n in sorted(os.listdir(charts_dir))
        if n.lower().endswith(".png") and os.path.isfile(os.path.join(charts_dir, n))
    ] if os.path.isdir(charts_dir) else []
    library["sections"].append(
        {"id": "charts", "title": "Demo charts (measured, read from JSON)", "items": chart_items}
    )

    # 7. STT input control (clean synth WAVs + WER per STT rung).
    stt_in_dir = os.path.join(REPO_ROOT, "processed", "stt_input_test")
    stt_in_matrix = load_json(os.path.join(stt_in_dir, "clean_synth_matrix.json")) or {}
    stt_in_items = []
    for name in audio_files(stt_in_dir):
        stem = os.path.splitext(name)[0]  # e.g. clean_sk_generic
        stt_in_items.append({"name": stem, "file": "../../processed/stt_input_test/" + name})
    # Attach the full WER matrix once (lab.js renders it as a table).
    if stt_in_matrix.get("results"):
        stt_in_items.append({"name": "_matrix", "file": "",
                             "matrix": stt_in_matrix["results"],
                             "ref_chars": stt_in_matrix.get("ref_chars")})
    if stt_in_items:
        library["sections"].append(
            {"id": "stt_input", "title": "STT input control (clean synth vs mic)", "items": stt_in_items}
        )

    # 8. Zero-shot clones (OmniVoice outputs + receipts).
    # Reference heads (ref_*) are clone INPUTS, not outputs; legacy hello/thirty_second
    # runs are superseded by the v2b clones (owner review 2026-09-29) — all stay on
    # disk with receipts, only current outputs are showcased.
    omni_dir = os.path.join(REPO_ROOT, "processed", "omnivoice")
    omni_items = []
    for name in audio_files(omni_dir):
        stem = os.path.splitext(name)[0]
        if stem.startswith("ref_") or stem.startswith("hello_reference") or stem.startswith("thirty_second_reference"):
            continue
        receipt = load_json(os.path.join(omni_dir, stem + ".json")) or {}
        lat = receipt.get("latency_seconds", {})
        omni_items.append({
            "name": stem, "file": "../../processed/omnivoice/" + name,
            "meta": {
                "reference": os.path.basename(str(receipt.get("reference", "?"))),
                "target_text": receipt.get("target_text", ""),
                "synthesis_s": lat.get("synthesis"), "rtf": lat.get("rtf"),
            },
        })
    if omni_items:
        library["sections"].append(
            {"id": "zeroshot", "title": "Zero-shot voice clones (OmniVoice, isolated eval)", "items": omni_items}
        )

    # 9. Demo audio (offline EN<->SK pipeline receipts + WAVs).
    demo_dir = os.path.join(REPO_ROOT, "processed", "demo_audio")
    demo_items = []
    for name in audio_files(demo_dir):
        stem = os.path.splitext(name)[0]
        receipt = load_json(os.path.join(demo_dir, stem + ".json")) or {}
        lat = receipt.get("latency_seconds", {})
        demo_items.append({
            "name": stem, "file": "../../processed/demo_audio/" + name,
            "meta": {
                "direction": receipt.get("direction", ""),
                "transcript": receipt.get("transcript", ""),
                "translation": receipt.get("translation", ""),
                "stt_s": lat.get("stt"), "mt_s": lat.get("mt"), "tts_s": lat.get("tts"),
            },
        })
    if demo_items:
        library["sections"].append(
            {"id": "demo_audio", "title": "Demo turns (offline pipeline, playable fallback)", "items": demo_items}
        )

    # 10. Meeting sim (scripted 2-speaker turns + absolute timeline; Lab bottom section).
    meet_dir = os.path.join(REPO_ROOT, "processed", "meeting")
    meet = load_json(os.path.join(meet_dir, "meeting_timeline.json")) or {}
    meet_items = []
    if meet.get("mix_wav"):
        meet_items.append({
            "name": "_mix",
            "file": "../../" + meet["mix_wav"],
            "meta": {"total_s": meet.get("total_s"),
                     "timing_model": meet.get("timing_model", ""),
                     "chapters": meet.get("chapters", [])},
        })
    for t in meet.get("turns", []):
        wav = t.get("playback_wav", "")
        meet_items.append({
            "name": f"turn_{t.get('n', '?'):02d}_{t.get('direction', '?').replace('->', '-')}",
            "file": "../../" + wav if wav else "",
            "meta": {
                "speaker": t.get("speaker", ""), "direction": t.get("direction", ""),
                "script": t.get("script", ""), "heard": t.get("transcript", ""),
                "translation": t.get("translation", ""),
                "starts_at": t.get("t_start_s"), "speech_s": t.get("speech_s"),
                "stt_s": t.get("stt_s"), "mt_s": t.get("mt_s"),
                "tts_s": t.get("tts_s"), "playback_s": t.get("playback_s"),
                "first_word_s": t.get("mt_first_word_s"),
                "first_audio_s": t.get("tts_first_chunk_s"),
                "playback_starts": t.get("first_audio_at", t.get("playback_start_s")),
                "streaming": t.get("streaming", False),
                "live": t.get("live", False),
            },
        })
    if meet_items:
        library["sections"].append(
            {"id": "meeting", "title": "Simulated meeting (scripted turns + timeline)", "items": meet_items}
        )

    # 11. Bulk HQ corpus (OmniVoice clones for the future Piper voice).
    bulk_dir = os.path.join(REPO_ROOT, "processed", "bulk_hq")
    bulk = load_json(os.path.join(bulk_dir, "manifest.json")) or {}
    bulk_items = []
    for c in bulk.get("clips", []):
        wav = os.path.basename(c.get("wav", ""))
        if wav.endswith(".wav") and os.path.isfile(os.path.join(bulk_dir, wav)):
            bulk_items.append({
                "name": os.path.splitext(wav)[0],
                "file": "../../processed/bulk_hq/" + wav,
                "meta": {"audio_s": c.get("audio_s"), "rtf": c.get("rtf"),
                         "qc_wer": c.get("qc_wer_smallsk"), "qc_rms": c.get("qc_rms")},
            })
    if bulk_items:
        library["sections"].append(
            {"id": "corpus", "title": "Piper training corpus (OmniVoice bulk HQ)", "items": bulk_items}
        )

    out = os.path.join(REPO_ROOT, "ui", "voice-lab", "library.json")
    with open(out, "w") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
    counts = {s["id"]: len(s["items"]) for s in library["sections"]}
    print(f"Wrote {out}: {counts}")


if __name__ == "__main__":
    main()
