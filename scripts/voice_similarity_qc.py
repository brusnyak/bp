"""Objective voice-similarity QC: cosine similarity of speaker embeddings vs real reference.

What changed 2026-09-24: candidates are no longer hardcoded paths on the author's
Desktop (unreproducible anywhere else). Each candidate is synthesized fresh from the
project's own live TTS engines (backend/tts/base.py registry) into OUT_DIR, then scored.
Run from a clean checkout on any machine with the models present.

Two-venv split (resemblyzer must NOT go into the project's main venv/requirements.txt):
    # 1. synthesize (project venv — torch, piper, TTS engines, models):
    ./venv/bin/python scripts/voice_similarity_qc.py --synthesize-only
    # 2. score (isolated qc venv — resemblyzer only):
    python3 -m venv /path/to/qc-venv
    /path/to/qc-venv/bin/pip install resemblyzer "setuptools<81"
    # setuptools<81 is required: resemblyzer's webrtcvad dependency imports
    # pkg_resources, which setuptools>=81 stopped shipping.
    /path/to/qc-venv/bin/python scripts/voice_similarity_qc.py --score-only
    # Or both steps at once if one venv happens to have all deps:
    ./venv/bin/python scripts/voice_similarity_qc.py

Adding a future language bootstrap: extend CANDIDATE_TEXTS + candidate_specs()
(e.g. an sk_SK-personal entry once backend/tts/piper_models/sk_SK-personal-medium.onnx
exists) — no other file changes needed.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

OUT_DEFAULT = os.path.join(REPO_ROOT, "processed", "voice_qc")

EN_REFERENCE_WAV = os.path.join(REPO_ROOT, "speaker_voices", "voice_rec_1m.wav")
# Optional: recorded SK reference once the owner records the SK session
# (see documentation/voice_and_app_direction_2026-09.md §3). Absent → SK candidates
# are scored against the EN reference with a cross-lingual flag (speaker embeddings
# are text-independent, so this still measures identity, just noisier).
SK_REFERENCE_WAV = os.path.join(REPO_ROOT, "speaker_voices", "sk_ref.wav")

CANDIDATE_TEXTS = {
    "en": "This is a short test sentence synthesized for voice similarity quality control.",
    "sk": "Toto je krátka testovacia veta vytvorená na kontrolu kvality podobnosti hlasu.",
}

# Threshold reference point (NOT a formal calibration): resemblyzer's own
# demo05_fake_speech_detection.py draws its real/fake decision line at
# cosine similarity 0.84 (`plt.axhline(0.84, ls="dashed", ...)`). That demo
# compares utterances of the same reference speaker against real vs.
# TTS-generated audio, which is the closest published number to this use
# case, but it is illustrative, not a general speaker-verification EER cutoff.
RESEMBLYZER_DEMO_THRESHOLD = 0.84


def candidate_specs(include_xtts=False):
    """(label, registry_key, text_lang, speaker_wav_or_None).

    speaker_wav None → engine's own voice (Piper fixed voices). Set → cloned output.
    """
    specs = [
        # Fine-tuned personal voice (EN-only) — the §2b destination for EN.
        ("piper_personal_en", "piper_personal", "en", None),
        # Generic baseline in the target language — the floor every clone must beat.
        ("piper_generic_sk", "piper", "sk", None),
        # Live default: Hybrid cross-lingual clone from the single EN recording.
        ("hybrid_sk", "hybrid", "sk", EN_REFERENCE_WAV),
        # Same-language Hybrid control — isolates converter quality from cross-lingual loss.
        ("hybrid_en", "hybrid", "en", EN_REFERENCE_WAV),
    ]
    if include_xtts:
        # Slow (RTF ~1.7), kept out of the default set — fallback tier only.
        specs.append(("xtts_en", "xtts", "en", EN_REFERENCE_WAV))
    return specs


def synthesize_all(out_dir, include_xtts=False):
    import soundfile as sf

    from backend.tts.base import TTS_ENGINES

    os.makedirs(out_dir, exist_ok=True)
    manifest = {"candidates": []}
    for label, registry_key, lang, speaker_wav in candidate_specs(include_xtts):
        if speaker_wav is not None and not os.path.exists(speaker_wav):
            print(f"SKIP {label}: speaker reference not found at {speaker_wav}")
            continue
        print(f"Synthesizing {label} via TTS_ENGINES[{registry_key!r}] ({lang}) ...")
        engine = TTS_ENGINES[registry_key]()
        wav, sr, latency = engine.synthesize(
            CANDIDATE_TEXTS[lang], language=lang, speaker_wav_path=speaker_wav
        )
        out_path = os.path.join(out_dir, f"{label}.wav")
        sf.write(out_path, wav, sr)
        manifest["candidates"].append(
            {
                "label": label,
                "wav": out_path,
                "engine": registry_key,
                "language": lang,
                "sample_rate": sr,
                "synthesis_latency_s": round(latency, 4),
            }
        )
        print(f"  -> {out_path} ({sr} Hz, {latency:.2f}s synthesis)")
    manifest_path = os.path.join(out_dir, "candidates.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")
    return manifest_path


def cosine_similarity(a, b):
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_all(out_dir):
    from resemblyzer import VoiceEncoder, preprocess_wav

    manifest_path = os.path.join(out_dir, "candidates.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print("Loading resemblyzer VoiceEncoder...")
    encoder = VoiceEncoder()

    references = {"en": EN_REFERENCE_WAV}
    if os.path.exists(SK_REFERENCE_WAV):
        references["sk"] = SK_REFERENCE_WAV
    else:
        print(f"NOTE: no SK reference at {SK_REFERENCE_WAV} — SK candidates scored "
              f"against the EN reference (cross-lingual, noisier).")

    ref_embeds = {}
    for lang, path in references.items():
        print(f"Embedding {lang} reference: {path}")
        ref_embeds[lang] = encoder.embed_utterance(preprocess_wav(path))

    results = []
    for cand in manifest["candidates"]:
        path = cand["wav"]
        if not os.path.exists(path):
            print(f"SKIP {cand['label']}: file not found at {path}")
            continue
        # Same-language reference when available, else EN (flagged cross-lingual).
        ref_lang = cand["language"] if cand["language"] in ref_embeds else "en"
        cross = "cross-lingual" if ref_lang != cand["language"] else "same-language"
        embed = encoder.embed_utterance(preprocess_wav(path))
        sim = cosine_similarity(ref_embeds[ref_lang], embed)
        results.append((cand["label"], sim))
        print(f"{cand['label']:20s} vs {ref_lang} reference [{cross}]: "
              f"cosine similarity = {sim:.4f}")

    print(f"\nResemblyzer demo threshold reference point: {RESEMBLYZER_DEMO_THRESHOLD}")
    print("Gate rule (documentation/voice_and_app_direction_2026-09.md §2b): a voice registers "
          "as personal only on similarity pass AND blind-listening pass — RTF proves speed, "
          "this script + listeners prove identity.")
    # Persisted for the voice-lab eval page (ui/voice-lab/ reads scores.json if present).
    scores_path = os.path.join(out_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(
            {
                "threshold_reference": RESEMBLYZER_DEMO_THRESHOLD,
                "results": [
                    {"label": label, "cosine_similarity": round(sim, 4)} for label, sim in results
                ],
            },
            f,
            indent=2,
        )
    print(f"Scores: {scores_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Synthesize QC candidates and score voice similarity.")
    parser.add_argument("--synthesize-only", action="store_true", help="only synthesize (project venv)")
    parser.add_argument("--score-only", action="store_true", help="only score (qc venv)")
    parser.add_argument("--out-dir", default=OUT_DEFAULT, help="candidate wav + manifest dir")
    parser.add_argument("--include-xtts", action="store_true", help="include slow XTTS fallback candidate")
    args = parser.parse_args()

    do_synth = args.synthesize_only or not args.score_only
    do_score = args.score_only or not args.synthesize_only

    if do_synth:
        synthesize_all(args.out_dir, include_xtts=args.include_xtts)
    if do_score:
        score_all(args.out_dir)


if __name__ == "__main__":
    main()
