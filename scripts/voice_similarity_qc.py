"""Objective voice-similarity QC: cosine similarity of speaker embeddings vs a real reference.

Does NOT use RTF or subjective description as a quality signal — only measured
cosine similarity between resemblyzer speaker embeddings.

Setup (isolated venv, do not install into the project's main venv/requirements.txt):
    python3 -m venv /path/to/venv
    /path/to/venv/bin/pip install resemblyzer piper-tts "setuptools<81"
    # setuptools<81 is required: resemblyzer's webrtcvad dependency imports
    # pkg_resources, which setuptools>=81 stopped shipping.

Usage:
    <venv>/bin/python scripts/voice_similarity_qc.py

Edit CANDIDATES below to add/remove samples (e.g. for a future language bootstrap).
"""

import os
import wave
import io

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE_WAV = os.path.join(REPO_ROOT, "speaker_voices/voice_rec_1m.wav")

PIPER_MODEL = os.path.join(REPO_ROOT, "backend/tts/piper_models/en_US-personal-medium.onnx")
PIPER_TEST_SENTENCE = "This is a short test sentence synthesized by my fine-tuned Piper voice."
PIPER_OUTPUT_WAV = "/tmp/voice_qc_piper_personal.wav"

# (label, wav path) pairs to compare against REFERENCE_WAV.
# The Piper entry is synthesized on the fly (see synthesize_piper_sample below);
# every other entry must already exist on disk.
CANDIDATES = [
    ("piper_personal_en", PIPER_OUTPUT_WAV),
    ("xtts_de", "/Users/yegor/Desktop/personal_voice_test_de_xtts_not_piper.wav"),
    ("xtts_cz", "/Users/yegor/Desktop/personal_voice_test_cz_xtts.wav"),
]

# Threshold reference point (NOT a formal calibration): resemblyzer's own
# demo05_fake_speech_detection.py draws its real/fake decision line at
# cosine similarity 0.84 (`plt.axhline(0.84, ls="dashed", ...)`). That demo
# compares utterances of the same reference speaker against real vs.
# TTS-generated audio, which is the closest published number to this use
# case, but it is illustrative, not a general speaker-verification EER cutoff.
RESEMBLYZER_DEMO_THRESHOLD = 0.84


def synthesize_piper_sample():
    from piper import PiperVoice

    voice = PiperVoice.load(PIPER_MODEL)
    with wave.open(PIPER_OUTPUT_WAV, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize_wav(PIPER_TEST_SENTENCE, wav_file)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    if not os.path.exists(PIPER_OUTPUT_WAV):
        print(f"Synthesizing Piper reference sample -> {PIPER_OUTPUT_WAV}")
        synthesize_piper_sample()

    print("Loading resemblyzer VoiceEncoder...")
    encoder = VoiceEncoder()

    print(f"Embedding reference: {REFERENCE_WAV}")
    ref_wav = preprocess_wav(REFERENCE_WAV)
    ref_embed = encoder.embed_utterance(ref_wav)

    results = []
    for label, path in CANDIDATES:
        if not os.path.exists(path):
            print(f"SKIP {label}: file not found at {path}")
            continue
        wav = preprocess_wav(path)
        embed = encoder.embed_utterance(wav)
        sim = cosine_similarity(ref_embed, embed)
        results.append((label, path, sim))
        print(f"{label:20s} vs reference: cosine similarity = {sim:.4f}  ({path})")

    print(f"\nResemblyzer demo threshold reference point: {RESEMBLYZER_DEMO_THRESHOLD}")
    return results


if __name__ == "__main__":
    main()
