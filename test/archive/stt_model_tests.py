import os
import sys
import numpy as np
import soundfile as sf
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__)))) # Add test directory to sys.path

from test.model_testing_framework import ModelTestFramework
from backend.stt.faster_whisper_stt import FasterWhisperSTT

@pytest.fixture(scope="module")
def stt_framework():
    audio_path = "test/Can you hear me_.wav"
    transcript_path = "test/Can you hear me_transcript.txt"

    if not os.path.exists(audio_path):
        pytest.skip(f"Error: Audio file not found at {audio_path}")
    if not os.path.exists(transcript_path):
        pytest.skip(f"Error: Transcript file not found at {transcript_path}")

    return ModelTestFramework(audio_path, transcript_path=transcript_path)

@pytest.fixture(scope="module")
def faster_whisper_stt_model():
    return FasterWhisperSTT(model_size="base", device="auto", compute_type="int8")

def test_faster_whisper_stt(stt_framework: ModelTestFramework, faster_whisper_stt_model: FasterWhisperSTT):
    print("\n--- Testing Faster-Whisper STT (Base Model) ---")
    
    # Get ground truth
    ground_truth = stt_framework.ground_truth_transcript
    print(f"Ground Truth: '{ground_truth}'")

    # Measure latency and get transcription
    segments, latency = stt_framework.measure_latency(
        faster_whisper_stt_model.transcribe_audio, stt_framework.audio_data, stt_framework.samplerate, language="en"
    )
    predicted_transcript = " ".join([s.text for s in segments[0]]).strip().lower() if segments and segments[0] else ""
    
    print(f"Predicted Transcription: '{predicted_transcript}'")
    print(f"Transcription Latency: {latency:.4f} seconds")
    
    stt_results = stt_framework.evaluate_stt(predicted_transcript)
    
    assert predicted_transcript != "", "Faster-Whisper transcription failed."
    assert latency > 0, "Faster-Whisper latency not measured correctly."
    if stt_results:
        print(f"Faster-Whisper WER: {stt_results['wer']:.4f}")
        # Assert that WER is below a reasonable threshold for the given audio
        assert stt_results['wer'] < 0.3, f"Faster-Whisper WER is too high: {stt_results['wer']:.4f}" # Adjusted threshold to 0.3
    else:
        pytest.fail("STT evaluation results are missing.")
