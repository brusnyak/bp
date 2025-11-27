import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Add the parent directory (BP) to sys.path to import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.utils.audio_utils import load_audio

# The F5-TTS related imports and code have been removed as F5-TTS is no longer part of the project.
# The functionality for calculating speaker similarity using ECAPA_TDNN_SMALL is also removed.
# If speaker similarity is needed in the future, a new implementation using a different model
# compatible with the current project dependencies will be required.

def calculate_speaker_similarity(audio_path1: str, audio_path2: str) -> float:
    """
    Placeholder function for speaker similarity calculation.
    Returns -1.0 as the F5-TTS dependent implementation has been removed.
    """
    print("Speaker similarity calculation is currently not supported as F5-TTS has been removed.")
    return -1.0

if __name__ == "__main__":
    # Example usage
    audio_file1 = "test/Hello.wav"
    audio_file2 = "test_output/f5_tts_tests/full_pipeline_output_F5_base_en-sk.wav" # Example generated F5-TTS audio

    if os.path.exists(audio_file1) and os.path.exists(audio_file2):
        similarity_score = calculate_speaker_similarity(audio_file1, audio_file2)
        print(f"Similarity between '{audio_file1}' and '{audio_file2}': {similarity_score:.4f}")
    else:
        print("One or both audio files not found for similarity test.")
