import sys
import os
import time
import numpy as np
import soundfile as sf
from typing import List, Dict, Tuple, Optional

# Add the parent directory (BP) to sys.path to import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.utils.audio_utils import load_audio

# Function to calculate Word Error Rate (WER) - simplified for demonstration
def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Simple WER calculation (Levenshtein distance based)
    # For a more robust WER, a dedicated library like jiwer would be used.
    # This is a basic implementation to show differences.
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    if n == 0:
        return float(m) # If reference is empty, WER is just the number of inserted words
    return dp[n][m] / n

def load_transcription(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def compare_stt_models():
    print("Starting STT model comparison (local transcription)...")

    # Define audio files and their ground truth transcriptions
    test_cases = [
        {
            "audio_path": "test/Can you hear me_.wav",
            "transcript_path": "test/Can you hear me_transcript.txt",
            "language": "en",
            "description": "Short English audio"
        },
        {
            "audio_path": "test/My test speech_xtts_speaker_clean.wav",
            "transcript_path": "test/My test speech transcript.txt",
            "language": "en",
            "description": "Long English audio"
        },
    ]

    # Define STT model sizes to compare
    stt_model_sizes = ["tiny", "base", "medium", "large"] # Add other sizes as needed
    results: List[Dict[str, any]] = []

    for stt_model_size in stt_model_sizes:
        print(f"\n--- Starting comparison for FasterWhisperSTT model: {stt_model_size} ---")
        
        # Initialize FasterWhisperSTT locally for direct transcription
        try:
            faster_whisper_stt = FasterWhisperSTT(model_size=stt_model_size, compute_type="int8")
        except Exception as e:
            print(f"Error initializing FasterWhisperSTT model {stt_model_size}: {e}")
            continue

        current_model_results: List[Dict[str, any]] = []
        
        for case in test_cases:
            audio_path = case["audio_path"]
            transcript_path = case["transcript_path"]
            language = case["language"]
            description = case["description"]

            print(f"\n--- Processing: {description} ({audio_path}) with {stt_model_size} model ---")

            # Load audio data and resample to 16kHz
            audio_data, sample_rate = load_audio(audio_path, target_sr=16000)
            
            # Load ground truth transcription
            ground_truth = load_transcription(transcript_path)
            print(f"Ground Truth: '{ground_truth}'")

            # --- Perform local transcription ---
            transcribed_segments, transcription_time, detected_lang = faster_whisper_stt.transcribe_audio(
                audio_data, sample_rate, language=language, vad_filter=True # Use FasterWhisper's VAD
            )
            local_transcribed_text = " ".join([s.text for s in transcribed_segments]).strip()
            local_wer = calculate_wer(ground_truth, local_transcribed_text)
            
            print(f"Local Transcription: '{local_transcribed_text}'")
            print(f"Local Transcription Time: {transcription_time:.4f}s")
            print(f"Local WER: {local_wer:.2f}")

            current_model_results.append({
                "description": description,
                "audio_path": audio_path,
                "ground_truth": ground_truth,
                "local_stt": {
                    "transcription": local_transcribed_text,
                    "transcription_time": transcription_time,
                    "wer": local_wer,
                    "detected_language": detected_lang
                }
            })
        results.append({"model_size": stt_model_size, "test_cases": current_model_results})
    
    print("\n--- Overall Comparison Summary ---")
    for model_res in results:
        print(f"\nModel Size: {model_res['model_size']}")
        for case_res in model_res['test_cases']:
            print(f"  Test Case: {case_res['description']}")
            print(f"    Ground Truth: {case_res['ground_truth']}")
            print(f"    Local Transcription: '{case_res['local_stt']['transcription']}'")
            print(f"    Transcription Time: {case_res['local_stt']['transcription_time']:.4f}s")
            print(f"    WER: {case_res['local_stt']['wer']:.2f}")
            print(f"    Detected Language: {case_res['local_stt']['detected_language']}")

if __name__ == "__main__":
    compare_stt_models()
