import os
import sys
import numpy as np
import soundfile as sf

# Add the current working directory to sys.path for absolute imports
sys.path.append(os.getcwd())

from test.model_testing_framework import ModelTestFramework
from backend.stt.mlx_whisper_stt import MLXWhisperSTT
from backend.stt.faster_whisper_stt import FasterWhisperSTT

def run_stt_tests():
    audio_path = "test/My test speech_xtts_speaker_clean.wav"
    transcript_path = "test/My test speech transcript.txt"

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found at {transcript_path}")
        return

    framework = ModelTestFramework(audio_path, transcript_path=transcript_path)

    print("\n--- Testing MLX-Whisper STT ---")
    try:
        mlx_whisper_stt = MLXWhisperSTT(model_size="tiny") # Use tiny model for faster testing
        transcription_result, latency = framework.measure_latency(
            mlx_whisper_stt.transcribe_audio, framework.audio_data, framework.samplerate, language="en"
        )
        predicted_transcript = transcription_result[0]
        print(f"MLX-Whisper Latency: {latency:.4f} seconds")
        stt_results = framework.evaluate_stt(predicted_transcript)
        if stt_results:
            print(f"MLX-Whisper WER: {stt_results['wer']:.4f}")

    except Exception as e:
        print(f"Error testing MLX-Whisper: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Faster-Whisper STT (Base Model) ---")
    try:
        faster_whisper_stt = FasterWhisperSTT(model_size="base", device="auto", compute_type="int8")
        transcription_result, latency = framework.measure_latency(
            faster_whisper_stt.transcribe_audio, framework.audio_data, framework.samplerate, language="en"
        )
        predicted_transcript = transcription_result[0]
        print(f"Faster-Whisper Latency: {latency:.4f} seconds")
        stt_results = framework.evaluate_stt(predicted_transcript)
        if stt_results:
            print(f"Faster-Whisper WER: {stt_results['wer']:.4f}")

    except Exception as e:
        print(f"Error testing Faster-Whisper: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_stt_tests()
