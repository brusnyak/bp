import os
import sys
import numpy as np
import soundfile as sf

# Add the current working directory to sys.path for absolute imports
sys.path.append(os.getcwd())

from test.model_testing_framework import ModelTestFramework
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT

def run_en_to_sk_pipeline_tests():
    # Use existing English audio and its Slovak translation for testing the pipeline
    audio_path = "test/My test speech_xtts_speaker_clean.wav"
    transcript_path = "test/My test speech transcript.txt" # English reference transcript
    translation_path = "test/My test speech translation.txt" # Slovak reference translation

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found at {transcript_path}")
        return
    if not os.path.exists(translation_path):
        print(f"Error: Translation file not found at {translation_path}")
        return

    framework = ModelTestFramework(
        audio_path,
        transcript_path=transcript_path,
        translation_path=translation_path
    )

    audio_data, samplerate = framework.audio_data, framework.samplerate

    print("\n--- Testing Faster-Whisper STT (English Input) ---")
    try:
        faster_whisper_stt = FasterWhisperSTT(model_size="large-v3", device="auto", compute_type="int8")
        
        # Transcribe in English
        transcription_segments, latency = framework.measure_latency(
            faster_whisper_stt.transcribe_audio, audio_data, samplerate, language="en"
        )
        predicted_transcript = " ".join([s.text for s in transcription_segments])
        print(f"Faster-Whisper (English) Latency: {latency:.4f} seconds")
        stt_results = framework.evaluate_stt(predicted_transcript)
        if stt_results:
            print(f"Faster-Whisper (English) WER: {stt_results['wer']:.4f}")

    except Exception as e:
        print(f"Error testing Faster-Whisper with English input: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing CTranslate2 MT (English to Slovak) ---")
    try:
        mt_model_name = "Helsinki-NLP/opus-mt-en-sk"
        mt_model_path = f"ct2_models/{mt_model_name.replace('/', '--')}"
        ctranslate2_mt = CTranslate2MT(model_path=mt_model_path, device="auto")

        # Use the predicted STT transcript as input for MT evaluation
        input_text_for_mt = predicted_transcript
        
        if input_text_for_mt:
            translation_result, latency = framework.measure_latency(
                ctranslate2_mt.translate, input_text_for_mt, "en", "sk"
            )
            predicted_translation = translation_result
            print(f"CTranslate2 MT (en-sk) Latency: {latency:.4f} seconds")
            mt_results = framework.evaluate_mt(predicted_translation)
            if mt_results:
                print(f"CTranslate2 MT (en-sk) BLEU: {mt_results['bleu']:.2f}")
                print(f"CTranslate2 MT (en-sk) METEOR: {mt_results['meteor']:.4f}")
        else:
            print("Skipping MT test: No STT transcript available.")

    except Exception as e:
        print(f"Error testing CTranslate2 MT with Slovak input: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_en_to_sk_pipeline_tests()
