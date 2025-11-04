import os
import sys
import numpy as np
import soundfile as sf
import torch
import time # Import time for measuring TTFB

# Add the current working directory to sys.path for absolute imports
sys.path.append(os.getcwd())

from model_testing_framework import ModelTestFramework, TimeoutException # Import TimeoutException
from backend.tts.chatterbox_tts import ChatterboxTTSModel
from backend.tts.piper_tts import PiperTTS # Assuming this file exists
from backend.tts.xtts_tts import XTTSv2 # Import XTTSv2
# from backend.tts.bark_tts import BarkTTS # Import BarkTTS
# from backend.tts.melotts_tts import MeloTTS # MeloTTS is incompatible

def run_tts_tests():
    audio_path = "test/My test speech_xtts_speaker_clean.wav"
    voice_training_path = "test/Voice-Training.wav"
    translation_path = "test/My test speech translation.txt" # Use Slovak translation as input for TTS

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    if not os.path.exists(voice_training_path):
        print(f"Error: Voice training file not found at {voice_training_path}")
        return
    if not os.path.exists(translation_path):
        print(f"Error: Translation file not found at {translation_path}")
        return

    framework = ModelTestFramework(
        audio_path=audio_path,
        translation_path=translation_path
    )

    # Use the reference translation as input for TTS
    mt_output_text = framework.reference_translation

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"TTS models will attempt to use device: {device}")

    print("\n--- Testing Chatterbox TTS ---")
    try:
        tts_chatterbox = ChatterboxTTSModel() # Use ChatterboxTTSModel class
        audio_output_path_chatterbox = "output_chatterbox_sk.wav"
        def save_chatterbox_audio(audio_tensor, path):
            sf.write(path, audio_tensor.squeeze().cpu().numpy(), samplerate=24000)

        synthesized_audio_tensor, latency = framework.measure_latency(
            tts_chatterbox.synthesize,
            text=mt_output_text,
            reference_audio=voice_training_path, # For zero-shot cloning
            save_result_func=save_chatterbox_audio,
            output_path=audio_output_path_chatterbox,
            timeout=20 # Set a timeout for Chatterbox TTS
        )
        if synthesized_audio_tensor is not None:
            print(f"Chatterbox TTS Latency: {latency:.4f} seconds")
            framework.evaluate_tts(audio_output_path_chatterbox)
        else:
            print(f"Chatterbox TTS test skipped due to timeout or error.")

    except Exception as e:
        print(f"Error testing Chatterbox TTS: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Piper TTS ---")
    try:
        # Piper is ultra-fast fallback, lightweight, native Czech voices (close to Slovak).
        # Requires manual download of model files.
        # Assuming cs_CZ-jirka-medium.onnx is available in backend/tts/piper_models/
        
        tts_piper = PiperTTS(model_id="cs_CZ-jirka-medium", device=device)

        audio_output_path_piper = "output_piper_sk.wav"
        synthesized_audio, latency = framework.measure_latency(
            tts_piper.synthesize,
            text=mt_output_text,
            output_path=audio_output_path_piper,
            timeout=10 # Set a timeout for Piper TTS
        )
        if synthesized_audio is not None:
            print(f"Piper TTS Latency: {latency:.4f} seconds")
            framework.evaluate_tts(audio_output_path_piper)
        else:
            print(f"Piper TTS test skipped due to timeout or error.")

    except Exception as e:
        print(f"Error testing Piper TTS: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing XTTS v2 ---")
    try:
        tts_xtts = XTTSv2()
        audio_output_path_xtts = "output_xtts_sk.wav"
        
        # Measure latency for streaming, including time to first chunk
        start_time = time.time()
        audio_chunks_generator = tts_xtts.synthesize_stream(
            text=mt_output_text,
            lang="cs", # Specify Czech language, as Slovak is not supported
            speaker_wav=voice_training_path # For zero-shot cloning
        )

        first_chunk = None
        all_chunks = []
        ttfb = -1.0

        try:
            # Set a timeout for the entire streaming process
            signal.signal(signal.SIGALRM, framework.timeout_handler)
            signal.alarm(60) # 60 seconds timeout for XTTS streaming

            for i, chunk in enumerate(audio_chunks_generator):
                if i == 0:
                    ttfb = time.time() - start_time
                    first_chunk = chunk
                all_chunks.append(chunk)
            
            end_time = time.time()
            total_latency = end_time - start_time

            if all_chunks:
                # Save the full audio from all chunks
                full_audio = np.concatenate([chunk.cpu().numpy() for chunk in all_chunks])
                sf.write(audio_output_path_xtts, full_audio, samplerate=24000)
                print(f"XTTS v2 Time to First Chunk (TTFB): {ttfb:.4f} seconds")
                print(f"XTTS v2 Total Latency (streaming): {total_latency:.4f} seconds")
                framework.evaluate_tts(audio_output_path_xtts)
            else:
                print(f"XTTS v2 streaming test produced no audio chunks.")

        except TimeoutException:
            print(f"XTTS v2 streaming test timed out after 60 seconds.")
        except Exception as e:
            print(f"Error during XTTS v2 streaming: {e}")
            import traceback
            traceback.print_exc()
        finally:
            signal.alarm(0) # Disable the alarm

    except Exception as e:
        print(f"Error initializing XTTS v2: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tts_tests()
