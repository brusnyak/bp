import sys
import os
import numpy as np
import soundfile as sf

# Add the path to the backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.tts.piper_tts import PiperTTS

def run_standalone_piper_test():
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)

    test_phrase_sk = "Dobrý deň, toto je test môjho hlasu na účely klonovania."
    test_phrase_cs = "Dobrý den, toto je test mého hlasu pro účely klonování." # Czech translation of the phrase

    # Test SK voice
    print("\n--- Testing SK_SK-lili-medium directly ---")
    try:
        piper_sk = PiperTTS(model_id="sk_SK-lili-medium", device="cpu")
        audio_sk, sr_sk, time_sk = piper_sk.synthesize(test_phrase_sk, language="sk")
        output_path_sk = os.path.join(output_dir, "standalone_piper_sk_output.wav")
        sf.write(output_path_sk, audio_sk, sr_sk)
        print(f"Saved SK synthesized audio to {output_path_sk}")
    except Exception as e:
        print(f"Error testing SK voice: {e}")

    # Test CS voice
    print("\n--- Testing CS_CZ-jirka-medium directly ---")
    try:
        piper_cs = PiperTTS(model_id="cs_CZ-jirka-medium", device="cpu")
        audio_cs, sr_cs, time_cs = piper_cs.synthesize(test_phrase_cs, language="cs")
        output_path_cs = os.path.join(output_dir, "standalone_piper_cs_output.wav")
        sf.write(output_path_cs, audio_cs, sr_cs)
        print(f"Saved CS synthesized audio to {output_path_cs}")
    except Exception as e:
        print(f"Error testing CS voice: {e}")

if __name__ == "__main__":
    run_standalone_piper_test()
