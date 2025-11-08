import os
import sys

# Add the current working directory to sys.path for absolute imports
sys.path.append(os.getcwd())

from backend.main import PIPER_MODEL_MAPPING
from backend.tts.piper_tts import PiperTTS

def check_piper_model_paths():
    print(f"Current working directory: {os.getcwd()}")
    
    test_model_ids = ["sk_SK-lili-medium", "en_US-ryan-medium", "cs_CZ-jirka-medium"]
    base_model_dir = os.path.join("backend", "tts", "piper_models")

    for model_id in test_model_ids:
        onnx_model_path = os.path.join(base_model_dir, f"{model_id}.onnx")
        json_config_path = os.path.join(base_model_dir, f"{model_id}.onnx.json")

        print(f"\n--- Checking model: {model_id} ---")
        print(f"Absolute ONNX path: {os.path.abspath(onnx_model_path)}")
        print(f"Absolute JSON path: {os.path.abspath(json_config_path)}")
        print(f"os.path.exists(ONNX): {os.path.exists(onnx_model_path)}")
        print(f"os.path.exists(JSON): {os.path.exists(json_config_path)}")

        if os.path.exists(onnx_model_path) and os.path.exists(json_config_path):
            print(f"Files for {model_id} found. Attempting to load PiperVoice...")
            try:
                # Attempt to load the model directly
                _ = PiperTTS(model_id=model_id, device="cpu") # Force CPU for this test to avoid MPS issues
                print(f"SUCCESS: PiperVoice for {model_id} loaded successfully.")
            except Exception as e:
                print(f"FAILURE: Failed to load PiperVoice for {model_id}: {e}")
        else:
            print(f"FAILURE: Files for {model_id} NOT found by os.path.exists().")

if __name__ == "__main__":
    check_piper_model_paths()
