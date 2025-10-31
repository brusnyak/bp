import os
import requests

PIPER_MODELS_DIR = "backend/tts/piper_models"
PIPER_MODEL_URL_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/cs/cs_CZ/jirka/medium/"
PIPER_MODEL_ONNX = "cs_CZ-jirka-medium.onnx"
PIPER_MODEL_JSON = "cs_CZ-jirka-medium.onnx.json"

def download_file(url, path):
    print(f"Downloading {url} to {path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {path}")

def main():
    os.makedirs(PIPER_MODELS_DIR, exist_ok=True)

    onnx_path = os.path.join(PIPER_MODELS_DIR, PIPER_MODEL_ONNX)
    json_path = os.path.join(PIPER_MODELS_DIR, PIPER_MODEL_JSON)

    if not os.path.exists(onnx_path):
        download_file(PIPER_MODEL_URL_BASE + PIPER_MODEL_ONNX, onnx_path)
    else:
        print(f"{PIPER_MODEL_ONNX} already exists. Skipping download.")

    if not os.path.exists(json_path):
        download_file(PIPER_MODEL_URL_BASE + PIPER_MODEL_JSON, json_path)
    else:
        print(f"{PIPER_MODEL_JSON} already exists. Skipping download.")

    print("Piper TTS model download process complete.")

if __name__ == "__main__":
    main()
