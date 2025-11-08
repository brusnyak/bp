import os
import requests
import sys

PIPER_MODELS_DIR = os.path.join("backend", "tts", "piper_models")
HUGGINGFACE_PIPER_VOICES_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

def download_file(url, path):
    """Downloads a file from a given URL to a specified path."""
    print(f"Downloading {url} to {path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download {url}: {e}")
        return False

def get_piper_model_urls(model_id):
    """Constructs Hugging Face download URLs for a given Piper model ID."""
    # Example model_id: "sk_SK-lili-medium"
    # URL structure: https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang}_{country}/{speaker}/{quality}/{model_id}.onnx
    parts = model_id.split('-')
    if len(parts) < 3:
        raise ValueError(f"Invalid Piper model ID format: {model_id}. Expected format like 'lang_COUNTRY-speaker-quality'.")
    
    lang_code = parts[0].split('_')[0] # e.g., 'sk' from 'sk_SK'
    lang_country_code = parts[0] # e.g., 'sk_SK'
    speaker_name = parts[1] # e.g., 'lili'
    quality = parts[2] # e.g., 'medium'

    base_url_path = f"{HUGGINGFACE_PIPER_VOICES_BASE}/{lang_code}/{lang_country_code}/{speaker_name}/{quality}"
    
    onnx_url = f"{base_url_path}/{model_id}.onnx"
    json_url = f"{base_url_path}/{model_id}.onnx.json"
    
    return onnx_url, json_url

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_piper_models.py <model_id>")
        print("Example: python download_piper_models.py sk_SK-lili-medium")
        sys.exit(1)

    model_id = sys.argv[1]
    os.makedirs(PIPER_MODELS_DIR, exist_ok=True)

    onnx_url, json_url = get_piper_model_urls(model_id)
    
    onnx_path = os.path.join(PIPER_MODELS_DIR, f"{model_id}.onnx")
    json_path = os.path.join(PIPER_MODELS_DIR, f"{model_id}.onnx.json")

    downloaded_onnx = False
    if not os.path.exists(onnx_path):
        downloaded_onnx = download_file(onnx_url, onnx_path)
    else:
        print(f"{model_id}.onnx already exists. Skipping download.")
        downloaded_onnx = True # Assume it's there if it exists

    downloaded_json = False
    if not os.path.exists(json_path):
        downloaded_json = download_file(json_url, json_path)
    else:
        print(f"{model_id}.onnx.json already exists. Skipping download.")
        downloaded_json = True # Assume it's there if it exists

    if downloaded_onnx and downloaded_json:
        print(f"Piper TTS model '{model_id}' download process complete.")
    else:
        print(f"WARNING: Piper TTS model '{model_id}' download incomplete or failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
