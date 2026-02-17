import os
import torch
import time
import numpy as np
import soundfile as sf
from melo.api import TTS
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("--- MeloTTS Speed Benchmark (English) ---")
    
    # Paths
    base_dir = "/Users/yegor/Documents/STU/BP"
    output_dir = os.path.join(base_dir, 'test_output/bench/melotts')
    os.makedirs(output_dir, exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Text to synthesize
    text = "The quick brown fox jumps over the lazy dog."
    
    # 1. Initialize MeloTTS
    logging.info("Initializing MeloTTS (English)...")
    start_init = time.perf_counter()
    # Languages: EN, ES, FR, ZH, JP, KR
    model = TTS(language='EN', device=device)
    init_time = time.perf_counter() - start_init
    logging.info(f"MeloTTS initialized in {init_time:.2f}s")
    
    # 2. Synthesis
    logging.info(f"Synthesizing: '{text}'")
    speaker_ids = model.hps.data.spk2id
    # Use the first speaker
    speaker_id = list(speaker_ids.values())[0]
    
    # Warmup
    model.tts_to_file("Warmup", speaker_id, os.path.join(output_dir, 'warmup.wav'), speed=1.0)
    
    start_syn = time.perf_counter()
    save_path = os.path.join(output_dir, 'melotts_en.wav')
    model.tts_to_file(text, speaker_id, save_path, speed=1.0)
    syn_time = time.perf_counter() - start_syn
    
    # Get audio duration
    data, samplerate = sf.read(save_path)
    duration = len(data) / samplerate
    rtf = syn_time / duration
    
    logging.info(f"Synthesis complete in {syn_time:.2f}s for {duration:.2f}s audio. RTF: {rtf:.2f}")
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    main()
