import os
import torch
import time
import numpy as np
import soundfile as sf
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("--- OpenVoice V2 Slovak Benchmark ---")
    
    # Paths
    base_dir = "/Users/yegor/Documents/STU/BP"
    ckpt_converter = os.path.join(base_dir, 'models/openvoice_v2/checkpoints_v2/converter')
    output_dir = os.path.join(base_dir, 'test_output/bench/openvoice')
    os.makedirs(output_dir, exist_ok=True)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Some OpenVoice components might not be stable on MPS, fallback to CPU if needed
    # device = "cpu"
    logging.info(f"Using device: {device}")
    
    # 1. Initialize ToneColorConverter
    logging.info("Initializing ToneColorConverter...")
    start_init = time.perf_counter()
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    init_time = time.perf_counter() - start_init
    logging.info(f"ToneColorConverter initialized in {init_time:.2f}s")
    
    # 2. Extract Target Speaker Embedding
    # We use the existing reference speaker from the project
    reference_speaker = os.path.join(base_dir, 'test/My test speech_xtts_speaker_clean.wav')
    if not os.path.exists(reference_speaker):
        logging.error(f"Reference speaker file not found: {reference_speaker}")
        return
        
    logging.info("Extracting target speaker embedding...")
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)
    
    # 3. Source Audio (Piper Slovak Proxy)
    # Strategy: Use an already generated Piper sample OR generate a new one
    # For now, let's assume we use one of the existing bench samples
    source_path = os.path.join(base_dir, 'test_output/bench/piper_conversational_slovak.wav')
    if not os.path.exists(source_path):
        logging.error(f"Source Piper audio not found at {source_path}. Please run Piper benchmark first.")
        return
        
    logging.info(f"Using source audio: {source_path}")
    
    # 4. Extract Source Speaker Embedding (from Piper output)
    logging.info("Extracting source speaker embedding...")
    source_se, _ = se_extractor.get_se(source_path, tone_color_converter, vad=True)
    
    # 5. Tone Color Conversion
    logging.info("Converting tone color...")
    save_path = os.path.join(output_dir, 'piper_openvoice_v2_cloned.wav')
    
    encode_message = "@MyShell"
    start_conv = time.perf_counter()
    
    tone_color_converter.convert(
        audio_src_path=source_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message
    )
    
    conv_time = time.perf_counter() - start_conv
    logging.info(f"Tone color conversion complete in {conv_time:.2f}s")
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    main()
