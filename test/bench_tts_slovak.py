import os
import time
import torch
import numpy as np
import soundfile as sf
import sys

# Add backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from backend.tts.piper_tts import PiperTTS
    from backend.tts.coqui_tts import CoquiTTS
except ImportError as e:
    print(f"Error importing TTS models: {e}")
    sys.exit(1)

def benchmark_tts(model, text, name, output_dir="test_output/bench"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Benchmarking {name} ---")
    
    start_time = time.time()
    try:
        # Both models return (audio, sample_rate, synthesis_time)
        if "Coqui" in name:
            speaker_wav = "test/My test speech_xtts_speaker_clean.wav"
            # Using 'cs' (Czech) as a proxy for Slovak (sk) because XTTS v2 doesn't natively support 'sk'
            audio, sr, tts_time = model.synthesize(text, language="cs", speaker_wav_path=speaker_wav)
        else:
            audio, sr, tts_time = model.synthesize(text)
        
        duration = time.time() - start_time
        print(f"Total loop time: {duration:.4f}s")
        print(f"Internal synthesis time: {tts_time:.4f}s")
        
        output_path = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_slovak.wav")
        sf.write(output_path, audio, sr)
        print(f"Saved to {output_path}")
        
        return tts_time
    except Exception as e:
        print(f"Error benchmarking {name}: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    test_texts = {
        "technical": "V tomto experimente systém konvertuje hovorenú angličtinu do slovenčiny v reálnom čase.",
        "conversational": "Ahoj, ako sa dnes máš? Dúfam, že tvoj deň prebieha úspešne a všetko ide podľa tvojich predstáv.",
        "poetic": "Krvavé sonety sú zbierkou básní, v ktorých autor vyjadruje svoj odpor k vojne a utrpeniu."
    }
    
    # 1. Piper (Baseline)
    print("Initializing Piper...")
    piper = PiperTTS(model_id="cs_CZ-jirka-medium")
    
    # 2. Coqui TTS (formerly XTTS v2)
    print("Initializing CoquiTTS...")
    coqui = CoquiTTS()
    
    results = {}
    
    for category, text in test_texts.items():
        print(f"\n--- Testing {category} ---")
        # Test Piper
        p_dur = benchmark_tts(piper, text, f"Piper_{category}")
        # Test Coqui
        c_dur = benchmark_tts(coqui, text, f"CoquiTTS_{category}")
        
    print("\nBenchmark generation finished. Samples available in test_output/bench/")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
