
import time
import logging
import shutil
import numpy as np
import soundfile as sf
from backend.tts.coqui_tts import CoquiTTS
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Constants
OUTPUT_DIR = Path("test_output/quality_samples")
SPEAKER_WAV = Path("test/Hello.wav")

def ensure_output_dir():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_voice_quality():
    print("\n" + "="*80)
    print("COQUI TTS VOICE QUALITY TEST")
    print("="*80)
    
    if not SPEAKER_WAV.exists():
        print(f"❌ Reference file not found: {SPEAKER_WAV}")
        return

    # Copy reference file for easy comparison
    shutil.copy(SPEAKER_WAV, OUTPUT_DIR / "reference_original.wav")
    print(f"Saved reference audio to {OUTPUT_DIR}/reference_original.wav")

    # Initialize
    print(">>> Initializing CoquiTTS (Optimized)...", flush=True)
    tts = CoquiTTS(device="cpu", enable_warmup=True)
    
    test_cases = [
        {
            "lang": "en",
            "text": "Hello, this is a test of the voice cloning capabilities. I should sound like the original speaker.",
            "filename": "cloned_en.wav"
        },
        {
            "lang": "cs", 
            "text": "Ahoj, tohle je test klonování hlasu. Měl bych znít jako původní mluvčí.",
            "filename": "cloned_cs.wav"
        }
    ]
    
    for case in test_cases:
        print(f"\n>>> Generating {case['lang'].upper()} sample...", flush=True)
        print(f"    Text: \"{case['text']}\"", flush=True)
        start = time.perf_counter()
        
        try:
            # Use streaming to verify it works in this context too, or standard synthesize?
            # Let's use standard synthesize for quality check to ensure full context is considered,
            # although streaming is what will be used live. 
            # Actually, we should test what we use. Let's use streaming and concatenate.
            
            stream = tts.synthesize_stream(
                text=case['text'],
                language=case['lang'],
                speaker_wav_path=str(SPEAKER_WAV),
                use_cache=True
            )
            
            audio_chunks = []
            for chunk in stream:
                audio_chunks.append(chunk)
                
            full_audio = np.concatenate(audio_chunks)
            duration = time.perf_counter() - start
            
            output_path = OUTPUT_DIR / case['filename']
            sf.write(output_path, full_audio, 24000)
            
            audio_duration = len(full_audio) / 24000
            print(f"✅ Generated {case['filename']} in {duration:.2f}s (audio duration: {audio_duration:.2f}s)")
            
        except Exception as e:
            print(f"❌ Failed to generate {case['lang']}: {e}")

    print("\n" + "="*80)
    print(f"Test complete. Samples saved to: {OUTPUT_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    ensure_output_dir()
    test_voice_quality()
