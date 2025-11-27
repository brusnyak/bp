
import soundfile as sf
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

SAMPLES_DIR = Path("test_output/quality_samples")

def analyze_end_noise(filename):
    path = SAMPLES_DIR / filename
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    data, samplerate = sf.read(path)
    
    # Check last 100ms
    num_samples = int(0.1 * samplerate)
    if len(data) < num_samples:
        print(f"⚠️ File {filename} is too short for analysis.")
        return

    last_chunk = data[-num_samples:]
    max_amp = np.max(np.abs(last_chunk))
    rms = np.sqrt(np.mean(last_chunk**2))
    
    print(f"Analysis for {filename}:")
    print(f"  Duration: {len(data)/samplerate:.2f}s")
    print(f"  Last 100ms Max Amplitude: {max_amp:.4f}")
    print(f"  Last 100ms RMS Power: {rms:.4f}")
    
    if max_amp > 0.05: # Threshold for "noise"
        print("  ⚠️ POTENTIAL ARTIFACT DETECTED (High amplitude at end)")
    else:
        print("  ✅ End seems clean")
    print("-" * 40)

def main():
    print("\n" + "="*80)
    print("AUDIO ARTIFACT ANALYSIS")
    print("="*80)
    
    files = ["cloned_en.wav", "cloned_sk_cs.wav"]
    for f in files:
        analyze_end_noise(f)

if __name__ == "__main__":
    main()
