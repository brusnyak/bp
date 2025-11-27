
import time
import logging
import numpy as np
import soundfile as sf
from backend.tts.coqui_tts import CoquiTTS
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Constants
TEST_OUTPUT_DIR = Path("test_output/coqui_streaming")
SPEAKER_WAV = Path("test/Hello.wav")

def ensure_output_dir():
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_streaming_performance():
    print("\n" + "="*80)
    print("TESTING COQUI TTS STREAMING PERFORMANCE")
    print("="*80)
    
    # Initialize
    print(">>> Initializing CoquiTTS (with warmup)...", flush=True)
    tts = CoquiTTS(device="cpu", enable_warmup=True)
    
    text = "Hello, this is a test of the streaming synthesis capability. It should start playing much faster than the full synthesis."
    
    print(f"\n>>> Starting Streaming Synthesis for: '{text}'", flush=True)
    
    start_time = time.perf_counter()
    first_chunk_time = None
    chunk_count = 0
    total_audio = []
    
    try:
        # Call streaming method
        stream = tts.synthesize_stream(
            text=text,
            language="en",
            speaker_wav_path=str(SPEAKER_WAV),
            use_cache=True
        )
        
        print(">>> Stream generator created. Iterating...", flush=True)
        
        for chunk in stream:
            current_time = time.perf_counter()
            
            if chunk_count == 0:
                first_chunk_time = current_time - start_time
                print(f"✅ FIRST CHUNK RECEIVED in {first_chunk_time:.2f}s", flush=True)
            
            chunk_count += 1
            total_audio.append(chunk)
            
            # Optional: Print dot for every chunk to visualize flow
            print(".", end="", flush=True)
            
        total_time = time.perf_counter() - start_time
        print(f"\n\n>>> Streaming completed in {total_time:.2f}s", flush=True)
        
        # Save full audio
        full_audio = np.concatenate(total_audio)
        output_path = TEST_OUTPUT_DIR / "streaming_result.wav"
        sf.write(output_path, full_audio, 24000)
        print(f"Saved full audio to {output_path}", flush=True)
        
        return {
            "first_chunk_latency": first_chunk_time,
            "total_time": total_time,
            "chunk_count": chunk_count
        }
        
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}", flush=True)
        return None

if __name__ == "__main__":
    ensure_output_dir()
    results = test_streaming_performance()
    
    if results:
        print("\n" + "="*80)
        print("STREAMING RESULTS SUMMARY")
        print("="*80)
        print(f"Time to First Chunk: {results['first_chunk_latency']:.2f}s")
        print(f"Total Generation Time: {results['total_time']:.2f}s")
        print(f"Total Chunks: {results['chunk_count']}")
        
        if results['first_chunk_latency'] < 4.0:
            print("\n✅ TARGET MET: Perceived latency is < 4s!")
        else:
            print("\n❌ TARGET MISSED: First chunk took > 4s.")
        print("="*80 + "\n")
