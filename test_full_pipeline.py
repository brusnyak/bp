#!/usr/bin/env python3
"""
Full S2S (Speech-to-Speech) Pipeline Test
Tests the complete STT -> MT -> TTS pipeline with speaker voice cloning
"""

import time
import sys
import os
import numpy as np
import soundfile as sf

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Test audio files from speaker_voices directory
TEST_AUDIO_FILES = [
    ("speaker_voices/hello.wav", "Hello, this is a test of my voice for cloning purposes.", "Test 1: hello.wav"),
    ("speaker_voices/Can you hear me_.wav", "Hey laddy can you hear me well?", "Test 2: Can you hear me_.wav"),
    ("speaker_voices/My test speech_xtts_speaker_clean.wav", "In this experiment the system converts spoken English into text translate it into Slova and then send the sizes it back into speech", "Test 3: My test speech_xtts_speaker_clean.wav"),
]

# Output directory
OUTPUT_DIR = "test_output/s2s_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_stt(audio_path: str):
    """Test Speech-to-Text"""
    print("\n" + "=" * 60)
    print("=== STT (Speech-to-Text) ===")
    print("=" * 60)
    
    from stt.faster_whisper_stt import FasterWhisperSTT
    
    # Initialize model
    print("Loading FasterWhisper STT model (base)...")
    start_init = time.perf_counter()
    stt_model = FasterWhisperSTT(model_size="base", compute_type="int8")
    init_time = time.perf_counter() - start_init
    print(f"Model initialization time: {init_time:.4f}s")
    
    # Load audio
    print(f"Loading audio from: {audio_path}")
    audio, sr = sf.read(audio_path)
    
    # Resample if needed
    if sr != 16000:
        print(f"Resampling from {sr} to 16000 Hz...")
        from scipy import signal
        num_samples = int(len(audio) * 16000 / sr)
        audio = signal.resample(audio, num_samples)
        sr = 16000
    
    # Transcribe
    print("Transcribing...")
    start_time = time.perf_counter()
    segments, stt_time, detected_lang = stt_model.transcribe_audio(
        audio.astype(np.float32), 
        sr, 
        language="en",
        vad_filter=False
    )
    transcribe_latency = time.perf_counter() - start_time
    
    transcribed_text = " ".join([s.text for s in segments]) if segments else ""
    
    print(f"\nSTT Results:")
    print(f"  Transcribed text: '{transcribed_text}'")
    print(f"  Detected language: {detected_lang}")
    print(f"  Processing time: {transcribe_latency:.4f}s")
    print(f"  Audio duration: {len(audio)/sr:.2f}s")
    print(f"  RTF: {transcribe_latency / (len(audio)/sr):.2f}x")
    
    return transcribed_text, transcribe_latency


def test_mt(text: str, source_lang: str = "en", target_lang: str = "sk"):
    """Test Machine Translation"""
    print("\n" + "=" * 60)
    print("=== MT (Machine Translation) ===")
    print("=" * 60)
    
    from mt.ctranslate2_mt import CTranslate2MT
    
    # Initialize model
    model_path = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    print(f"Loading MT model: {model_path}...")
    start_init = time.perf_counter()
    mt_model = CTranslate2MT(model_path=model_path, device="auto")
    init_time = time.perf_counter() - start_init
    print(f"Model initialization time: {init_time:.4f}s")
    
    # Translate
    print(f"Translating: '{text}'")
    start_time = time.perf_counter()
    translated_text, mt_latency = mt_model.translate(text, source_lang, target_lang)
    translate_time = time.perf_counter() - start_time
    
    print(f"\nMT Results:")
    print(f"  Translated text: '{translated_text}'")
    print(f"  Processing time: {translate_time:.4f}s")
    
    return translated_text, translate_time


def test_tts(text: str, speaker_wav: str, tts_model_choice: str = "piper"):
    """Test Text-to-Speech"""
    print("\n" + "=" * 60)
    print("=== TTS (Text-to-Speech) ===")
    print("=" * 60)
    
    if tts_model_choice == "piper":
        from tts.piper_tts import PiperTTS
        
        print("Loading Piper TTS model (cs_CZ-jirka-medium)...")
        start_init = time.perf_counter()
        tts_model = PiperTTS(model_id="cs_CZ-jirka-medium", device="cpu")
        init_time = time.perf_counter() - start_init
        print(f"Model initialization time: {init_time:.4f}s")
        
        # Synthesize (without voice cloning)
        print(f"Synthesizing: '{text}'")
        start_time = time.perf_counter()
        audio, sr, tts_latency = tts_model.synthesize(text, language="sk")
        synthesis_time = time.perf_counter() - start_time
        
        print(f"\nTTS Results (Piper - no cloning):")
        print(f"  Output duration: {len(audio)/sr:.2f}s")
        print(f"  Processing time: {synthesis_time:.4f}s")
        
    elif tts_model_choice == "xtts":
        from tts.coqui_tts import CoquiTTS
        
        print("Loading Coqui TTS model (XTTS v2)...")
        start_init = time.perf_counter()
        tts_model = CoquiTTS(device="cpu", enable_warmup=False)
        init_time = time.perf_counter() - start_init
        print(f"Model initialization time: {init_time:.4f}s")
        
        # Synthesize with voice cloning
        print(f"Synthesizing with voice cloning: '{text}'")
        print(f"Using speaker: {speaker_wav}")
        start_time = time.perf_counter()
        audio, sr, tts_latency = tts_model.synthesize(
            text, 
            language="cs",  # Use Czech as proxy for Slovak
            speaker_wav_path=speaker_wav,
            use_cache=True
        )
        synthesis_time = time.perf_counter() - start_time
        
        print(f"\nTTS Results (XTTS with cloning):")
        print(f"  Output duration: {len(audio)/sr:.2f}s")
        print(f"  Processing time: {synthesis_time:.4f}s")
    
    return audio, sr, synthesis_time


def run_full_pipeline_test(audio_path: str, expected_text: str = None, tts_model: str = "piper"):
    """Run the full S2S pipeline test"""
    print("\n" + "=" * 80)
    print(f"FULL PIPELINE TEST: {os.path.basename(audio_path)}")
    print("=" * 80)
    
    total_start = time.perf_counter()
    
    # Step 1: STT
    stt_text, stt_time = test_stt(audio_path)
    if not stt_text and expected_text:
        stt_text = expected_text
        print(f"  Using expected text: '{stt_text}'")
    
    # Step 2: MT
    mt_text, mt_time = test_mt(stt_text, source_lang="en", target_lang="sk")
    
    # Step 3: TTS
    # For speaker cloning, use the audio file itself as reference
    tts_audio, tts_sr, tts_time = test_tts(mt_text, audio_path, tts_model)
    
    # Total time
    total_time = time.perf_counter() - total_start
    
    # Save output
    output_filename = os.path.basename(audio_path).replace(".wav", f"_output_{tts_model}.wav")
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    sf.write(output_path, tts_audio, tts_sr)
    print(f"\n✅ Output saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"  STT time: {stt_time:.4f}s")
    print(f"  MT time:  {mt_time:.4f}s")
    print(f"  TTS time: {tts_time:.4f}s")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:    {total_time:.4f}s")
    print(f"\nTarget: <1.5s end-to-end latency")
    if total_time < 1.5:
        print("✅ TARGET ACHIEVED!")
    else:
        print(f"⚠️  Need to reduce by {total_time - 1.5:.4f}s")
    
    return {
        "stt_time": stt_time,
        "mt_time": mt_time,
        "tts_time": tts_time,
        "total_time": total_time,
        "output_path": output_path
    }


def main():
    print("BP Speech-to-Speech Translation - Full Pipeline Benchmark")
    print("=" * 80)
    
    results = []
    
    # Test with different audio files and TTS models
    for audio_path, expected_text, test_name in TEST_AUDIO_FILES:
        if not os.path.exists(audio_path):
            print(f"\n⚠️  Audio file not found: {audio_path}")
            continue
            
        print(f"\n{'#' * 80}")
        print(f"# {test_name}")
        print(f"# Audio: {audio_path}")
        print(f"{'#' * 80}")
        
        # Test with Piper (fast, no cloning)
        print("\n" + "=" * 60)
        print("Testing with PIPER TTS (no voice cloning)")
        print("=" * 60)
        result_piper = run_full_pipeline_test(audio_path, expected_text, "piper")
        result_piper["model"] = "piper"
        results.append(result_piper)
    
    # Summary
    print("\n\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)
    for r in results:
        print(f"\n{r['model'].upper()} model:")
        print(f"  STT:    {r['stt_time']:.4f}s")
        print(f"  MT:     {r['mt_time']:.4f}s")
        print(f"  TTS:    {r['tts_time']:.4f}s")
        print(f"  TOTAL:  {r['total_time']:.4f}s")
        print(f"  Output: {r['output_path']}")
    
    # Calculate averages
    if results:
        avg_total = sum(r['total_time'] for r in results) / len(results)
        print(f"\n{'=' * 40}")
        print(f"AVERAGE TOTAL LATENCY: {avg_total:.4f}s")
        print(f"{'=' * 40}")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()