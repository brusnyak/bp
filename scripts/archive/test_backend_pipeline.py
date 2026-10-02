#!/usr/bin/env python3
"""
Quick backend pipeline test script.
Tests STT -> MT -> TTS without the UI/WebSocket complexity.
"""

import sys
import os
import asyncio
import numpy as np
import soundfile as sf
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.piper_tts import PiperTTS

async def test_pipeline(audio_file_path: str, source_lang: str = "en", target_lang: str = "sk"):
    """
    Test the full pipeline with a WAV file.
    
    Args:
        audio_file_path: Path to a 16kHz mono WAV file
        source_lang: Source language code
        target_lang: Target language code
    """
    print(f"\n{'='*60}")
    print(f"BACKEND PIPELINE TEST: {source_lang} -> {target_lang}")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract base name for output files
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    
    # Check for ground truth files
    audio_dir = os.path.dirname(audio_file_path)
    transcript_file = os.path.join(audio_dir, f"{base_name}_transcript.txt")
    translation_file = os.path.join(audio_dir, f"{base_name}_translation.txt")
    
    ground_truth_transcript = None
    ground_truth_translation = None
    
    if os.path.exists(transcript_file):
        with open(transcript_file, 'r') as f:
            ground_truth_transcript = f.read().strip()
    
    if os.path.exists(translation_file):
        with open(translation_file, 'r') as f:
            ground_truth_translation = f.read().strip()
    
    # Load audio
    print(f"[1/5] Loading audio from: {audio_file_path}")
    audio_np, sr = sf.read(audio_file_path)
    print(f"      Audio: {len(audio_np)/sr:.2f}s, {sr}Hz, {audio_np.dtype}")
    
    if sr != 16000:
        print(f"      WARNING: Audio should be 16kHz, got {sr}Hz")
        print(f"      This may cause STT hallucinations!")
    
    # Calculate RMS
    rms = np.sqrt(np.mean(audio_np**2))
    print(f"      RMS: {rms:.4f} (should be > 0.02 for VAD threshold)")
    
    if ground_truth_transcript:
        print(f"\n      Ground truth transcript: '{ground_truth_transcript}'")
    if ground_truth_translation:
        print(f"      Ground truth translation: '{ground_truth_translation}'")
    
    # Initialize models
    print(f"\n[2/5] Initializing models...")
    
    # STT
    print("      - Loading FasterWhisperSTT (base)...")
    stt_start = time.time()
    stt_model = FasterWhisperSTT(model_size="base", compute_type="int8")
    print(f"        Done in {time.time() - stt_start:.2f}s")
    
    # MT
    print(f"      - Loading CTranslate2MT ({source_lang}-{target_lang})...")
    mt_start = time.time()
    mt_model_path = f"ct2_models/Helsinki-NLP--opus-mt-{source_lang}-{target_lang}"
    if not os.path.exists(mt_model_path):
        print(f"        ERROR: MT model not found at {mt_model_path}")
        print(f"        Run: python backend/mt/convert_opus_mt_to_ct2.py --model_name Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
        return
    mt_model = CTranslate2MT(model_path=mt_model_path, device="cpu")
    print(f"        Done in {time.time() - mt_start:.2f}s")
    
    # TTS  
    print("      - Loading PiperTTS (cs_CZ-jirka-medium)...")
    tts_start = time.time()
    tts_model = PiperTTS(model_id="cs_CZ-jirka-medium", device="mps")
    print(f"        Done in {time.time() - tts_start:.2f}s")
    
    # Run pipeline
    print(f"\n[3/5] Running pipeline...")
    
    # STT
    print("      [STT] Transcribing...")
    stt_proc_start = time.time()
    segments, stt_time, detected_lang = stt_model.transcribe_audio(
        audio_np, sr, language=source_lang, vad_filter=True  # Enable VAD to reduce hallucinations
    )
    transcribed_text = " ".join([s.text for s in segments]).strip()
    stt_proc_time = time.time() - stt_proc_start
    print(f"        ✓ Transcribed in {stt_proc_time:.2f}s")
    print(f"        STT Output: '{transcribed_text}'")
    print(f"        Detected lang: {detected_lang}")
    
    #Verify against ground truth
    if ground_truth_transcript:
        if transcribed_text.lower() == ground_truth_transcript.lower():
            print(f"        ✅ STT MATCHES ground truth perfectly!")
        else:
            print(f"        ⚠️  STT MISMATCH:")
            print(f"            Expected: '{ground_truth_transcript}'")
            print(f"            Got:      '{transcribed_text}'")
            print(f"        This could be a Whisper hallucination or audio quality issue!")
    
    if not transcribed_text:
        print("\n        ❌ ERROR: No transcription detected!")
        print("        This means STT is not picking up speech.")
        print("        Possible causes:")
        print("          1. Audio too quiet (check RMS above)")
        print("          2. Audio format issue (should be 16kHz mono)")
        print("          3. VAD filtering too aggressive")
        return
    
    # MT
    print(f"\n      [MT] Translating...")
    mt_proc_start = time.time()
    translated_text, _ = mt_model.translate(transcribed_text, source_lang, target_lang)
    mt_proc_time = time.time() - mt_proc_start
    print(f"        ✓ Translated in {mt_proc_time:.2f}s")
    print(f"        MT Output: '{translated_text}'")
    
    # Verify against ground truth
    if ground_truth_translation:
        if translated_text.lower() == ground_truth_translation.lower():
            print(f"        ✅ MT MATCHES ground truth perfectly!")
        else:
            print(f"        ⚠️  MT difference (may be acceptable):")
            print(f"            Expected: '{ground_truth_translation}'")
            print(f"            Got:      '{translated_text}'")
    
    # TTS
    print(f"\n      [TTS] Synthesizing...")
    tts_proc_start = time.time()
    output_audio_path = os.path.join(output_dir, f"{base_name}_output.wav")
    success = tts_model.synthesize(translated_text, output_audio_path)
    tts_proc_time = time.time() - tts_proc_start
    
    if success:
        print(f"        ✓ Synthesized in {tts_proc_time:.2f}s")
        print(f"        Audio saved to: {output_audio_path}")
        
        # Check output audio (skip reading if there's an issue)
        try:
            output_audio, output_sr = sf.read(output_audio_path)
            print(f"        Output: {len(output_audio)/output_sr:.2f}s, {output_sr}Hz")
        except Exception as e:
            print(f"        Note: Could not read output audio ({e}), but file was created")
    else:
        print(f"        ❌ TTS failed!")
        return
    
    # Save text outputs
    print(f"\n[4/5] Saving text outputs...")
    transcript_output = os.path.join(output_dir, f"{base_name}_stt_output.txt")
    translation_output = os.path.join(output_dir, f"{base_name}_mt_output.txt")
    
    with open(transcript_output, 'w') as f:
        f.write(transcribed_text)
    with open(translation_output, 'w') as f:
        f.write(translated_text)
    
    print(f"      Saved transcription to: {transcript_output}")
    print(f"      Saved translation to: {translation_output}")
    
    # Summary
    print(f"\n[5/5] Summary")
    print(f"      Total processing time: {stt_proc_time + mt_proc_time + tts_proc_time:.2f}s")
    print(f"        - STT: {stt_proc_time:.2f}s")
    print(f"        - MT:  {mt_proc_time:.2f}s")
    print(f"        - TTS: {tts_proc_time:.2f}s")
    
    # Overall status
    all_correct = True
    if ground_truth_transcript and transcribed_text.lower() != ground_truth_transcript.lower():
        all_correct = False
    
    if all_correct:
        print(f"\n      ✅ PIPELINE TEST SUCCESSFUL!\n")
    else:
        print(f"\n      ⚠️  PIPELINE COMPLETED WITH WARNINGS\n")
        print(f"      Check STT output above for hallucination issues.")
    
    print(f"      Play output: afplay {output_audio_path}")
    print(f"      All outputs saved to: {output_dir}/")
    print(f"{'='*60}\n")


def create_test_audio():
    """Create a simple test audio file if none exists."""
    test_file = "/tmp/test_hello.wav"
    
    print("No test audio provided. Creating a simple test tone...")
    print("(This won't contain speech, but will test the audio path)")
    
    # Create 2 seconds of 440Hz tone (A4)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # amplitude 0.3
    
    sf.write(test_file, audio, sr)
    print(f"Created test audio: {test_file}")
    return test_file


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Try to use existing test audio
        test_audio_candidates = [
            "test/Hello.wav",
            "test/Can you hear me_.wav",
            "BP xtts/tests/My test speech_xtts_speaker.wav"
        ]
        
        audio_file = None
        for candidate in test_audio_candidates:
            if os.path.exists(candidate):
                audio_file = candidate
                break
        
        if not audio_file:
            print("\nNo test audio file found.")
            print("Usage: python test_backend_pipeline.py <path_to_wav_file>")
            print("\nOr record a quick test:")
            print("  sox -d -r 16000 -c 1 test.wav trim 0 5  # Record 5 seconds")
            print("  python test_backend_pipeline.py test.wav")
            sys.exit(1)
    
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    asyncio.run(test_pipeline(audio_file))
