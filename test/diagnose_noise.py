#!/usr/bin/env python3
"""
Noise Diagnostic Test for Live Speech Translation
==================================================
This test helps identify the source of spurious "You" transcriptions by:
1. Analyzing existing test audio files
2. Testing VAD sensitivity with different thresholds
3. Analyzing RMS levels during silence vs speech
4. Testing Whisper STT on low-level noise

Usage:
    python test/diagnose_noise.py [audio_file.wav]
    
If no file specified, uses test/Can you hear me_.wav

This will:
- Analyze RMS levels and VAD triggers
- Test if Whisper transcribes background noise
- Provide recommendations for threshold adjustments
"""

import sys
import os
import numpy as np
import soundfile as sf
import webrtcvad
import time
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.audio_utils import load_audio
from backend.stt.faster_whisper_stt import FasterWhisperSTT

# Configuration matching backend/main.py
AUDIO_SAMPLE_RATE = 16000
VAD_FRAME_DURATION = 30  # ms
VAD_AGGRESSIVENESS = 3  # Current setting
SILENCE_RMS_THRESHOLD = 0.005  # Current threshold
PRE_VAD_BUFFER_DURATION = 0.5  # seconds

# Test parameters
RECORDING_DURATION = 10  # seconds
CHUNK_SIZE = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)

def load_audio_file(file_path):
    """Load audio file for analysis"""
    print(f"\n{'='*60}")
    print(f"📁 LOADING AUDIO FILE")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    # Load and resample to 16kHz
    audio_data, sr = load_audio(file_path, target_sr=AUDIO_SAMPLE_RATE)
    
    duration = len(audio_data) / AUDIO_SAMPLE_RATE
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {AUDIO_SAMPLE_RATE}Hz")
    print(f"Samples: {len(audio_data)}")
    print(f"✅ Audio loaded successfully")
    
    return audio_data

def analyze_rms_levels(audio_data):
    """Analyze RMS levels in the recorded audio"""
    print(f"\n{'='*60}")
    print(f"📊 RMS LEVEL ANALYSIS")
    print(f"{'='*60}")
    
    frame_size = CHUNK_SIZE
    rms_values = []
    
    for i in range(0, len(audio_data) - frame_size, frame_size):
        frame = audio_data[i:i + frame_size]
        rms = np.sqrt(np.mean(frame**2))
        rms_values.append(rms)
    
    rms_array = np.array(rms_values)
    
    print(f"RMS Statistics:")
    print(f"  Min:     {np.min(rms_array):.6f}")
    print(f"  Max:     {np.max(rms_array):.6f}")
    print(f"  Mean:    {np.mean(rms_array):.6f}")
    print(f"  Median:  {np.median(rms_array):.6f}")
    print(f"  Std Dev: {np.std(rms_array):.6f}")
    print(f"\nCurrent threshold: {SILENCE_RMS_THRESHOLD}")
    
    # Count frames above threshold
    above_threshold = np.sum(rms_array > SILENCE_RMS_THRESHOLD)
    percentage = (above_threshold / len(rms_array)) * 100
    
    print(f"\nFrames above threshold: {above_threshold}/{len(rms_array)} ({percentage:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if percentage > 50:
        suggested_threshold = np.percentile(rms_array, 95)
        print(f"  ⚠️  {percentage:.1f}% of frames exceed threshold!")
        print(f"  ⚠️  Your environment is NOISY")
        print(f"  ✅ Suggested threshold: {suggested_threshold:.6f} (95th percentile)")
    elif percentage > 10:
        suggested_threshold = np.percentile(rms_array, 90)
        print(f"  ⚠️  {percentage:.1f}% of frames exceed threshold")
        print(f"  ⚠️  Some background noise detected")
        print(f"  ✅ Suggested threshold: {suggested_threshold:.6f} (90th percentile)")
    else:
        print(f"  ✅ Threshold seems appropriate ({percentage:.1f}% above)")
        print(f"  ✅ Environment is relatively quiet")
    
    return rms_array

def test_vad_sensitivity(audio_data):
    """Test VAD with different aggressiveness levels"""
    print(f"\n{'='*60}")
    print(f"🎯 VAD SENSITIVITY TEST")
    print(f"{'='*60}")
    
    frame_size = CHUNK_SIZE
    
    for aggressiveness in [1, 2, 3]:
        vad = webrtcvad.Vad(aggressiveness)
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i + frame_size]
            
            # Normalize and convert to int16
            max_val = np.max(np.abs(frame))
            if max_val > 0:
                frame_normalized = frame / max_val
            else:
                frame_normalized = frame
            frame_int16 = (frame_normalized * 32767).astype(np.int16)
            
            is_speech = vad.is_speech(frame_int16.tobytes(), AUDIO_SAMPLE_RATE)
            if is_speech:
                speech_frames += 1
            total_frames += 1
        
        percentage = (speech_frames / total_frames) * 100
        print(f"Aggressiveness {aggressiveness}: {speech_frames}/{total_frames} frames detected as speech ({percentage:.1f}%)")
    
    print(f"\nCurrent setting: Aggressiveness {VAD_AGGRESSIVENESS}")
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  - Aggressiveness 1: Least aggressive (detects more speech)")
    print(f"  - Aggressiveness 3: Most aggressive (detects less speech)")
    print(f"  - If seeing high percentages, VAD is triggering on noise")

def test_whisper_on_noise(audio_data, stt_model):
    """Test what Whisper transcribes from the noise"""
    print(f"\n{'='*60}")
    print(f"🎤 WHISPER STT TEST ON NOISE")
    print(f"{'='*60}")
    print("Testing what Whisper transcribes from your 'silence'...")
    
    # Test on full recording
    segments, duration, detected_lang = stt_model.transcribe_audio(
        audio_data,
        AUDIO_SAMPLE_RATE,
        language="en",
        vad_filter=False
    )
    
    transcribed_text = " ".join([s.text for s in segments]) if segments else ""
    
    print(f"\nWhisper transcription of silence:")
    print(f"  '{transcribed_text}'")
    
    if transcribed_text.strip():
        print(f"\n⚠️  WARNING: Whisper is hallucinating on background noise!")
        print(f"  Common hallucinations: 'You', 'Thank you', punctuation")
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Filter out single-word transcriptions")
        print(f"  2. Increase pre-VAD RMS threshold")
        print(f"  3. Add minimum audio duration check (e.g., 0.5s)")
    else:
        print(f"\n✅ Whisper correctly identified silence (no transcription)")

def main():
    print(f"\n{'#'*60}")
    print(f"# NOISE DIAGNOSTIC TEST")
    print(f"# Live Speech Translation - Debugging 'You' Transcriptions")
    print(f"{'#'*60}")
    
    # Get audio file from command line or use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "test/Can you hear me_.wav"
        print(f"\nNo audio file specified, using default: {audio_file}")
        print(f"Usage: python test/diagnose_noise.py [audio_file.wav]\n")
    
    # Step 1: Load audio file
    audio_data = load_audio_file(audio_file)
    
    # Step 2: Analyze RMS levels
    rms_values = analyze_rms_levels(audio_data)
    
    # Step 3: Test VAD sensitivity
    test_vad_sensitivity(audio_data)
    
    # Step 4: Test Whisper on noise
    print(f"\nInitializing Whisper STT model...")
    stt_model = FasterWhisperSTT(model_size="base", device="auto", compute_type="int8")
    test_whisper_on_noise(audio_data, stt_model)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📋 DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    print(f"Current Configuration:")
    print(f"  - VAD Aggressiveness: {VAD_AGGRESSIVENESS}")
    print(f"  - Pre-VAD RMS Threshold: {SILENCE_RMS_THRESHOLD}")
    print(f"  - VAD Frame Duration: {VAD_FRAME_DURATION}ms")
    print(f"\nAnalyzed file: {audio_file}")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
