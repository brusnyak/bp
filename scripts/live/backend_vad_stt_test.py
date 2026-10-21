import sys
import os
import numpy as np
import soundfile as sf
import webrtcvad
import time
from collections import deque
from typing import List, Dict, Tuple, Any

# Add the parent directory (BP) to sys.path to import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.audio_utils import load_audio
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from test.stt_comparison import calculate_wer, load_transcription # Import from stt_comparison.py

# Configuration from backend/main.py for consistency
AUDIO_SAMPLE_RATE = 16000 # Standard sample rate for VAD and STT
VAD_FRAME_DURATION = 30 # ms
VAD_AGGRESSIVENESS = 3 # Most aggressive
MIN_SPEECH_DURATION = 0.2 # seconds
SILENCE_TIMEOUT = 0.5 # seconds
STREAMING_CHUNK_LENGTH = 0.5 # seconds (not directly used in this test, but for context)

def run_backend_vad_stt_test(audio_file_path: str, transcript_file_path: str, stt_model_size: str = "base"):
    print(f"Starting backend VAD + STT test for {audio_file_path} with {stt_model_size} model...")

    # Load audio data and resample to 16kHz
    audio_data, sample_rate = load_audio(audio_file_path, target_sr=AUDIO_SAMPLE_RATE)
    
    # Load ground truth transcription
    ground_truth = load_transcription(transcript_file_path)
    print(f"Ground Truth: '{ground_truth}'")

    # Initialize VAD and STT models
    vad_instance = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    faster_whisper_stt = FasterWhisperSTT(model_size=stt_model_size, compute_type="int8")

    audio_buffer_for_vad = deque()
    speech_frames_for_stt = []
    full_transcription_segments = []
    in_speech_segment = False
    segment_start_time = None # Time when the current speech segment started accumulating

    frame_size_samples = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
    
    print(f"VAD Configuration: Aggressiveness={VAD_AGGRESSIVENESS}, Frame Duration={VAD_FRAME_DURATION}ms, Min Speech={MIN_SPEECH_DURATION}s, Silence Timeout={SILENCE_TIMEOUT}s")
    print(f"Processing audio in {frame_size_samples} samples ({VAD_FRAME_DURATION}ms) frames.")

    # Simulate streaming audio frame by frame
    for i in range(0, len(audio_data), frame_size_samples):
        frame_float32 = audio_data[i:i + frame_size_samples]

        if len(frame_float32) < frame_size_samples:
            # Pad with zeros if the last frame is too short
            frame_float32 = np.pad(frame_float32, (0, frame_size_samples - len(frame_float32)), 'constant')

        # Normalize frame_float32 to [-1.0, 1.0] range and convert to int16 for VAD
        max_abs_frame_val = np.max(np.abs(frame_float32))
        if max_abs_frame_val > 0:
            frame_float32_normalized = frame_float32 / max_abs_frame_val
        else:
            frame_float32_normalized = frame_float32
        frame_int16 = (frame_float32_normalized * 32767).astype(np.int16)

        is_speech = vad_instance.is_speech(frame_int16.tobytes(), AUDIO_SAMPLE_RATE)
        
        current_time_in_audio = (i / AUDIO_SAMPLE_RATE)
        # print(f"Time: {current_time_in_audio:.3f}s - Is Speech: {is_speech}") # Uncomment for detailed VAD logging

        if is_speech:
            speech_frames_for_stt.append(frame_float32)
            if not in_speech_segment:
                print(f"  -> Speech segment STARTED at {current_time_in_audio:.3f}s")
                in_speech_segment = True
                segment_start_time = time.perf_counter() # Mark start of segment processing
            # Update last_speech_time to prevent premature timeout
            last_speech_time = time.perf_counter() 
        elif in_speech_segment:
            # Check for silence timeout
            if (time.perf_counter() - last_speech_time) > SILENCE_TIMEOUT:
                if speech_frames_for_stt:
                    # Process the accumulated speech segment with STT
                    full_segment_np = np.concatenate(speech_frames_for_stt)
                    segment_duration = len(full_segment_np) / AUDIO_SAMPLE_RATE
                    print(f"  -> Speech segment ENDED at {current_time_in_audio:.3f}s. Duration: {segment_duration:.3f}s")
                    
                    # Perform STT on the segment
                    stt_segments, stt_latency, detected_lang = faster_whisper_stt.transcribe_audio(
                        full_segment_np, AUDIO_SAMPLE_RATE, language="en", vad_filter=False # VAD already handled
                    )
                    transcribed_text = " ".join([s.text for s in stt_segments]).strip()
                    full_transcription_segments.append(transcribed_text)
                    print(f"    Transcribed: '{transcribed_text}' (STT Latency: {stt_latency:.4f}s)")
                    
                    speech_frames_for_stt.clear()
                in_speech_segment = False
        else:
            # Not in speech segment and current frame is not speech
            speech_frames_for_stt.clear() # Clear any residual frames if VAD was toggling

    # Handle any remaining speech at the end of the audio
    if in_speech_segment and speech_frames_for_stt:
        full_segment_np = np.concatenate(speech_frames_for_stt)
        segment_duration = len(full_segment_np) / AUDIO_SAMPLE_RATE
        print(f"  -> Final speech segment ENDED at end of audio. Duration: {segment_duration:.3f}s")
        
        # Perform STT on the final segment
        stt_segments, stt_latency, detected_lang = faster_whisper_stt.transcribe_audio(
            full_segment_np, AUDIO_SAMPLE_RATE, language="en", vad_filter=False # VAD already handled
        )
        transcribed_text = " ".join([s.text for s in stt_segments]).strip()
        full_transcription_segments.append(transcribed_text)
        print(f"    Transcribed: '{transcribed_text}' (STT Latency: {stt_latency:.4f}s)")

    final_transcription = " ".join(full_transcription_segments).strip()
    final_wer = calculate_wer(ground_truth, final_transcription)

    print("\n--- Test Results ---")
    print(f"Ground Truth: '{ground_truth}'")
    print(f"Final Transcription: '{final_transcription}'")
    print(f"WER: {final_wer:.4f}")

    if final_wer < 0.2: # Example threshold
        print("VAD + STT Test: PASSED (WER below threshold)")
    else:
        print("VAD + STT Test: FAILED (WER above threshold)")

if __name__ == "__main__":
    audio_file = "test/Can you hear me_.wav"
    transcript_file = "test/Can you hear me_transcript.txt"
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        sys.exit(1)
    if not os.path.exists(transcript_file):
        print(f"Error: Transcript file not found at {transcript_file}")
        sys.exit(1)
    run_backend_vad_stt_test(audio_file, transcript_file, stt_model_size="base")
