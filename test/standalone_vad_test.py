import sys
import os
import numpy as np
import soundfile as sf
import webrtcvad
import time
from collections import deque

# Add the parent directory (BP) to sys.path to import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.audio_utils import load_audio

# Configuration from backend/main.py for consistency
AUDIO_SAMPLE_RATE = 16000 # Standard sample rate for VAD
VAD_FRAME_DURATION = 30 # ms
VAD_AGGRESSIVENESS = 3 # Most aggressive (from backend.main)
MIN_SPEECH_DURATION = 0.2 # seconds (from backend.main)
SILENCE_TIMEOUT = 0.5 # seconds (from backend.main)
STREAMING_CHUNK_LENGTH = 0.5 # seconds (from backend.main)
SILENCE_RMS_THRESHOLD = 0.015 # From backend.main
PRE_VAD_BUFFER_DURATION = 0.5 # seconds (from backend.main)

def run_standalone_vad_test(audio_file_path: str):
    print(f"Starting standalone VAD test for {audio_file_path}...")

    # Load audio data and resample to 16kHz
    audio_data, sample_rate = load_audio(audio_file_path, target_sr=AUDIO_SAMPLE_RATE)
    
    vad_instance = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    audio_queue = deque()
    speech_frames = []
    in_speech_segment = False
    last_speech_time = time.perf_counter()

    # New variables for pre-VAD silence detection
    pre_vad_buffer = deque()
    pre_vad_threshold_met = False
    
    frame_size_samples = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
    pre_vad_buffer_samples = int(AUDIO_SAMPLE_RATE * PRE_VAD_BUFFER_DURATION)
    
    print(f"VAD Configuration: Aggressiveness={VAD_AGGRESSIVENESS}, Frame Duration={VAD_FRAME_DURATION}ms, Min Speech={MIN_SPEECH_DURATION}s, Silence Timeout={SILENCE_TIMEOUT}s")
    print(f"Pre-VAD Configuration: RMS Threshold={SILENCE_RMS_THRESHOLD}, Buffer Duration={PRE_VAD_BUFFER_DURATION}s")
    print(f"Processing audio in {frame_size_samples} samples ({VAD_FRAME_DURATION}ms) frames.")

    # Simulate streaming audio frame by frame
    for i in range(0, len(audio_data), frame_size_samples):
        frame_float32 = audio_data[i:i + frame_size_samples]

        if len(frame_float32) < frame_size_samples:
            # Pad with zeros if the last frame is too short
            frame_float32 = np.pad(frame_float32, (0, frame_size_samples - len(frame_float32)), 'constant')

        current_time_in_audio = (i / AUDIO_SAMPLE_RATE)

        # Pre-VAD silence detection logic
        if not pre_vad_threshold_met:
            pre_vad_buffer.extend(frame_float32)
            if len(pre_vad_buffer) >= pre_vad_buffer_samples:
                current_pre_vad_buffer_np = np.array(list(pre_vad_buffer), dtype=np.float32)
                buffer_rms = np.sqrt(np.mean(current_pre_vad_buffer_np**2))
                print(f"Time: {current_time_in_audio:.3f}s - Pre-VAD buffer RMS: {buffer_rms:.4f} (Threshold: {SILENCE_RMS_THRESHOLD})")
                if buffer_rms > SILENCE_RMS_THRESHOLD:
                    print(f"  -> Pre-VAD silence threshold met. Starting VAD processing at {current_time_in_audio:.3f}s")
                    pre_vad_threshold_met = True
                    # Add the buffered audio to the main audio_queue for VAD processing
                    audio_queue.extend(pre_vad_buffer)
                pre_vad_buffer.clear() # Clear buffer after check
            continue # Skip VAD processing until threshold is met

        # Normalize frame_float32 to [-1.0, 1.0] range and convert to int16 for VAD
        max_abs_frame_val = np.max(np.abs(frame_float32))
        if max_abs_frame_val > 0:
            frame_float32_normalized = frame_float32 / max_abs_frame_val
        else:
            frame_float32_normalized = frame_float32
        frame_int16 = (frame_float32_normalized * 32767).astype(np.int16)

        is_speech = vad_instance.is_speech(frame_int16.tobytes(), AUDIO_SAMPLE_RATE)
        
        print(f"Time: {current_time_in_audio:.3f}s - Is Speech: {is_speech}")

        if is_speech:
            speech_frames.append(frame_float32)
            if not in_speech_segment:
                print(f"  -> Speech segment STARTED at {current_time_in_audio:.3f}s")
                in_speech_segment = True
            last_speech_time = time.perf_counter()
        elif in_speech_segment:
            # Check for silence timeout
            if (time.perf_counter() - last_speech_time) > SILENCE_TIMEOUT:
                if speech_frames:
                    # Process the accumulated speech segment
                    full_segment_np = np.concatenate(speech_frames)
                    segment_duration = len(full_segment_np) / AUDIO_SAMPLE_RATE
                    print(f"  -> Speech segment ENDED at {current_time_in_audio:.3f}s. Duration: {segment_duration:.3f}s")
                    speech_frames.clear()
                in_speech_segment = False
        else:
            speech_frames.clear()

    # Handle any remaining speech at the end of the audio
    if in_speech_segment and speech_frames:
        full_segment_np = np.concatenate(speech_frames)
        segment_duration = len(full_segment_np) / AUDIO_SAMPLE_RATE
        print(f"  -> Final speech segment ENDED at end of audio. Duration: {segment_duration:.3f}s")

if __name__ == "__main__":
    audio_file = "test/Can you hear me_.wav"
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        sys.exit(1)
    run_standalone_vad_test(audio_file)
