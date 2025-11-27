import sys
import os
import numpy as np
import soundfile as sf
import webrtcvad
import time
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
import torch # Import torch to check for MPS availability
import glob # Import glob for file cleanup

# Add the parent directory (BP) to sys.path to import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.audio_utils import load_audio
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.piper_tts import PiperTTS
from backend.tts.f5_tts import F5_TTS # Import F5_TTS
from test.stt_comparison import calculate_wer, load_transcription # Import from stt_comparison.py
from test.voice_similarity import calculate_speaker_similarity # Import new similarity function

# Configuration from backend/main.py for consistency
AUDIO_SAMPLE_RATE = 16000 # Standard sample rate for VAD and STT
VAD_FRAME_DURATION = 30 # ms
VAD_AGGRESSIVENESS = 3 # Most aggressive
MIN_SPEECH_DURATION = 0.1 # seconds (Reduced to capture shorter initial speech segments)
SILENCE_TIMEOUT = 1.0 # seconds (Increased to allow for longer natural pauses)
STREAMING_CHUNK_LENGTH = 0.5 # seconds
SILENCE_RMS_THRESHOLD = 0.005 # Reduced threshold to be more sensitive to speech beginnings
PRE_VAD_BUFFER_DURATION = 0.5 # seconds, how much audio to buffer before checking RMS

def run_full_pipeline_test(
    audio_file_path: str,
    transcript_file_path: str,
    translation_file_path: str,
    stt_model_size: str = "base",
    source_lang: str = "en",
    target_lang: str = "sk",
    tts_model_choice: str = "piper", # New parameter for TTS model choice
    piper_tts_model_id: str = "cs_CZ-jirka-medium", # Renamed for clarity
    speaker_wav_path: Optional[str] = None, # New parameter for F5-TTS
    speaker_text: Optional[str] = None, # New parameter for F5-TTS
    speaker_lang: Optional[str] = None, # New parameter for F5-TTS
    piper_tts_instance: Optional[PiperTTS] = None, # Pass preloaded PiperTTS instance
    f5_tts_instance: Optional[F5_TTS] = None, # Pass preloaded F5_TTS instance
    f5_nfe_step: int = 32, # New parameter for F5-TTS
    f5_sway_sampling_coef: float = -1,
    f5_cfg_strength: float = 2, # New parameter for F5-TTS
    f5_speed: float = 1.0, # New parameter for F5-TTS
    f5_fix_duration: Optional[float] = None,
    f5_cross_fade_duration: float = 0.15,
    f5_gen_text_override: Optional[str] = None,
    mt_instance: Optional[CTranslate2MT] = None, # New parameter for MT instance
):
    print(f"Starting full pipeline test for {audio_file_path} with {stt_model_size} STT, {source_lang}-{target_lang} MT, {tts_model_choice} TTS (F5 NFE: {f5_nfe_step}, Sway: {f5_sway_sampling_coef}, CfgStrength: {f5_cfg_strength}, Speed: {f5_speed}, FixDur: {f5_fix_duration}, CrossFade: {f5_cross_fade_duration})...")

    # Load audio data and resample to 16kHz
    audio_data, sample_rate = load_audio(audio_file_path, target_sr=AUDIO_SAMPLE_RATE)
    
    # Load ground truth transcription and translation
    ground_truth_transcript = load_transcription(transcript_file_path)
    ground_truth_translation = load_transcription(translation_file_path)
    print(f"Ground Truth Transcript: '{ground_truth_transcript}'")
    print(f"Ground Truth Translation: '{ground_truth_translation}'")

    # Initialize VAD, STT models
    vad_instance = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    faster_whisper_stt = FasterWhisperSTT(model_size=stt_model_size, compute_type="int8") # Standardized compute_type to int8
    
    # Use provided MT instance or initialize if needed
    ctranslate2_mt = mt_instance
    if ctranslate2_mt is None and source_lang != target_lang:
        mt_model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        mt_model_path = f"ct2_models/{mt_model_name.replace('/', '--')}"
        ctranslate2_mt = CTranslate2MT(model_path=mt_model_path, device="cpu")
    elif source_lang == target_lang:
        print(f"Skipping MT model initialization as source_lang ({source_lang}) == target_lang ({target_lang}).")

    audio_queue = deque()
    speech_frames = [] # This will accumulate all VAD-detected speech frames for the entire utterance
    last_speech_time = time.perf_counter()
    in_speech_segment = False
    last_streaming_process_time = time.perf_counter() # Track last time a streaming chunk was processed

    pre_vad_buffer = deque()
    pre_vad_threshold_met = False

    full_transcription_segments = [] # Only for final transcription
    full_translation_segments = [] # Only for final translation
    full_synthesized_audio = []

    frame_size_samples = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
    streaming_chunk_samples = int(AUDIO_SAMPLE_RATE * STREAMING_CHUNK_LENGTH)
    pre_vad_buffer_samples = int(AUDIO_SAMPLE_RATE * PRE_VAD_BUFFER_DURATION)
    
    print(f"VAD Configuration: Aggressiveness={VAD_AGGRESSIVENESS}, Frame Duration={VAD_FRAME_DURATION}ms, Min Speech={MIN_SPEECH_DURATION}s, Silence Timeout={SILENCE_TIMEOUT}s")
    print(f"Pre-VAD Configuration: RMS Threshold={SILENCE_RMS_THRESHOLD}, Buffer Duration={PRE_VAD_BUFFER_DURATION}s")
    print(f"Processing audio in {frame_size_samples} samples ({VAD_FRAME_DURATION}ms) frames.")

    # Helper function to process a segment (STT -> MT -> TTS)
    def process_audio_segment(segment_np: np.ndarray, is_final_segment: bool) -> Dict[str, float]:
        nonlocal full_transcription_segments, full_translation_segments, full_synthesized_audio
        
        if segment_np.size == 0:
            return {"stt_latency": 0.0, "mt_latency": 0.0, "tts_latency": 0.0}

        # STT
        stt_start = time.perf_counter()
        stt_segments, stt_latency, detected_lang = faster_whisper_stt.transcribe_audio(
            segment_np, AUDIO_SAMPLE_RATE, language=source_lang, vad_filter=False
        )
        stt_end = time.perf_counter()
        transcribed_text = " ".join([s.text for s in stt_segments]).strip()
        
        if is_final_segment:
            full_transcription_segments.append(transcribed_text)
            print(f"    Transcribed (final): '{transcribed_text}' (STT Latency: {stt_latency:.4f}s)")
        else:
            print(f"    Transcribed (streaming): '{transcribed_text}' (STT Latency: {stt_latency:.4f}s)")
        
        # MT
        mt_start = time.perf_counter()
        translated_text = transcribed_text # Default to transcribed text
        mt_latency = 0.0
        if source_lang != target_lang and ctranslate2_mt:
            translated_text, mt_latency = ctranslate2_mt.translate(transcribed_text, source_lang, target_lang)
        else:
            print(f"    Source and target languages are the same ({source_lang}). Skipping machine translation.")
        mt_end = time.perf_counter()
        
        if is_final_segment:
            full_translation_segments.append(translated_text)
            print(f"    Translated (final): '{translated_text}' (MT Latency: {mt_latency:.4f}s)")
        else:
            print(f"    Translated (streaming): '{translated_text}' (MT Latency: {mt_latency:.4f}s)")

        # Determine text for TTS
        text_for_tts = f5_gen_text_override if tts_model_choice == "F5" and f5_gen_text_override is not None else translated_text
        if tts_model_choice == "F5" and target_lang == "sk":
            print(f"    F5-TTS (Slovak) input text: '{text_for_tts}'")

        # TTS
        tts_start = time.perf_counter()
        audio_waveform, tts_sample_rate, tts_latency = None, None, 0.0

        if tts_model_choice == "piper" and piper_tts_instance:
            audio_waveform, tts_sample_rate, tts_latency = piper_tts_instance.synthesize(text_for_tts, target_lang)
        elif tts_model_choice == "F5" and f5_tts_instance:
            if not speaker_wav_path:
                print("    WARNING: F5-TTS selected but no speaker_wav_path provided. Skipping F5-TTS.")
            else:
                audio_waveform, tts_sample_rate, tts_latency = f5_tts_instance.synthesize(
                    text_for_tts, language=target_lang,
                    speaker_wav_path=speaker_wav_path,
                    speaker_text=speaker_text,
                    speaker_lang=speaker_lang,
                    nfe_step=f5_nfe_step,
                    sway_sampling_coef=f5_sway_sampling_coef,
                    cfg_strength=f5_cfg_strength,
                    speed=f5_speed,
                    fix_duration=f5_fix_duration,
                    cross_fade_duration=f5_cross_fade_duration,
                )
        else:
            print(f"    WARNING: Selected TTS model '{tts_model_choice}' not initialized or invalid. No TTS will be performed.")

        tts_end = time.perf_counter()
        if audio_waveform is not None:
            # Conditionally convert to numpy array based on TTS model choice
            if tts_model_choice == "piper":
                full_synthesized_audio.append(audio_waveform) # Already a numpy array
            elif tts_model_choice == "F5":
                full_synthesized_audio.append(audio_waveform.squeeze(0).cpu().numpy()) # Convert torch.Tensor to numpy array
        
        if is_final_segment:
            print(f"    Synthesized audio (final TTS Latency: {tts_latency:.4f}s)")
        else:
            print(f"    Synthesized audio (streaming TTS Latency: {tts_latency:.4f}s)")
        
        return {"stt_latency": stt_latency, "mt_latency": mt_latency, "tts_latency": tts_latency}

    # Simulate streaming audio frame by frame
    total_pipeline_start_time = time.perf_counter()
    all_stt_latencies = []
    all_mt_latencies = []
    all_tts_latencies = []

    for i in range(0, len(audio_data), frame_size_samples):
        frame_float32 = audio_data[i:i + frame_size_samples]

        if len(frame_float32) < frame_size_samples:
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
                    audio_queue.extend(pre_vad_buffer)
                pre_vad_buffer.clear()
            continue
        
        if pre_vad_threshold_met:
            audio_queue.extend(frame_float32) # Add current frame to audio_queue for VAD processing

        while len(audio_queue) >= frame_size_samples:
            vad_frame_float32 = np.array([audio_queue.popleft() for _ in range(frame_size_samples)], dtype=np.float32)
            
            max_abs_frame_val = np.max(np.abs(vad_frame_float32))
            if max_abs_frame_val > 0:
                vad_frame_float32_normalized = vad_frame_float32 / max_abs_frame_val
            else:
                vad_frame_float32_normalized = vad_frame_float32
            frame_int16 = (vad_frame_float32_normalized * 32767).astype(np.int16)

            is_speech = vad_instance.is_speech(frame_int16.tobytes(), AUDIO_SAMPLE_RATE)
            
            if is_speech:
                speech_frames.append(frame_float32)
                if not in_speech_segment:
                    print(f"  -> Speech segment STARTED at {current_time_in_audio:.3f}s")
                    in_speech_segment = True
                last_speech_time = time.perf_counter()
                
                # Check if enough speech has accumulated for a streaming chunk
                if in_speech_segment and (time.perf_counter() - last_streaming_process_time) >= STREAMING_CHUNK_LENGTH:
                    # Process a copy of the current speech_frames for streaming
                    # This ensures speech_frames is not cleared for the final segment
                    streaming_segment_np = np.concatenate(speech_frames)
                    metrics = process_audio_segment(streaming_segment_np, is_final_segment=False)
                    all_stt_latencies.append(metrics["stt_latency"])
                    all_mt_latencies.append(metrics["mt_latency"])
                    all_tts_latencies.append(metrics["tts_latency"])
                    last_streaming_process_time = time.perf_counter()

            elif in_speech_segment:
                # Silence detected after speech, or speech ended
                if (time.perf_counter() - last_speech_time) > SILENCE_TIMEOUT:
                    if speech_frames:
                        final_speech_segment_np = np.concatenate(speech_frames)
                        metrics = process_audio_segment(final_speech_segment_np, is_final_segment=True)
                        all_stt_latencies.append(metrics["stt_latency"])
                        all_mt_latencies.append(metrics["mt_latency"])
                        all_tts_latencies.append(metrics["tts_latency"])
                        speech_frames.clear()
                    in_speech_segment = False
                    # Ensure all buffers are cleared after a full speech segment ends
                    audio_queue.clear()
                    speech_frames.clear() # Redundant but safe
                    pre_vad_buffer.clear()
                    pre_vad_threshold_met = False
            else:
                # Not in speech segment and current frame is not speech, clear buffers
                speech_frames.clear() # Redundant but safe

    # Handle any remaining speech at the end of the audio
    if in_speech_segment and speech_frames:
        final_speech_segment_np = np.concatenate(speech_frames)
        print(f"  -> Final speech segment ENDED at end of audio. Duration: {len(final_speech_segment_np) / AUDIO_SAMPLE_RATE:.3f}s")
        metrics = process_audio_segment(final_speech_segment_np, is_final_segment=True)
        all_stt_latencies.append(metrics["stt_latency"])
        all_mt_latencies.append(metrics["mt_latency"])
        all_tts_latencies.append(metrics["tts_latency"])
        speech_frames.clear()

    total_pipeline_end_time = time.perf_counter()
    total_e2e_latency = total_pipeline_end_time - total_pipeline_start_time

    final_transcription = " ".join(full_transcription_segments).strip()
    final_translation = " ".join(full_translation_segments).strip()
    
    # Save synthesized audio for verification
    if full_synthesized_audio:
        output_dir = "test_output"
        if tts_model_choice == "F5":
            output_dir = "test_output/f5_tts_tests"
        
        os.makedirs(output_dir, exist_ok=True)
        output_audio_path = os.path.join(output_dir, f"full_pipeline_output_{tts_model_choice}_{stt_model_size}_{source_lang}-{target_lang}.wav")
        
        # Use the native sample rate for saving F5-TTS output (24000 Hz)
        # For Piper, it's 22050 Hz, but this test file is primarily for F5-TTS.
        # We'll hardcode 24000 Hz for F5-TTS outputs here.
        if tts_model_choice == "F5":
            sf.write(output_audio_path, np.concatenate(full_synthesized_audio), 24000) # F5-TTS native sample rate
        else: # Assume Piper or other TTS, use its native rate if known, or default to AUDIO_SAMPLE_RATE
            sf.write(output_audio_path, np.concatenate(full_synthesized_audio), AUDIO_SAMPLE_RATE) # Default to 16000 Hz for other cases
        print(f"Saved synthesized audio to {output_audio_path}")

    print("\n--- Test Results ---")
    print(f"Ground Truth Transcript: '{ground_truth_transcript}'")
    print(f"Final Transcription: '{final_transcription}'")
    final_wer = calculate_wer(ground_truth_transcript, final_transcription)
    print(f"WER: {final_wer:.4f}")

    print(f"Ground Truth Translation: '{ground_truth_translation}'")
    print(f"Final Translation: '{final_translation}'")
    print(f"MT Comparison (Case-Insensitive): '{ground_truth_translation.lower()}' vs '{final_translation.lower()}'") # Added for debugging

    if final_wer < 0.2: # Example threshold
        print("Full Pipeline Test (STT): PASSED (WER below threshold)")
    else:
        print("Full Pipeline Test (STT): FAILED (WER above threshold)")

    if ground_truth_translation.lower() == final_translation.lower():
        print("Full Pipeline Test (MT): PASSED (Translation matches ground truth, case-insensitive)")
    else:
        print("Full Pipeline Test (MT): FAILED (Translation mismatch)")
    
    print("\n--- Latency Metrics ---")
    if all_stt_latencies:
        print(f"Average STT Latency: {np.mean(all_stt_latencies):.4f}s")
    if all_mt_latencies:
        print(f"Average MT Latency: {np.mean(all_mt_latencies):.4f}s")
    if all_tts_latencies:
        print(f"Average TTS Latency: {np.mean(all_tts_latencies):.4f}s")
    print(f"Total End-to-End Pipeline Latency: {total_e2e_latency:.4f}s")
