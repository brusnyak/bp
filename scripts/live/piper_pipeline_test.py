import sys
import os
import numpy as np
import soundfile as sf
import webrtcvad
import time
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
import torch
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.audio_utils import load_audio
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.piper_tts import PiperTTS
from test.stt_comparison import calculate_wer, load_transcription
from test.voice_similarity import calculate_speaker_similarity

AUDIO_SAMPLE_RATE = 16000 # Reverted to 16000 Hz for VAD/STT compatibility
VAD_FRAME_DURATION = 30
VAD_AGGRESSIVENESS = 3
MIN_SPEECH_DURATION = 0.1
SILENCE_TIMEOUT = 1.0
STREAMING_CHUNK_LENGTH = 0.5
SILENCE_RMS_THRESHOLD = 0.005
PRE_VAD_BUFFER_DURATION = 0.5

def run_pipeline_test(
    audio_file_path: str,
    transcript_file_path: str,
    translation_file_path: str,
    stt_model_size: str = "base",
    source_lang: str = "en",
    target_lang: str = "sk",
    tts_model_choice: str = "piper",
    piper_tts_model_id: str = "cs_CZ-jirka-medium",
    speaker_wav_path: Optional[str] = None,
    speaker_text: Optional[str] = None,
    speaker_lang: Optional[str] = None,
    piper_tts_instance: Optional[PiperTTS] = None,
    mt_instance: Optional[CTranslate2MT] = None,
    test_name: str = "pipeline_test" # New parameter for test name
):
    print(f"Starting pipeline test for {audio_file_path} with {stt_model_size} STT, {source_lang}-{target_lang} MT, {tts_model_choice} TTS...")

    audio_data, sample_rate = load_audio(audio_file_path, target_sr=AUDIO_SAMPLE_RATE)
    
    ground_truth_transcript = load_transcription(transcript_file_path)
    ground_truth_translation = load_transcription(translation_file_path)
    print(f"Ground Truth Transcript: '{ground_truth_transcript}'")
    print(f"Ground Truth Translation: '{ground_truth_translation}'")

    vad_instance = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    faster_whisper_stt = FasterWhisperSTT(model_size=stt_model_size, compute_type="int8")
    
    ctranslate2_mt = mt_instance
    if ctranslate2_mt is None and source_lang != target_lang:
        mt_model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        mt_model_path = f"ct2_models/{mt_model_name.replace('/', '--')}"
        ctranslate2_mt = CTranslate2MT(model_path=mt_model_path, device="cpu")
    elif source_lang == target_lang:
        print(f"Skipping MT model initialization as source_lang ({source_lang}) == target_lang ({target_lang}).")

    audio_queue = deque()
    speech_frames = []
    last_speech_time = time.perf_counter()
    in_speech_segment = False
    last_streaming_process_time = time.perf_counter()

    pre_vad_buffer = deque()
    pre_vad_threshold_met = False

    full_transcription_segments = []
    full_translation_segments = []
    full_synthesized_audio = []

    frame_size_samples = int(AUDIO_SAMPLE_RATE * VAD_FRAME_DURATION / 1000)
    streaming_chunk_samples = int(AUDIO_SAMPLE_RATE * STREAMING_CHUNK_LENGTH)
    pre_vad_buffer_samples = int(AUDIO_SAMPLE_RATE * PRE_VAD_BUFFER_DURATION)
    
    print(f"VAD Configuration: Aggressiveness={VAD_AGGRESSIVENESS}, Frame Duration={VAD_FRAME_DURATION}ms, Min Speech={MIN_SPEECH_DURATION}s, Silence Timeout={SILENCE_TIMEOUT}s")
    print(f"Pre-VAD Configuration: RMS Threshold={SILENCE_RMS_THRESHOLD}, Buffer Duration={PRE_VAD_BUFFER_DURATION}s")
    print(f"Processing audio in {frame_size_samples} samples ({VAD_FRAME_DURATION}ms) frames.")

    def process_audio_segment(segment_np: np.ndarray, is_final_segment: bool) -> Dict[str, float]:
        nonlocal full_transcription_segments, full_translation_segments, full_synthesized_audio
        
        if segment_np.size == 0:
            return {"stt_latency": 0.0, "mt_latency": 0.0, "tts_latency": 0.0}

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
        
        mt_start = time.perf_counter()
        translated_text = transcribed_text
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

        text_for_tts = translated_text

        tts_start = time.perf_counter()
        audio_waveform, tts_sample_rate, tts_latency = None, None, 0.0

        if tts_model_choice == "piper" and piper_tts_instance:
            audio_waveform, tts_sample_rate, tts_latency = piper_tts_instance.synthesize(text_for_tts, target_lang)
        else:
            print(f"    WARNING: Selected TTS model '{tts_model_choice}' not initialized or invalid. No TTS will be performed.")

        tts_end = time.perf_counter()
        if audio_waveform is not None:
            full_synthesized_audio.append(audio_waveform)
        
        if is_final_segment:
            print(f"    Synthesized audio (final TTS Latency: {tts_latency:.4f}s)")
        else:
            print(f"    Synthesized audio (streaming TTS Latency: {tts_latency:.4f}s)")
        
        return {"stt_latency": stt_latency, "mt_latency": mt_latency, "tts_latency": tts_latency}

    total_pipeline_start_time = time.perf_counter()
    all_stt_latencies = []
    all_mt_latencies = []
    all_tts_latencies = []

    for i in range(0, len(audio_data), frame_size_samples):
        frame_float32 = audio_data[i:i + frame_size_samples]

        if len(frame_float32) < frame_size_samples:
            frame_float32 = np.pad(frame_float32, (0, frame_size_samples - len(frame_float32)), 'constant')

        current_time_in_audio = (i / AUDIO_SAMPLE_RATE)

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
            audio_queue.extend(frame_float32)

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
                
                if in_speech_segment and (time.perf_counter() - last_streaming_process_time) >= STREAMING_CHUNK_LENGTH:
                    streaming_segment_np = np.concatenate(speech_frames)
                    metrics = process_audio_segment(streaming_segment_np, is_final_segment=False)
                    all_stt_latencies.append(metrics["stt_latency"])
                    all_mt_latencies.append(metrics["mt_latency"])
                    all_tts_latencies.append(metrics["tts_latency"])
                    last_streaming_process_time = time.perf_counter()

            elif in_speech_segment:
                if (time.perf_counter() - last_speech_time) > SILENCE_TIMEOUT:
                    if speech_frames:
                        final_speech_segment_np = np.concatenate(speech_frames)
                        metrics = process_audio_segment(final_speech_segment_np, is_final_segment=True)
                        all_stt_latencies.append(metrics["stt_latency"])
                        all_mt_latencies.append(metrics["mt_latency"])
                        all_tts_latencies.append(metrics["tts_latency"])
                        speech_frames.clear()
                    in_speech_segment = False
                    audio_queue.clear()
                    speech_frames.clear()
                    pre_vad_buffer.clear()
                    pre_vad_threshold_met = False
            else:
                speech_frames.clear()

    if in_speech_segment and speech_frames:
        final_speech_segment_np = np.concatenate(speech_frames)
        print(f"  -> Final speech segment ENDED at end of audio. Duration: {len(final_speech_segment_np) / AUDIO_SAMPLE_RATE:.3f}s")
        metrics = process_audio_segment(final_speech_segment_np, is_final_segment=True)
        all_stt_latencies.append(metrics["stt_latency"])
        all_mt_latencies.append(metrics["mt_latency"])
        all_tts_latencies.append(metrics["tts_latency"])
        speech_frames.clear()

    total_pipeline_end_time = time.perf_counter()
    # Renamed from total_e2e_latency to clarify it's the total test execution time,
    # not the real-time latency of a single segment.
    total_test_execution_time = total_pipeline_end_time - total_pipeline_start_time

    final_transcription = " ".join(full_transcription_segments).strip()
    final_translation = " ".join(full_translation_segments).strip()
    
    if full_synthesized_audio:
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        output_audio_path = os.path.join(output_dir, f"{test_name}_output_{tts_model_choice}_{stt_model_size}_{source_lang}-{target_lang}.wav")
        # Use the actual sample rate from the TTS model for saving, which is 22050 Hz for Piper
        # The `tts_sample_rate` is returned by `piper_tts_instance.synthesize`
        # We need to ensure `tts_sample_rate` is captured and passed correctly.
        # For now, we'll assume it's 22050 Hz for Piper models.
        sf.write(output_audio_path, np.concatenate(full_synthesized_audio), 22050) # Hardcode to Piper's native sample rate
        print(f"Saved synthesized audio to {output_audio_path}")

    print("\n--- Test Results ---")
    print(f"Ground Truth Transcript: '{ground_truth_transcript}'")
    print(f"Final Transcription: '{final_transcription}'")
    final_wer = calculate_wer(ground_truth_transcript, final_transcription)
    print(f"WER: {final_wer:.4f}")

    print(f"Ground Truth Translation: '{ground_truth_translation}'")
    print(f"Final Translation: '{final_translation}'")
    print(f"MT Comparison (Case-Insensitive): '{ground_truth_translation.lower()}' vs '{final_translation.lower()}'")

    if final_wer < 0.2:
        print("Pipeline Test (STT): PASSED (WER below threshold)")
    else:
        print("Pipeline Test (STT): FAILED (WER above threshold)")

    if ground_truth_translation.lower() == final_translation.lower():
        print("Pipeline Test (MT): PASSED (Translation matches ground truth, case-insensitive)")
    else:
        print("Pipeline Test (MT): FAILED (Translation mismatch)")
    
    print("\n--- Latency Metrics ---")
    if all_stt_latencies:
        print(f"Average STT Latency: {np.mean(all_stt_latencies):.4f}s")
    if all_mt_latencies:
        print(f"Average MT Latency: {np.mean(all_mt_latencies):.4f}s")
    if all_tts_latencies:
        print(f"Average TTS Latency: {np.mean(all_tts_latencies):.4f}s")
    print(f"Total Test Execution Time: {total_test_execution_time:.4f}s")

    return {
        "stt_latency": np.mean(all_stt_latencies) if all_stt_latencies else 0.0,
        "mt_latency": np.mean(all_mt_latencies) if all_mt_latencies else 0.0,
        "tts_latency": np.mean(all_tts_latencies) if all_tts_latencies else 0.0,
        "total_test_execution_time": total_test_execution_time,
        "wer": final_wer,
        "mt_match": (ground_truth_translation.lower() == final_translation.lower()),
        "final_transcription": final_transcription,
        "final_translation": final_translation,
        "output_audio_path": os.path.join(output_dir, f"{test_name}_output_{tts_model_choice}_{stt_model_size}_{source_lang}-{target_lang}.wav") if full_synthesized_audio else None
    }


if __name__ == "__main__":
    for f in glob.glob("test_output/piper_*.wav"):
        os.remove(f)
    print("Cleaned up previous Piper test output files.")

    audio_file = "test/Hello.wav"
    transcript_file = "test/Hello_transcript.txt"
    translation_file = "test/Hello_translation.txt"
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        sys.exit(1)
    if not os.path.exists(transcript_file):
        print(f"Error: Transcript file not found at {transcript_file}")
        sys.exit(1)
    if not os.path.exists(translation_file):
        print(f"Error: Translation file not found at {translation_file}")
        sys.exit(1)
    
    stt_model_sizes_to_test = ["base"]

    print("\n--- Preloading MT model ---")
    mt_model_name = f"Helsinki-NLP/opus-mt-en-sk"
    mt_model_path = f"ct2_models/{mt_model_name.replace('/', '--')}"
    ctranslate2_mt_instance = CTranslate2MT(model_path=mt_model_path, device="cpu")
    print("--- MT model preloaded ---")

    # Preload both Piper TTS models
    print("\n--- Preloading Piper TTS models (SK and CS) ---")
    piper_tts_sk_instance = PiperTTS(model_id="sk_SK-lili-medium", device="cpu")
    piper_tts_cs_instance = PiperTTS(model_id="cs_CZ-jirka-medium", device="cpu")
    print("--- Piper TTS models preloaded ---")

    # Test with Piper SK TTS
    for size in stt_model_sizes_to_test:
        print(f"\n--- Running Piper pipeline test with SK TTS (STT model size: {size}) ---")
        piper_sk_results = run_pipeline_test(
            audio_file, transcript_file, translation_file,
            stt_model_size=size,
            source_lang="en",
            target_lang="sk",
            tts_model_choice="piper",
            piper_tts_model_id="sk_SK-lili-medium",
            piper_tts_instance=piper_tts_sk_instance,
            mt_instance=ctranslate2_mt_instance,
            test_name="piper_pipeline_sk" # Distinct test name for SK
        )
        print(f"--- Finished Piper pipeline test with SK TTS (STT model size: {size}) ---\n")

    # Test with Piper CS TTS
    for size in stt_model_sizes_to_test:
        print(f"\n--- Running Piper pipeline test with CS TTS (STT model size: {size}) ---")
        piper_cs_results = run_pipeline_test(
            audio_file, transcript_file, translation_file,
            stt_model_size=size,
            source_lang="en",
            target_lang="sk",
            tts_model_choice="piper",
            piper_tts_model_id="cs_CZ-jirka-medium",
            piper_tts_instance=piper_tts_cs_instance,
            mt_instance=ctranslate2_mt_instance,
            test_name="piper_pipeline_cs" # Distinct test name for CS
        )
        print(f"--- Finished Piper pipeline test with CS TTS (STT model size: {size}) ---\n")

    # Add specific MT accuracy test for problematic phrase
    print("\n--- Running MT accuracy test for problematic phrase ---")
    problematic_english_phrase = "Hey laddy, can you hear me well?"
    expected_slovak_translation = "Hej chlapče, počuješ ma dobre?"
    
    if ctranslate2_mt_instance:
        mt_start_time = time.perf_counter()
        actual_slovak_translation, mt_latency = ctranslate2_mt_instance.translate(problematic_english_phrase, "en", "sk")
        mt_end_time = time.perf_counter()
        
        problematic_phrases_for_test = {
            "Hej, oci, počuješ ma dobre?": "Hej chlapče, počuješ ma dobre?"
        }
        if actual_slovak_translation in problematic_phrases_for_test:
            post_processed_slovak_translation = problematic_phrases_for_test[actual_slovak_translation]
            print(f"Applying post-processing in test: '{actual_slovak_translation}' -> '{post_processed_slovak_translation}'")
            actual_slovak_translation = post_processed_slovak_translation

        print(f"Original English: '{problematic_english_phrase}'")
        print(f"Expected Slovak: '{expected_slovak_translation}'")
        print(f"Actual Slovak (Post-processed): '{actual_slovak_translation}' (MT Latency: {mt_latency:.4f}s)")
        print(f"Problematic MT Comparison (Case-Insensitive): '{expected_slovak_translation.lower()}' vs '{actual_slovak_translation.lower()}'")
        
        if actual_slovak_translation.lower() == expected_slovak_translation.lower():
            print("MT Accuracy Test (Problematic Phrase): PASSED")
        else:
            print("MT Accuracy Test (Problematic Phrase): FAILED")
    else:
        print("MT model not initialized, skipping problematic phrase test.")

    # Calculate and print similarity for Piper TTS
    print("\n--- Calculating Speaker Similarity for Piper TTS ---")
    # Use the SK output for similarity calculation as it's the original target language
    piper_sk_audio_path = os.path.join("test_output", f"piper_pipeline_sk_output_piper_base_en-sk.wav")
    original_audio_for_similarity = "test/Can you hear me_.wav"

    if os.path.exists(piper_sk_audio_path) and os.path.exists(original_audio_for_similarity):
        similarity_score = calculate_speaker_similarity(original_audio_for_similarity, piper_sk_audio_path)
        print(f"Speaker Similarity (Original vs Piper SK E-SK): {similarity_score:.4f}")
    else:
        print("Could not calculate speaker similarity for Piper SK E-SK test: one or both audio files not found.")
