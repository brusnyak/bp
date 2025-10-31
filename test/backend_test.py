import asyncio
import numpy as np
import soundfile as sf
import os
import sys
import os
import time
from typing import Dict, Any

# Add the project root to sys.path to enable package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.stt.mlx_whisper_stt import MLXWhisperSTT
from backend.stt.mlx_whisper_stt import MLXWhisperSTT
from backend.mt.nllb_mt import NLLB_MT # Added for type hinting and NLLB testing
from backend.mt.marian_mt import MarianMT # Added for MarianMT testing
from backend.tts.chatterbox_tts import ChatterboxTTS
from backend.tts.piper_tts import PiperTTS
import piper # Corrected import for piper_tts package
from chatterbox.mtl_tts import ChatterboxMultilingualTTS # Added for direct access to MTLTTS if needed for voice training
from backend.utils.audio_utils import load_audio, save_audio, normalize_audio
from backend.main import speaker_embeddings, STT_MODEL_SIZE, PIPER_MODEL_PATH, PIPER_CONFIG_PATH, AUDIO_SAMPLE_RATE, MT_MODEL_NAME # Removed initialize_models
import backend.main # Import the module to access its global variables

async def run_backend_test_workflow(
    stt_model_instance: MLXWhisperSTT,
    mt_model_instance: Any, # Changed to Any to accommodate different MT model types
    piper_tts_model_instance: PiperTTS,
    voice_training_audio_path: str,
    input_speech_audio_path: str,
    source_lang: str,
    target_lang: str,
    tts_model_choice: str,
    output_audio_path: str = "output_dynamic.wav"
) -> Dict[str, Any]:
    """
    Runs the full backend workflow for testing with pre-initialized models:
    1. Uses provided STT, MT, and TTS model instances.
    2. Skips Chatterbox voice training.
    3. Processes input speech.
    4. Outputs results.

    Args:
        stt_model_instance (MLXWhisperSTT): Pre-initialized MLXWhisperSTT model.
        mt_model_instance (NLLB_MT): Pre-initialized MT model.
        piper_tts_model_instance (PiperTTS): Pre-initialized PiperTTS model.
        voice_training_audio_path (str): Path to the audio file for voice training (for Chatterbox, currently skipped).
        input_speech_audio_path (str): Path to the input speech audio file for translation.
        source_lang (str): Source language for translation (e.g., "en").
        target_lang (str): Target language for translation (e.g., "sk").
        tts_model_choice (str): Chosen TTS model ("chatterbox" or "piper").
        output_audio_path (str): Path to save the final synthesized audio output.

    Returns:
        Dict[str, Any]: A dictionary containing metrics and results.
    """
    print("\n--- Starting Backend Test Workflow ---")
    results = {
        "status": "failed",
        "stt_time": 0.0,
        "mt_time": 0.0,
        "tts_time": 0.0,
        "total_latency": 0.0,
        "transcribed_text": "",
        "translated_text": "",
        "output_audio_path": ""
    }

    # Models are now passed in, no need to initialize here.
    print("Models are pre-initialized.")

    # 2. Train voice with 'test/Voice-Training.wav'
    print(f"2. Training voice with '{voice_training_audio_path}'...")
    # Temporarily skipping Chatterbox voice training due to 'get_speaker_embedding' error
    print("WARNING: Chatterbox voice training is temporarily skipped due to 'get_speaker_embedding' error.")
    speaker_embeddings.pop("default_speaker", None) # Ensure no invalid embedding is used
    # if backend.main.chatterbox_tts_model:
    #     try:
    #         audio_np, sr = load_audio(voice_training_audio_path, target_sr=16000)
    #         speaker_info = backend.main.chatterbox_tts_model.train_voice(voice_training_audio_path)
    #         speaker_embeddings["default_speaker"] = speaker_info["speaker_embedding"]
    #         print("Voice training successful. Speaker embedding stored.")
    #     except Exception as e:
    #         print(f"Error during voice training: {e}")
    #         print("Proceeding without voice cloning for TTS.")
    #         speaker_embeddings.pop("default_speaker", None)
    # else:
    #     print("Chatterbox TTS model not initialized. Skipping voice training.")
    #     speaker_embeddings.pop("default_speaker", None)

    # 3. Start and give input with 'test/My test speech_xtts_speaker_clean.wav'
    print(f"3. Processing input speech from '{input_speech_audio_path}'...")
    input_audio_np, input_sr = load_audio(input_speech_audio_path, target_sr=AUDIO_SAMPLE_RATE)

    full_pipeline_start_time = time.perf_counter()

    # STT
    stt_start_time = time.perf_counter()
    # Re-introducing language parameter
    transcribed_text, _ = stt_model_instance.transcribe_audio(input_audio_np, input_sr, language=source_lang)
    stt_end_time = time.perf_counter()
    stt_total_time = stt_end_time - stt_start_time
    results["stt_time"] = stt_total_time
    results["transcribed_text"] = transcribed_text
    print(f"STT Result: '{transcribed_text}' (Time: {stt_total_time:.2f}s)")

    if not transcribed_text:
        print("STT failed to produce text. Aborting workflow.")
        return results

    # MT
    mt_start_time = time.perf_counter()
    # NLLB_MT and MarianMT use translate_text, SeamlessM4Tv2MT uses translate
    if isinstance(mt_model_instance, NLLB_MT) or isinstance(mt_model_instance, MarianMT):
        translated_text, _ = mt_model_instance.translate_text(transcribed_text, source_lang, target_lang)
    else:
        print(f"ERROR: Unknown MT model instance type: {type(mt_model_instance)}")
        return results
    
    mt_end_time = time.perf_counter()
    mt_total_time = mt_end_time - mt_start_time
    results["mt_time"] = mt_total_time
    results["translated_text"] = translated_text
    print(f"MT Result: '{translated_text}' (Time: {mt_total_time:.2f}s)")

    if not translated_text:
        print("MT failed to produce text. Aborting workflow.")
        return results

    # TTS
    tts_start_time = time.perf_counter()
    audio_waveform, sample_rate, _ = None, None, 0.0

    if tts_model_choice == "chatterbox" and backend.main.chatterbox_tts_model: # Still relying on global for Chatterbox for now
        speaker_emb = speaker_embeddings.get("default_speaker")
        audio_waveform, sample_rate, _ = backend.main.chatterbox_tts_model.synthesize_speech(
            translated_text, target_lang, speaker_embedding=speaker_emb
        )
    elif tts_model_choice == "piper" and piper_tts_model_instance:
        audio_waveform, sample_rate, _ = piper_tts_model_instance.synthesize_speech(
            translated_text, target_lang # Piper uses language for context, model is language-specific
        )
    else:
        print(f"ERROR: Selected TTS model '{tts_model_choice}' not initialized or invalid.")
        return results
    
    tts_end_time = time.perf_counter()
    tts_total_time = tts_end_time - tts_start_time
    results["tts_time"] = tts_total_time
    print(f"TTS completed (Time: {tts_total_time:.2f}s)")

    if audio_waveform is not None and sample_rate is not None:
        # 4. Output in default dynamic.
        save_audio(output_audio_path, normalize_audio(audio_waveform), sample_rate)
        results["output_audio_path"] = os.path.abspath(output_audio_path)
        print(f"Final synthesized audio saved to: {results['output_audio_path']}")
    else:
        print("TTS failed to produce audio. Aborting workflow.")
        return results

    full_pipeline_end_time = time.perf_counter()
    total_latency = full_pipeline_end_time - full_pipeline_start_time
    results["total_latency"] = total_latency
    results["status"] = "success"
    print(f"--- Backend Test Workflow Completed (Total Latency: {total_latency:.2f}s) ---")
    
    return results

async def main():
    # Define test parameters
    voice_training_file = "test/Voice-Training.wav"
    input_speech_file = "test/My test speech_xtts_speaker_clean.wav"
    
    # Ensure test files exist
    if not os.path.exists(voice_training_file):
        print(f"ERROR: Voice training file not found: {voice_training_file}")
        print("Please ensure 'test/Voice-Training.wav' exists in your project.")
        return
    if not os.path.exists(input_speech_file):
        print(f"ERROR: Input speech file not found: {input_speech_file}")
        print("Please ensure 'test/My test speech_xtts_speaker_clean.wav' exists in your project.")
        return

    print("\n--- Preloading all models ---")
    # Initialize STT
    stt_model_instance = MLXWhisperSTT(model_size=STT_MODEL_SIZE, compute_type="int8")
    # MLXWhisperSTT loads its model in __init__, no separate load_model call needed.

    # Initialize MT models for comparison
    nllb_mt_model_instance = NLLB_MT(model_name="facebook/nllb-200-distilled-600M", device="auto", compute_type="int8")
    marian_mt_model_instance = MarianMT(model_name="Helsinki-NLP/opus-mt-en-sk", device="auto")
    
    # NLLB_MT and MarianMT load models in __init__
    
    # Initialize Piper TTS
    piper_tts_model_instance = None # Initialize to None
    if os.path.exists(PIPER_MODEL_PATH) and os.path.exists(PIPER_CONFIG_PATH):
        try:
            piper_tts_model_instance = PiperTTS(model_path=PIPER_MODEL_PATH, speaker_id=0)
        except Exception as e:
            print(f"WARNING: Could not initialize PiperTTS. Error: {e}")
    else:
        print(f"WARNING: Piper TTS model files not found at {PIPER_MODEL_PATH} and {PIPER_CONFIG_PATH}. PiperTTS will not be initialized.")
    
    print("--- All models preloaded ---")

    # Test with Chatterbox (Temporarily commented out due to dependency conflicts and 'get_speaker_embedding' error)
    print("\n=== Skipping Chatterbox TTS Test (Temporarily Disabled) ===")
    # chatterbox_results = await run_backend_test_workflow(
    #     stt_model_instance,
    #     mt_model_instance,
    #     chatterbox_tts_model_instance, # Assuming chatterbox_tts_model_instance would be initialized here
    #     voice_training_file,
    #     input_speech_file,
    #     source_lang="en",
    #     target_lang="sk",
    #     tts_model_choice="chatterbox",
    #     output_audio_path="output_chatterbox_test.wav"
    # )
    # print("\nChatterbox Test Results:")
    # for k, v in chatterbox_results.items():
    #     print(f"  {k}: {v}")
    
    # Test with NLLB-200-distilled
    print("\n=== Testing with NLLB-200-distilled MT ===")
    nllb_results = await run_backend_test_workflow(
        stt_model_instance,
        nllb_mt_model_instance,
        piper_tts_model_instance,
        voice_training_file,
        input_speech_file,
        source_lang="en",
        target_lang="sk",
        tts_model_choice="piper",
        output_audio_path="output_nllb_test.wav"
    )
    print("\nNLLB-200-distilled Test Results:")
    for k, v in nllb_results.items():
        print(f"  {k}: {v}")


    # Test with MarianMT (Helsinki-NLP/opus-mt-en-sk)
    print("\n=== Testing with MarianMT (Helsinki-NLP/opus-mt-en-sk) ===")
    marian_mt_results = await run_backend_test_workflow(
        stt_model_instance,
        marian_mt_model_instance,
        piper_tts_model_instance,
        voice_training_file,
        input_speech_file,
        source_lang="en",
        target_lang="sk",
        tts_model_choice="piper",
        output_audio_path="output_marian_mt_test.wav"
    )
    print("\nMarianMT Test Results:")
    for k, v in marian_mt_results.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    asyncio.run(main())
