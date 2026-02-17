import os
import json
import shutil
import uuid
import asyncio
import numpy as np
import soundfile as sf
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.main import SPEAKER_VOICES_DIR, SPEAKER_VOICES_METADATA_FILE, AUDIO_SAMPLE_RATE, _read_speaker_voices_metadata, _write_speaker_voices_metadata

async def add_default_voices():
    print("Starting to add default voices...")

    test_voices_dir = "test"
    
    # Ensure speaker_voices directory exists
    os.makedirs(SPEAKER_VOICES_DIR, exist_ok=True)

    # Initialize STT model for transcription
    print("Initializing FasterWhisperSTT model...")
    stt_model = FasterWhisperSTT(model_size="tiny", compute_type="int8")
    print("FasterWhisperSTT model initialized.")

    # Get existing metadata
    metadata = _read_speaker_voices_metadata()
    existing_voice_names = {v["name"] for v in metadata}

    wav_files_to_add = [
        "Can you hear me_.wav",
        "My test speech_xtts_speaker_clean.wav",
        "Voice-Training.wav",
        "resampled_Voice-Training.wav",
        "hello.wav"
    ]

    for filename in wav_files_to_add:
        source_path = os.path.join(test_voices_dir, filename)
        destination_path = os.path.join(SPEAKER_VOICES_DIR, filename)
        voice_name = filename.replace(".wav", "")

        if not os.path.exists(source_path):
            print(f"Warning: Source WAV file not found: {source_path}. Skipping.")
            continue

        if voice_name in existing_voice_names:
            print(f"Voice '{voice_name}' already exists in metadata. Skipping.")
            continue

        print(f"Processing voice: {filename}")

        # Copy the WAV file to the speaker_voices directory
        shutil.copy(source_path, destination_path)
        print(f"Copied '{filename}' to '{SPEAKER_VOICES_DIR}'.")

        # Transcribe the audio
        audio_np, sr = sf.read(destination_path, dtype='float32')
        if sr != AUDIO_SAMPLE_RATE:
            # Resample if necessary
            from backend.utils.audio_utils import resample_audio
            audio_np = resample_audio(audio_np, original_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
            sr = AUDIO_SAMPLE_RATE
            sf.write(destination_path, audio_np, sr, format="WAV") # Overwrite with resampled audio

        loop = asyncio.get_event_loop()
        transcribed_segments, _, detected_lang = await loop.run_in_executor(
            None,
            lambda: stt_model.transcribe_audio(
                audio_np,
                AUDIO_SAMPLE_RATE,
                language=None, # Auto-detect language
                vad_filter=True
            ),
        )
        transcribed_text = " ".join([s.text for s in transcribed_segments]) if transcribed_segments else ""
        final_lang = detected_lang if detected_lang else "en" # Default to 'en' if detection fails
        print(f"  Transcribed: '{transcribed_text}' (Detected Lang: {final_lang})")

        # Add to metadata
        new_voice_entry = {
            "id": str(uuid.uuid4()), # Generate a unique ID
            "name": voice_name,
            "language": final_lang,
            "path": destination_path,
            "transcribed_text": transcribed_text,
            "user_id": None # Default voice, accessible to all users
        }
        metadata.append(new_voice_entry)
        existing_voice_names.add(voice_name) # Add to set to prevent duplicates

    _write_speaker_voices_metadata(metadata)
    print(f"Updated {SPEAKER_VOICES_METADATA_FILE} with {len(metadata)} voices.")
    print("Finished adding default voices.")

if __name__ == "__main__":
    asyncio.run(add_default_voices())
