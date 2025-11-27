import pytest
import asyncio
import websockets
import json
import numpy as np
import soundfile as sf
import os
import time
from unittest.mock import AsyncMock, MagicMock
from starlette.websockets import WebSocketDisconnect

# Assuming backend.main is accessible in the path
# For testing, we might need to adjust sys.path or mock imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from backend.main import handle_audio_stream, initialize_all_models, AUDIO_SAMPLE_RATE, VAD_FRAME_DURATION, MIN_SPEECH_DURATION, SILENCE_TIMEOUT, STREAMING_CHUNK_LENGTH
from backend.utils.audio_utils import resample_audio # Import resample_audio

# Mock WebSocket for testing
class MockWebSocket:
    def __init__(self):
        self.received_messages = []
        self.accepted = False
        self.client = MagicMock(host="test_host", port=12345)

    async def accept(self):
        self.accepted = True

    async def receive(self):
        # Simulate receiving audio bytes
        if self.audio_to_send:
            chunk = self.audio_to_send.pop(0)
            return {"type": "websocket.receive", "bytes": chunk.tobytes()}
        else:
            # After sending all audio, simulate a disconnect
            raise WebSocketDisconnect(code=1000, reason="Test finished")

    async def receive_bytes(self):
        # This is for the original handle_audio_stream, not the refactored one
        if self.audio_to_send:
            chunk = self.audio_to_send.pop(0)
            return chunk.tobytes()
        else:
            raise WebSocketDisconnect(code=1000, reason="Test finished")

    async def send_text(self, message):
        self.received_messages.append(json.loads(message))

    async def send_json(self, message):
        self.received_messages.append(message)

    async def send_bytes(self, message):
        self.received_messages.append({"type": "audio_bytes", "length": len(message)})

    def set_audio_to_send(self, audio_chunks):
        self.audio_to_send = audio_chunks

# Helper to create audio chunks
def create_audio_chunks(audio_np, chunk_length_seconds):
    chunk_size = int(AUDIO_SAMPLE_RATE * chunk_length_seconds)
    chunks = []
    for i in range(0, len(audio_np), chunk_size):
        chunks.append(audio_np[i:i + chunk_size])
    return chunks

@pytest.mark.asyncio
async def test_vad_speech_detection_and_processing():
    """
    Test VAD's ability to detect speech and trigger pipeline processing.
    Use a short speech segment followed by silence.
    """
    # Initialize models directly within the test
    await initialize_all_models(
        source_lang="en",
        target_lang="sk",
        tts_model_choice="piper",
        vad_enabled_param=True
    )

    websocket = MockWebSocket()
    await websocket.accept() # Simulate WebSocket connection acceptance
    
    # Load a known speech file
    speech_file_path = "test/Can you hear me_.wav"
    if not os.path.exists(speech_file_path):
        pytest.skip(f"Test audio file not found: {speech_file_path}")
    
    audio_np, sr = sf.read(speech_file_path, dtype='float32') # Explicitly read as float32
    # Ensure audio is mono
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    
    # Resample audio to AUDIO_SAMPLE_RATE if necessary
    if sr != AUDIO_SAMPLE_RATE:
        audio_np = resample_audio(audio_np, original_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
        sr = AUDIO_SAMPLE_RATE # Update sample rate to the new target

    # Create a short speech segment (e.g., 1 second)
    speech_segment = audio_np[:AUDIO_SAMPLE_RATE] # 1 second of speech
    
    # Add some silence after the speech
    silence_duration_seconds = SILENCE_TIMEOUT * 2 # Enough silence to trigger end of speech
    silence_segment = np.zeros(int(AUDIO_SAMPLE_RATE * silence_duration_seconds), dtype=np.float32)

    full_audio = np.concatenate([speech_segment, silence_segment])
    
    # Chunk the audio for streaming
    chunk_length_seconds = VAD_FRAME_DURATION / 1000 # Send in VAD frame durations
    audio_chunks = create_audio_chunks(full_audio, chunk_length_seconds)
    websocket.set_audio_to_send(audio_chunks)

    start_time = time.time()
    await handle_audio_stream(websocket)
    end_time = time.time()

    # Assertions
    assert websocket.accepted is True
    
    # Check for transcription and translation results
    transcription_found = False
    translation_found = False
    for msg in websocket.received_messages:
        if msg.get("type") == "transcription_result" and msg.get("transcribed"):
            transcription_found = True
            print(f"Transcription: {msg['transcribed']}")
        if msg.get("type") == "translation_result" and msg.get("translated"):
            translation_found = True
            print(f"Translation: {msg['translated']}")
        if msg.get("type") == "final_metrics":
            print(f"Final Metrics: {msg['metrics']}")

    assert transcription_found, "No transcription result found."
    assert translation_found, "No translation result found."
    
    # Check if VAD processing time is logged (DEBUG level)
    # This requires capturing logs, which is more complex. For now, rely on functional output.
    # You would typically use a logging capture fixture for this.

    print(f"Test completed in {end_time - start_time:.2f} seconds.")

@pytest.mark.asyncio
async def test_vad_silence_handling(): # Removed setup_models fixture
    """
    Test VAD's ability to correctly identify and ignore silence.
    No transcription or translation should occur for pure silence.
    """
    # Initialize models directly within the test
    await initialize_all_models(
        source_lang="en",
        target_lang="sk",
        tts_model_choice="piper",
        vad_enabled_param=True
    )

    websocket = MockWebSocket()
    await websocket.accept() # Simulate WebSocket connection acceptance
    
    # Create a long silence segment
    silence_duration_seconds = 5 # 5 seconds of silence
    silence_segment = np.zeros(int(AUDIO_SAMPLE_RATE * silence_duration_seconds), dtype=np.float32)

    # Chunk the audio for streaming
    chunk_length_seconds = VAD_FRAME_DURATION / 1000
    audio_chunks = create_audio_chunks(silence_segment, chunk_length_seconds)
    websocket.set_audio_to_send(audio_chunks)

    await handle_audio_stream(websocket)

    # Assertions
    assert websocket.accepted is True
    
    # No transcription or translation should be sent for pure silence
    for msg in websocket.received_messages:
        assert msg.get("type") not in ["transcription_result", "translation_result"], \
            f"Unexpected processing for silence: {msg.get('type')}"
        assert msg.get("type") != "audio_bytes", "Unexpected audio bytes for silence."

    print("Test completed: Silence correctly handled (no processing triggered).")

@pytest.mark.asyncio
async def test_vad_aggressiveness_and_short_speech(): # Removed setup_models fixture
    """
    Test VAD with a very short speech burst to see if it's detected.
    Given VAD_AGGRESSIVENESS=2 and MIN_SPEECH_DURATION=0.2s,
    a very short burst should be detected and processed.
    """
    # Initialize models directly within the test
    await initialize_all_models(
        source_lang="en",
        target_lang="sk",
        tts_model_choice="piper",
        vad_enabled_param=True
    )

    websocket = MockWebSocket()
    await websocket.accept() # Simulate WebSocket connection acceptance
    
    speech_file_path = "test/Can you hear me_.wav"
    if not os.path.exists(speech_file_path):
        pytest.skip(f"Test audio file not found: {speech_file_path}")
    
    audio_np, sr = sf.read(speech_file_path, dtype='float32') # Explicitly read as float32
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    
    if sr != AUDIO_SAMPLE_RATE:
        audio_np = resample_audio(audio_np, original_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
        sr = AUDIO_SAMPLE_RATE

    # Create a very short speech segment (e.g., 0.2 seconds, less than MIN_SPEECH_DURATION)
    short_speech_segment = audio_np[:int(AUDIO_SAMPLE_RATE * 0.2)]
    silence_segment = np.zeros(int(AUDIO_SAMPLE_RATE * SILENCE_TIMEOUT * 2), dtype=np.float32)
    full_audio = np.concatenate([short_speech_segment, silence_segment])

    chunk_length_seconds = VAD_FRAME_DURATION / 1000
    audio_chunks = create_audio_chunks(full_audio, chunk_length_seconds)
    websocket.set_audio_to_send(audio_chunks)

    await handle_audio_stream(websocket)

    transcription_found = False
    for msg in websocket.received_messages:
        if msg.get("type") == "transcription_result" and msg.get("transcribed"):
            transcription_found = True
            print(f"Transcription for short speech: {msg['transcribed']}")
            break
    
    # Depending on VAD sensitivity and MIN_SPEECH_DURATION, this might or might not be detected.
    # With current aggressive settings, it should ideally be detected and processed.
    assert transcription_found, "Short speech segment was not transcribed."
    print("Test completed: Short speech segment detection checked.")

@pytest.mark.asyncio
async def test_vad_long_speech_streaming(): # Removed setup_models fixture
    """
    Test VAD's ability to handle a longer speech segment,
    expecting multiple streaming results before a final one.
    """
    # Initialize models directly within the test
    await initialize_all_models(
        source_lang="en",
        target_lang="sk",
        tts_model_choice="piper",
        vad_enabled_param=True
    )

    websocket = MockWebSocket()
    await websocket.accept() # Simulate WebSocket connection acceptance
    
    speech_file_path = "test/My test speech_xtts_speaker_clean.wav"
    if not os.path.exists(speech_file_path):
        pytest.skip(f"Test audio file not found: {speech_file_path}")
    
    audio_np, sr = sf.read(speech_file_path, dtype='float32') # Explicitly read as float32
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    
    if sr != AUDIO_SAMPLE_RATE:
        audio_np = resample_audio(audio_np, original_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
        sr = AUDIO_SAMPLE_RATE

    # Use a longer segment (e.g., 3-4 seconds)
    long_speech_segment = audio_np[:int(AUDIO_SAMPLE_RATE * 4)] # 4 seconds of speech
    silence_segment = np.zeros(int(AUDIO_SAMPLE_RATE * SILENCE_TIMEOUT * 2), dtype=np.float32)
    full_audio = np.concatenate([long_speech_segment, silence_segment])

    chunk_length_seconds = VAD_FRAME_DURATION / 1000
    audio_chunks = create_audio_chunks(full_audio, chunk_length_seconds)
    websocket.set_audio_to_send(audio_chunks)

    await handle_audio_stream(websocket)

    streaming_transcriptions = 0
    final_transcription_found = False
    for msg in websocket.received_messages:
        if msg.get("type") == "transcription_result":
            if msg.get("is_final") == False and msg.get("transcribed"):
                streaming_transcriptions += 1
                print(f"Streaming Transcription: {msg['transcribed']}")
            elif msg.get("is_final") == True and msg.get("transcribed"):
                final_transcription_found = True
                print(f"Final Transcription: {msg['transcribed']}")
    
    assert streaming_transcriptions >= 1, "Expected at least one streaming transcription."
    assert final_transcription_found, "Expected a final transcription."
    print("Test completed: Long speech streaming checked.")
