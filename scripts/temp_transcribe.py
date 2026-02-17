import sys
import os
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.utils.audio_utils import load_audio

# Add the parent directory (BP) to sys.path to import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

audio_file = 'test/my_voice_reference_10s.wav'
transcript_output_file = 'test/my_voice_reference_10s_transcript.txt'

try:
    audio_data, sample_rate = load_audio(audio_file, target_sr=16000)
    stt_model = FasterWhisperSTT(model_size='base', compute_type='int8')
    segments, _, _ = stt_model.transcribe_audio(audio_data, sample_rate)
    transcript = ' '.join([s.text for s in segments]).strip()
    print(f'Transcript for {audio_file}: {transcript}')
    with open(transcript_output_file, 'w') as f:
        f.write(transcript)
    print(f'Transcript saved to {transcript_output_file}')
except Exception as e:
    print(f"Error during transcription: {e}")
    sys.exit(1)
