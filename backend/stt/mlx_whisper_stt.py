import mlx.core as mx
from mlx_whisper.transcribe import transcribe
import numpy as np
from typing import Optional, Tuple

class MLXWhisperSTT:
    def __init__(self, model_size: str = "tiny", compute_type: str = "int8"):
        """
        Initializes the MLXWhisperSTT model.

        Args:
            model_size (str): Size of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "large").
                              This should correspond to a Hugging Face model ID (e.g., "mlx-community/whisper-tiny").
            compute_type (str): Type of computation to use (e.g., "int8", "float16", "float32").
                                Note: MLX handles device automatically on Apple Silicon.
        """
        self.model_id = f"mlx-community/whisper-{model_size}" if "/" not in model_size else model_size
        self.compute_type = compute_type
        print(f"MLXWhisperSTT initialized with model_id={self.model_id}, compute_type={compute_type}")

    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, language: Optional[str] = None) -> Tuple[str, float]:
        """
        Transcribes audio data using MLX-Whisper.

        Args:
            audio_data (np.ndarray): Audio data as a NumPy array (float32).
            sample_rate (int): Sample rate of the audio data.
            language (Optional[str]): Language of the audio. If None, it will be detected.

        Returns:
            Tuple[str, float]: A tuple containing the transcribed text and the transcription time in seconds.
        """
        audio_mx = mx.array(audio_data.astype(np.float32))

        import time
        start_time = time.time()

        result = transcribe(audio_mx, path_or_hf_repo=self.model_id, language=language)
        
        transcribed_text = result['text'].strip()
        
        end_time = time.time()
        transcription_time = end_time - start_time

        print(f"Transcribed: '{transcribed_text}' in {transcription_time:.2f}s (language: {language or 'detected'})")
        return transcribed_text, transcription_time

if __name__ == "__main__":
    import soundfile as sf
    import os

    dummy_audio_path = "dummy_audio.wav"
    sample_rate = 16000
    duration = 5  # seconds
    frequency = 440  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    dummy_audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(dummy_audio_path, dummy_audio_data, sample_rate)

    stt_model = MLXWhisperSTT(model_size="tiny")
    
    audio_data, sr = sf.read(dummy_audio_path)

    text, time = stt_model.transcribe_audio(audio_data, sr, language="en")
    print(f"Final Transcription: {text}")
    print(f"Transcription Time: {time:.2f}s")

    os.remove(dummy_audio_path)
