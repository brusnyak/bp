from faster_whisper import WhisperModel
import numpy as np
from typing import Optional, Tuple


class FasterWhisperSTT:
    def __init__(
        self, model_size: str = "base", device: str = "auto", compute_type: str = "int8"
    ):
        """
        Initializes the FasterWhisperSTT model.

        Args:
            model_size (str): Size of the Whisper model to use (e.g., "tiny", "base", "small", "medium", "reset").
            device (str): Device to run the model on ("cpu", "cuda", "auto"). For M1/M2 Macs, "auto" should use MPS.
            compute_type (str): Type of computation to use (e.g., "int8", "float16", "float32").
        """
        self.model_size = model_size # Store model_size as an instance attribute
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(
            f"FasterWhisperSTT initialized with model_size={self.model_size}, device={device}, compute_type={compute_type}"
        )

    def transcribe_audio(
        self, audio_data: np.ndarray, sample_rate: int, language: Optional[str] = None, vad_filter: bool = True
    ) -> Tuple[list, float, Optional[str]]: # Changed return type to list of segments, transcription time, and detected language
        """
        Transcribes audio data using Faster-Whisper.

        Args:
            audio_data (np.ndarray): Audio data as a NumPy array (float32).
            sample_rate (int): Sample rate of the audio data.
            language (Optional[str]): Language of the audio. If None, it will be detected.

        Returns:
            Tuple[str, float]: A tuple containing the transcribed text and the transcription time in seconds.
        """
        # Faster-Whisper expects a file path or a numpy array of float32 at 16kHz
        # Ensure audio_data is float32 and resample if necessary (though for testing, we assume 16kHz)
        if sample_rate != 16000:
            # In a real scenario, you'd resample here. For now, assume input is 16kHz.
            print(
                "Warning: Faster-Whisper expects 16kHz audio. Resampling not implemented in this wrapper."
            )

        # Faster-Whisper's transcribe method expects a path or a float32 numpy array
        # We'll pass the numpy array directly.

        import time

        start_time = time.time()

        # Adjust Faster-Whisper's internal thresholds when an external VAD is used
        transcribe_options = {
            "language": language,
            "beam_size": 5,
            "vad_filter": vad_filter,
        }
        if not vad_filter: # If external VAD is enabled (and FasterWhisper's is disabled)
            transcribe_options["no_speech_threshold"] = None # Disable FasterWhisper's internal no-speech detection

        segments, info = self.model.transcribe(audio_data, **transcribe_options)

        end_time = time.time()
        transcription_time = end_time - start_time

        # Return the list of segments directly
        detected_language = info.language if info and hasattr(info, 'language') else None
        return list(segments), transcription_time, detected_language


if __name__ == "__main__":
    import soundfile as sf
    import os

    dummy_audio_path = "dummy_audio_faster.wav"
    sample_rate = 16000
    duration = 5  # seconds
    frequency = 440  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    dummy_audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sf.write(dummy_audio_path, dummy_audio_data, sample_rate)

    stt_model = FasterWhisperSTT(model_size="tiny")

    audio_data, sr = sf.read(dummy_audio_path)

    text, time = stt_model.transcribe_audio(audio_data, sr, language="en")
    print(f"Final Transcription: {text}")
    print(f"Transcription Time: {time:.2f}s")

    os.remove(dummy_audio_path)
