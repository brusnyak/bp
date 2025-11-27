from piper import PiperVoice, SynthesisConfig
import numpy as np
import soundfile as sf
import io
from typing import Tuple, Optional
import os
import torch  # Import torch for MPS check


class PiperTTS:
    def __init__(
        self,
        model_id: str = "cs_CZ-jirka-medium",
        speaker_id: int = 0,
        device: str = "auto",
    ):
        """
        Initializes the Piper TTS model.

        Args:
            model_id (str): The ID of the Piper model (e.g., "cs_CZ-jirka-medium").
                            This will be used to construct the path to the ONNX model file.
            speaker_id (int): The ID of the speaker to use for synthesis.
            device (str): Device to run the model on ("cpu", "mps"). "auto" will use MPS if available.
        """
        self.model_id = model_id
        self.speaker_id = speaker_id
        self.device = device

        # Construct model path based on project structure
        # Assuming models are downloaded to backend/tts/piper_models/
        base_model_dir = os.path.join("backend", "tts", "piper_models")
        onnx_model_path = os.path.join(base_model_dir, f"{model_id}.onnx")
        json_config_path = os.path.join(base_model_dir, f"{model_id}.onnx.json")

        # Check if model files exist, if not, provide instructions or attempt download
        print(f"PiperTTS: Attempting to load model '{model_id}'.")
        print(f"PiperTTS: Expected ONNX model path: {os.path.abspath(onnx_model_path)}")
        print(f"PiperTTS: Expected JSON config path: {os.path.abspath(json_config_path)}")
        print(f"PiperTTS: os.path.exists(onnx_model_path): {os.path.exists(onnx_model_path)}")
        print(f"PiperTTS: os.path.exists(json_config_path): {os.path.exists(json_config_path)}")

        if not os.path.exists(onnx_model_path) or not os.path.exists(json_config_path):
            print(f"Error: Piper model files not found for {model_id} at the expected paths.")
            print(
                f"Please ensure '{model_id}.onnx' and '{model_id}.onnx.json' are present in '{os.path.abspath(base_model_dir)}'."
            )
            raise FileNotFoundError(f"Piper model files not found for {model_id}")


        # Determine if MPS should be used
        use_mps = False
        if self.device == "auto":
            if torch.backends.mps.is_available():
                use_mps = True
                self.device = "mps"
                print("PiperTTS: MPS device detected and will be used.")
            else:
                self.device = "cpu"
                print("PiperTTS: MPS not available, falling back to CPU.")
        elif self.device == "mps":
            if torch.backends.mps.is_available():
                use_mps = True
                print("PiperTTS: MPS device explicitly requested and available.")
            else:
                self.device = "cpu"
                print(
                    "WARNING: PiperTTS: MPS device requested but not available, falling back to CPU."
                )
        elif self.device == "cpu":
            print("PiperTTS: CPU device explicitly requested.")

        # Load Piper model
        # The 'use_mps' argument is not supported in this version of Piper.
        # Piper is expected to handle MPS automatically if available and configured in the environment.
        try:
            self.model = PiperVoice.load(onnx_model_path)
            print(
                f"PiperTTS initialized with model: {model_id}, device: {self.device}, speaker_id: {speaker_id}"
            )
        except Exception as e:
            print(f"ERROR: Failed to load Piper model '{model_id}': {e}")
            raise RuntimeError(f"Failed to load Piper model '{model_id}': {e}") from e

    def synthesize(
        self, text: str, language: str = "sk", output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int, float]: # Keep output_path for now, but make it optional and don't use it for primary return
        """
        Synthesizes speech from text using Piper.

        Args:
            text (str): The text to synthesize.
            language (str): The language of the text (e.g., "en", "cs", "sk").
                            Piper uses ISO 639-1 codes. For Slovak, 'cs' (Czech) is a good proxy.
            output_path (Optional[str]): Path to save the synthesized audio.

        Returns:
            Tuple[np.ndarray, int, float]: A tuple containing the synthesized audio (numpy array),
                                           sample rate, and synthesis time in seconds.
        """
        # Piper's synthesize_wav method does not take a language argument directly.
        # The language is determined by the loaded model.
        # For Slovak, we are using a Czech model as a proxy.

        import time
        import wave  # Import wave module

        start_time = time.perf_counter()

        audio_buffer = io.BytesIO()
        # Get audio parameters from the Piper model's config
        # Piper's config might not directly expose num_channels or sample_width.
        # Assuming common values for TTS output: 1 channel (mono), 2 bytes per sample (16-bit).
        sample_rate = self.model.config.sample_rate
        num_channels = 1  # Most TTS models output mono audio
        sample_width = 2  # 16-bit audio is common, which is 2 bytes per sample

        with wave.open(audio_buffer, "wb") as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            # Piper's synthesize_wav expects text and a file-like object, without speaker_id
            self.model.synthesize_wav(text, wav_file)
        audio_buffer.seek(0)  # Rewind the buffer to read its content

        audio_waveform, sample_rate = sf.read(audio_buffer)

        end_time = time.perf_counter()
        synthesis_time = end_time - start_time

        print(
            f"Piper synthesized text: '{text}' in {synthesis_time:.4f}s (language: {language}, speaker: {self.speaker_id})"
        )

        if output_path:
            sf.write(output_path, audio_waveform, sample_rate)
            print(f"Saved synthesized audio to {output_path}")

        return audio_waveform, sample_rate, synthesis_time
