import logging
import os
import time
import torch
import numpy as np
import soundfile as sf
from typing import Tuple, Optional

from backend import hardware

# Try to import OmniVoice
try:
    from omnivoice import OmniVoice
    OMNIVOICE_AVAILABLE = True
except ImportError:
    OMNIVOICE_AVAILABLE = False
    # We'll check and raise an error in __init__ if needed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OmniVoiceTTS:
    SUPPORTS_CLONING = True
    REQUIRES_SPEAKER_WAV = False  # falls back to a default voice if none given

    def __init__(
        self,
        model_name: str = "k2-fsa/OmniVoice",
        device: str = "auto",
    ):
        """
        Initializes the OmniVoice TTS model.

        Args:
            model_name (str): The Hugging Face model ID or path.
            device (str): Device to run the model on ("cpu", "cuda", "mps", "auto").
                          "auto" will use MPS if available, then CUDA, then CPU.
        """
        if not OMNIVOICE_AVAILABLE:
            raise ImportError(
                "OmniVoice package is not installed. "
                "Please install it with: pip install omnivoice"
            )

        self.model_name = model_name
        self.device = device

        # Backend selection: cuda -> mps -> rocm -> cpu (backend/hardware.py, shared across TTS-cloning engines)
        if self.device == "auto":
            self.device = hardware.detect_backend("tts_clone")
        logging.info(f"OmniVoiceTTS: using backend '{self.device}'.")

        # Load the model
        logging.info(f"OmniVoiceTTS: Loading model '{self.model_name}' on device '{self.device}'...")
        start_time = time.perf_counter()
        try:
            self.model = OmniVoice.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            )
            self.model.eval()  # Set to evaluation mode
            load_time = time.perf_counter() - start_time
            logging.info(
                f"OmniVoiceTTS: Model loaded successfully in {load_time:.2f}s."
            )
        except Exception as e:
            logging.error(f"OmniVoiceTTS: Failed to load model: {e}")
            raise

        # OmniVoice typically uses 24kHz sample rate
        self.sample_rate = 24000

    def synthesize(
        self,
        text: str,
        language: str = "sk",
        speaker_wav_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Synthesizes speech from text using OmniVoice with optional voice cloning.

        Args:
            text (str): The text to synthesize.
            language (str): The language of the text (e.g., "en", "cs", "sk").
                            OmniVoice uses ISO 639-1 codes or full language names.
            speaker_wav_path (Optional[str]): Path to reference WAV file for voice cloning.
                                            If None, uses the default voice.

        Returns:
            Tuple[np.ndarray, int, float]: A tuple containing the synthesized audio (numpy array),
                                            sample rate, and synthesis time in seconds.
        """
        start_time = time.perf_counter()

        try:
            # Prepare generation parameters
            # OmniVoice expects language as string, we'll use the language code directly
            # For voice cloning, we need to provide the speaker audio
            generate_kwargs = {
                "text": text,
                "language": language,
                # We can add other parameters like speed, etc. if needed
            }

            if speaker_wav_path is not None:
                if not os.path.exists(speaker_wav_path):
                    raise FileNotFoundError(f"Speaker WAV file not found: {speaker_wav_path}")
                generate_kwargs["speaker_audio"] = speaker_wav_path

            # Generate audio
            # The model's generate method returns the audio waveform and sample rate
            # We assume it returns a tuple (audio, sample_rate) or a dict
            # Based on the OmniVoice documentation, we'll check the return type
            result = self.model.generate(**generate_kwargs)

            # Handle different return types
            if isinstance(result, tuple) and len(result) == 2:
                audio_wav, sample_rate = result
            elif isinstance(result, dict) and "audio" in result and "sampling_rate" in result:
                audio_wav = result["audio"]
                sample_rate = result["sampling_rate"]
            elif isinstance(result, list):
                # OmniVoice may return a list of audio chunks
                audio_wav = result
                sample_rate = self.sample_rate
            else:
                # Assume it returns the audio directly and we know the sample rate
                audio_wav = result
                sample_rate = self.sample_rate

            # Ensure audio is numpy array
            if isinstance(audio_wav, list):
                # Concatenate list of audio chunks
                audio_wav = np.concatenate(audio_wav)
            elif torch.is_tensor(audio_wav):
                audio_wav = audio_wav.cpu().numpy()

            # Normalize audio if needed (OmniVoice might output in [-1, 1] or [0, 1])
            if audio_wav.dtype == np.float32 or audio_wav.dtype == np.float64:
                # Check if we need to normalize to [-1, 1] for soundfile
                if np.max(np.abs(audio_wav)) > 1.0:
                    audio_wav = audio_wav / np.max(np.abs(audio_wav))

            end_time = time.perf_counter()
            synthesis_time = end_time - start_time

            logging.info(
                f"OmniVoiceTTS: Synthesized text: '{text[:50]}...' "
                f"(language: {language}, speaker: {'cloned' if speaker_wav_path else 'default'}) "
                f"in {synthesis_time:.4f}s"
            )

            return audio_wav, sample_rate, synthesis_time

        except Exception as e:
            logging.error(f"OmniVoiceTTS: Error during synthesis: {e}")
            raise