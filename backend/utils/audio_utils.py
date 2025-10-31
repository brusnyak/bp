import numpy as np
import soundfile as sf
import torchaudio
import torch
from typing import Tuple, Union
import os

from typing import Optional

def load_audio(file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and optionally resamples it.

    Args:
        file_path (str): Path to the audio file.
        target_sr (Optional[int]): Target sample rate. If None, uses original sample rate.

    Returns:
        Tuple[np.ndarray, int]: Audio data as a NumPy array (float32) and its sample rate.
    """
    audio_data, sr = sf.read(file_path, dtype='float32')

    # sf.read can return stereo audio, convert to mono if necessary
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if target_sr is not None and sr != target_sr:
        # Use torchaudio for resampling for better quality and GPU support if available
        audio_tensor = torch.from_numpy(audio_data).float()
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio_data = resampler(audio_tensor).numpy()
        sr = target_sr
    
    return audio_data, sr

def save_audio(file_path: str, audio_data: np.ndarray, sample_rate: int):
    """
    Saves audio data to a file.

    Args:
        file_path (str): Path to save the audio file.
        audio_data (np.ndarray): Audio data as a NumPy array (float32).
        sample_rate (int): Sample rate of the audio data.
    """
    sf.write(file_path, audio_data, sample_rate)

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalizes audio data to a -1.0 to 1.0 range.

    Args:
        audio_data (np.ndarray): Input audio data.

    Returns:
        np.ndarray: Normalized audio data.
    """
    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        return audio_data / max_abs
    return audio_data

if __name__ == "__main__":
    # Example Usage:
    import os

    # Create a dummy audio file for testing
    dummy_audio_path = "dummy_test_audio.wav"
    sample_rate_orig = 44100
    duration = 3  # seconds
    frequency = 1000  # Hz
    t = np.linspace(0, duration, int(sample_rate_orig * duration), endpoint=False)
    dummy_audio_data_orig = 0.6 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    save_audio(dummy_audio_path, dummy_audio_data_orig, sample_rate_orig)
    print(f"Created dummy audio at {dummy_audio_path} with SR {sample_rate_orig}")

    # Test loading and resampling
    target_sr = 16000
    loaded_audio, loaded_sr = load_audio(dummy_audio_path, target_sr=target_sr)
    print(f"Loaded audio with SR {loaded_sr}. Shape: {loaded_audio.shape}")
    assert loaded_sr == target_sr
    assert loaded_audio.dtype == np.float32

    # Test normalization
    normalized_audio = normalize_audio(loaded_audio)
    print(f"Normalized audio max abs: {np.max(np.abs(normalized_audio))}")
    assert np.isclose(np.max(np.abs(normalized_audio)), 1.0) or np.all(normalized_audio == 0)

    # Test saving
    output_audio_path = "output_resampled_normalized.wav"
    save_audio(output_audio_path, normalized_audio, loaded_sr)
    print(f"Saved processed audio to {output_audio_path}")

    # Clean up
    os.remove(dummy_audio_path)
    os.remove(output_audio_path)
    print("Cleaned up dummy files.")
