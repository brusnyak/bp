import numpy as np
import soundfile as sf
from typing import Tuple, Optional
import os
import sys
import logging

# Removed explicit FFmpeg environment variable settings as soundfile is now used for audio I/O
# and torchaudio is no longer a core dependency for the pipeline.
# if sys.platform == "darwin":
#     venv_lib_path = os.path.join(os.path.dirname(sys.executable), "..", "lib")
#     homebrew_lib_path = "/opt/homebrew/lib"
    
#     dyld_paths = []
#     if os.path.exists(venv_lib_path):
#         dyld_paths.append(venv_lib_path)
#     if os.path.exists(homebrew_lib_path):
#         dyld_paths.append(homebrew_lib_path)
    
#     if dyld_paths:
#         new_dyld_library_path = ":".join(dyld_paths)
#         if "DYLD_LIBRARY_PATH" in os.environ:
#             os.environ["DYLD_LIBRARY_PATH"] = f"{new_dyld_library_path}:{os.environ['DYLD_LIBRARY_PATH']}"
#         else:
#             os.environ["DYLD_LIBRARY_PATH"] = new_dyld_library_path
#         logging.info(f"AudioUtils: DYLD_LIBRARY_PATH set to: {os.environ['DYLD_LIBRARY_PATH']}")
#     else:
#         logging.warning("AudioUtils: Neither venv lib path nor Homebrew lib path found. FFmpeg linking might fail.")

# Removed explicit FFmpeg environment variable settings as soundfile is now used for audio I/O
# and torchaudio is no longer a core dependency for the pipeline.
# os.environ["TORCH_USE_LIBAV"] = "0"
# os.environ["TORCH_USE_EXTERNAL_FFMPEG"] = "1" # Ensure external FFmpeg is preferred
# logging.info("AudioUtils: TORCH_USE_LIBAV set to 0. TORCH_USE_EXTERNAL_FFMPEG set to 1.")


def load_audio(
    file_path: str, target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and optionally resamples it.

    Args:
        file_path (str): Path to the audio file.
        target_sr (Optional[int]): Target sample rate. If None, uses original sample rate.

    Returns:
        Tuple[np.ndarray, int]: Audio data as a NumPy array (float32) and its sample rate.
    """
    audio_data, sr = sf.read(file_path, dtype="float32")

    # sf.read can return stereo audio, convert to mono if necessary
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    logging.debug(f"AudioUtils: load_audio - Before resampling: audio_data min: {np.min(audio_data)}, max: {np.max(audio_data)}, sr: {sr}")

    if target_sr is not None and sr != target_sr:
        audio_data = resample_audio(audio_data, original_sr=sr, target_sr=target_sr)
        sr = target_sr
        logging.debug(f"AudioUtils: load_audio - After resampling: audio_data min: {np.min(audio_data)}, max: {np.max(audio_data)}, sr: {sr}")

    return audio_data, sr

def resample_audio(audio_data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """
    Resamples audio data to a new sample rate.

    Args:
        audio_data (np.ndarray): Input audio data.
        original_sr (int): Original sample rate of the audio data.
        target_sr (int): Target sample rate for resampling.

    Returns:
        np.ndarray: Resampled audio data.
    """
    from scipy.signal import resample_poly
    if original_sr == target_sr:
        return audio_data
    
    logging.debug(f"AudioUtils: resample_audio - Input audio_data min: {np.min(audio_data)}, max: {np.max(audio_data)}")

    # Calculate the number of samples for the new array
    num_samples = int(round(len(audio_data) * float(target_sr) / original_sr))
    
    # Perform resampling
    resampled_data = resample_poly(audio_data, target_sr, original_sr)
    
    logging.debug(f"AudioUtils: resample_audio - Output resampled_data min: {np.min(resampled_data)}, max: {np.max(resampled_data)}")
    return resampled_data


def save_audio(file_path: str, audio_data: np.ndarray, sample_rate: int):
    """
    Saves audio data to a file using soundfile.

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
    assert np.isclose(np.max(np.abs(normalized_audio)), 1.0) or np.all(
        normalized_audio == 0
    )

    # Test saving
    output_audio_path = "output_resampled_normalized.wav"
    save_audio(output_audio_path, normalized_audio, loaded_sr)
    print(f"Saved processed audio to {output_audio_path}")

    # Clean up
    os.remove(dummy_audio_path)
    os.remove(output_audio_path)
    print("Cleaned up dummy files.")
