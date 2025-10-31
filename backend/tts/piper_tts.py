from piper import PiperVoice, SynthesisConfig
import numpy as np
import soundfile as sf
import io
from typing import Tuple, Optional

class PiperTTS:
    def __init__(self, model_path: str, speaker_id: int = 0, **kwargs):
        """
        Initializes the Piper TTS model.

        Args:
            model_path (str): Path to the Piper ONNX model file.
            speaker_id (int): The ID of the speaker to use for synthesis.
            **kwargs: Additional arguments for PiperVoice.load (e.g., use_mps=True).
        """
        # PiperVoice.load expects only the model_path, it will automatically look for the .json config
        # Add use_mps=True for Apple Silicon MPS acceleration
        self.model = PiperVoice.load(model_path, **kwargs)
        self.speaker_id = speaker_id
        print(f"PiperTTS initialized with model: {model_path}, speaker_id: {speaker_id}")

    def synthesize_speech(self, text: str, language: str, speaker_id: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Synthesizes speech from text using Piper.

        Args:
            text (str): The text to synthesize.
            language (str): The language of the text (e.g., "en", "cs", "sk").
                            Piper uses ISO 639-1 codes. For Slovak, 'cs' (Czech) is a good proxy.
            speaker_id (Optional[int]): Override default speaker ID for this synthesis.

        Returns:
            Tuple[np.ndarray, int, float]: A tuple containing the synthesized audio (numpy array),
                                           sample rate, and synthesis time in seconds.
        """
        actual_speaker_id = speaker_id if speaker_id is not None else self.speaker_id
        
        import time
        import wave # Import wave module
        start_time = time.perf_counter()

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            self.model.synthesize_wav(text, wav_file)
        audio_buffer.seek(0) # Rewind the buffer to read its content

        audio_waveform, sample_rate = sf.read(audio_buffer)
        
        end_time = time.perf_counter()
        synthesis_time = end_time - start_time

        print(f"Piper synthesized text: '{text}' in {synthesis_time:.4f}s (language: {language}, speaker: {actual_speaker_id})")
        return audio_waveform, sample_rate, synthesis_time

if __name__ == "__main__":
    # Example Usage:
    import soundfile as sf
    import os

    # To run this example, you need to download a Piper model and its config.
    # For example, for a Czech model (as a proxy for Slovak):
    # Model: https://huggingface.co/rhasspy/piper-voices/resolve/main/cs/cs_CZ/vits/fairseq/medium/cs_CZ-fairseq-medium.onnx
    # Config: https://huggingface.co/rhasspy/piper-voices/resolve/main/cs/cs_CZ/vits/fairseq/medium/cs_CZ-fairseq-medium.json
    
    # Create dummy model and config files for demonstration if they don't exist
    dummy_model_path = "dummy_piper_model.onnx"
    dummy_config_path = "dummy_piper_model.json"

    if not os.path.exists(dummy_model_path):
        print(f"WARNING: {dummy_model_path} not found. Piper TTS example will use mock data.")
        # Create a minimal dummy ONNX file (not functional, just to satisfy file existence)
        with open(dummy_model_path, "wb") as f:
            f.write(b'\x08\x01\x12\x00') # Minimal ONNX header
    
    if not os.path.exists(dummy_config_path):
        print(f"WARNING: {dummy_config_path} not found. Piper TTS example will use mock data.")
        # Create a minimal dummy config file
        with open(dummy_config_path, "w") as f:
            f.write('{"audio": {"sample_rate": 22050}, "speakers": {"default": 0}}')

    # Mock PiperVoice.load_from_file for testing without actual model files
    class MockPiperVoice:
        def __init__(self, sample_rate=22050):
            class MockConfig:
                def __init__(self, sr):
                    self.sample_rate = sr
            self.config = MockConfig(sample_rate)
            self.speaker_id_map = {"default": 0} # Mock speaker map

        def synthesize_text(self, text, speaker_id=0):
            # Mock synthesis: generate a simple sine wave
            duration = len(text) * 0.05  # Estimate duration
            t = np.linspace(0, duration, int(self.config.sample_rate * duration), endpoint=False)
            audio_waveform_float32 = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
            audio_waveform_int16 = (audio_waveform_float32 * 32767).astype(np.int16)
            return audio_waveform_int16.tobytes()

    # Mock PiperVoice.load for testing without actual model files
    class MockPiperVoice:
        def __init__(self, sample_rate=22050):
            class MockConfig:
                def __init__(self, sr):
                    self.sample_rate = sr
            self.config = MockConfig(sample_rate)
            self.speaker_id_map = {"default": 0} # Mock speaker map

        def synthesize_wav(self, text, wav_file, speaker_id=0):
            # Mock synthesis: generate a simple sine wave
            duration = len(text) * 0.05  # Estimate duration
            t = np.linspace(0, duration, int(self.config.sample_rate * duration), endpoint=False)
            audio_waveform_float32 = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
            
            # Write mock WAV header and data
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2) # 16-bit PCM
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes((audio_waveform_float32 * 32767).astype(np.int16).tobytes())

    # Temporarily replace PiperVoice.load for testing
    original_load = PiperVoice.load
    PiperVoice.load = lambda mp, **kwargs: MockPiperVoice()

    try:
        # Initialize PiperTTS with dummy paths
        piper_tts_model = PiperTTS(model_path=dummy_model_path, speaker_id=0)

        # Synthesize speech in Czech (as proxy for Slovak)
        text_cs = "Ahoj, toto je testovacia veta v češtine."
        audio_cs, sr_cs, time_cs = piper_tts_model.synthesize_speech(text_cs, "cs")
        sf.write("output_cs_piper.wav", audio_cs, sr_cs)
        print(f"Saved Piper Czech speech to output_cs_piper.wav (Time: {time_cs:.4f}s)")

        # Synthesize speech in English (if the model supports it, or use a generic voice)
        text_en = "Hello, this is a test sentence in English from Piper."
        audio_en, sr_en, time_en = piper_tts_model.synthesize_speech(text_en, "en")
        sf.write("output_en_piper.wav", audio_en, sr_en)
        print(f"Saved Piper English speech to output_en_piper.wav (Time: {time_en:.4f}s)")

    finally:
        # Restore original load
        PiperVoice.load = original_load
        # Clean up dummy files
        if os.path.exists(dummy_model_path):
            os.remove(dummy_model_path)
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)
        if os.path.exists("output_cs_piper.wav"):
            os.remove("output_cs_piper.wav")
        if os.path.exists("output_en_piper.wav"):
            os.remove("output_en_piper.wav")
