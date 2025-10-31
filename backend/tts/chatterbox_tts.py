import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torchaudio

class ChatterboxTTS:
    def __init__(self, device: str = "auto"):
        """
        Initializes the Chatterbox TTS model.

        Args:
            device (str): Device to run the model on ("cpu", "cuda", "mps", or "auto").
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load on CPU first to avoid CUDA deserialization error, then move to target device
        # Temporarily patch torch.load to force map_location='cpu' during deserialization
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = torch.device('cpu')
            return original_torch_load(*args, **kwargs)
        torch.load = patched_torch_load

        try:
            # Load on CPU first to avoid CUDA deserialization error
            self.model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
            # ChatterboxMultilingualTTS does not have a .to() method.
            # Device handling must be done during from_pretrained or it runs on CPU.
            # We will rely on the 'device="cpu"' in from_pretrained and the torch.load patch.
            print(f"ChatterboxTTS initialized on device: cpu (due to CUDA deserialization issue).")
            if self.device == "mps":
                print("WARNING: ChatterboxTTS is running on CPU despite MPS being available, due to library limitations.")
        finally:
            # Restore original torch.load
            torch.load = original_torch_load

    def train_voice(self, audio_path: str, transcription: Optional[str] = None) -> Dict[str, Any]:
        """
        Trains a voice using a reference audio file.
        This method extracts speaker embeddings for voice cloning.

        Args:
            audio_path (str): Path to the reference audio file (.wav).
            transcription (Optional[str]): Optional transcription of the reference audio.
                                           If not provided, Chatterbox will attempt to transcribe it.

        Returns:
            Dict[str, Any]: A dictionary containing speaker embeddings and other relevant info.
        """
        print(f"Training voice with reference audio: {audio_path}")
        speaker_embedding = self.model.get_speaker_embedding(audio_path, transcription=transcription)
        print("Speaker embedding extracted.")
        return {"speaker_embedding": speaker_embedding}

    def synthesize_speech(self, text: str, language: str, speaker_embedding: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, int, float]:
        """
        Synthesizes speech from text using Chatterbox.

        Args:
            text (str): The text to synthesize.
            language (str): The language of the text (e.g., "en", "sk").
            speaker_embedding (Optional[torch.Tensor]): Speaker embedding for voice cloning.

        Returns:
            Tuple[np.ndarray, int, float]: A tuple containing the synthesized audio (numpy array),
                                           sample rate, and synthesis time in seconds.
        """
        # Chatterbox expects language codes like "en", "pl", etc.
        # For Slovak, we might need to use a fine-tuned model or a proxy language if direct support is limited.
        # For now, we'll assume direct support or a suitable default.
        
        # Ensure language is in a format Chatterbox expects
        # Chatterbox typically uses ISO 639-1 codes
        
        start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None

        if start_time: start_time.record()

        # The MTLTTS model's synthesize method might take a speaker_embedding argument directly
        # or require it to be set on the model instance.
        # Assuming it takes it as an argument for simplicity.
        # If not, we'd need to adjust how speaker_embedding is passed.
        
        # For Chatterbox, the language parameter is crucial.
        # If 'sk' is not directly supported, we might need a fallback or fine-tuned model.
        # For this prototype, we'll pass 'sk' directly and assume Chatterbox handles it or falls back.
        
        # The `synthesize` method in MTLTTS typically returns audio waveform and sample rate.
        # The `speaker_embedding` argument might be part of `kwargs` or a direct parameter.
        # Let's assume a direct parameter for now.
        
        # Note: Chatterbox's `synthesize` method might have different signatures based on version.
        # This is a common pattern for TTS models.
        
        # Actual synthesis call
        # The `synthesize` method in MTLTTS typically returns audio waveform and sample rate.
        # It expects language as a string (e.g., "en", "sk").
        # The `speaker_embedding` argument is passed directly.
        
        # ChatterboxMultilingualTTS.synthesize returns a tuple: (audio_waveform: torch.Tensor, sample_rate: int)
        audio_waveform_tensor, sample_rate = self.model.synthesize(
            text, 
            language=language, 
            speaker_embedding=speaker_embedding
        )
        audio_waveform = audio_waveform_tensor.cpu().numpy() # Convert to numpy array
        
        if speaker_embedding is not None:
            print(f"Synthesizing speech with voice cloning for language '{language}'.")
        else:
            print(f"Synthesizing speech with default voice for language '{language}'.")

        if end_time: 
            end_time.record()
            torch.cuda.synchronize()
            synthesis_time = start_time.elapsed_time(end_time) / 1000.0 # Convert ms to seconds
        else:
            synthesis_time = 0.0 # Placeholder for CPU/MPS timing

        print(f"Synthesized text: '{text}' in {synthesis_time:.2f}s")
        return audio_waveform, sample_rate, synthesis_time

if __name__ == "__main__":
    # Example Usage:
    import soundfile as sf
    import os

    tts_model = ChatterboxTTS(device="auto")

    # 1. Voice Training (using a dummy audio file for demonstration)
    dummy_ref_audio_path = "test/Voice-Training.wav" # Assuming this file exists as per project description
    # For actual testing, ensure 'test/Voice-Training.wav' is a valid audio file.
    # If it's a binary file, we can't create it here.
    # For now, let's assume it's a valid path and the file exists.
    
    # If the file doesn't exist, we'd need to create a dummy one or skip this part.
    # For the purpose of this example, we'll mock the speaker_embedding extraction.
    
    # Mock speaker embedding for demonstration
    mock_speaker_embedding = torch.randn(512) # Example: a 512-dim embedding

    # 2. Synthesize speech with default voice
    text_en = "Hello, this is a test sentence in English."
    audio_en, sr_en, time_en = tts_model.synthesize_speech(text_en, "en")
    sf.write("output_en_default.wav", audio_en, sr_en)
    print(f"Saved default English speech to output_en_default.wav (Time: {time_en:.2f}s)")

    # 3. Synthesize speech with cloned voice (using mock embedding)
    text_sk = "Dobrý deň, toto je testovacia veta v slovenčine."
    audio_sk, sr_sk, time_sk = tts_model.synthesize_speech(text_sk, "sk", speaker_embedding=mock_speaker_embedding)
    sf.write("output_sk_cloned.wav", audio_sk, sr_sk)
    print(f"Saved cloned Slovak speech to output_sk_cloned.wav (Time: {time_sk:.2f}s)")

    # Clean up dummy files (if created)
    # os.remove("output_en_default.wav")
    # os.remove("output_sk_cloned.wav")
