import pytest
import sys
import os
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.tts.coqui_tts import CoquiTTS

class TestCoquiTTSIntegration:
    
    @pytest.fixture
    def mock_tts(self):
        with patch('backend.tts.coqui_tts.TTS') as mock:
            yield mock

    @pytest.fixture
    def mock_torch_cuda(self):
        with patch('torch.cuda.is_available') as mock:
            yield mock

    @pytest.fixture
    def mock_torch_mps(self):
        with patch('torch.backends.mps.is_available') as mock:
            yield mock

    def test_device_detection_cuda(self, mock_tts, mock_torch_cuda, mock_torch_mps):
        mock_torch_cuda.return_value = True
        mock_torch_mps.return_value = False
        
        tts = CoquiTTS()
        assert tts.device == "cuda"

    def test_device_detection_mps(self, mock_tts, mock_torch_cuda, mock_torch_mps):
        mock_torch_cuda.return_value = False
        mock_torch_mps.return_value = True
        
        tts = CoquiTTS()
        # Note: The current implementation forces CPU even if MPS is available due to stability issues
        # Verify that it falls back to CPU or respects the warning logic
        # In coqui_tts.py: if self.device == "mps": self.device = "cpu"
        assert tts.device == "cpu" 

    def test_device_detection_cpu(self, mock_tts, mock_torch_cuda, mock_torch_mps):
        mock_torch_cuda.return_value = False
        mock_torch_mps.return_value = False
        
        tts = CoquiTTS()
        assert tts.device == "cpu"

    def test_speaker_embedding_caching(self, mock_tts):
        # Setup mock model
        # self.model is initialized as TTS(...).to(...)
        # So we need to configure the return value of .to()
        mock_model_instance = mock_tts.return_value.to.return_value
        mock_model_instance.synthesizer.tts_model.get_conditioning_latents.return_value = ("latent", "embedding")
        
        tts = CoquiTTS(device="cpu", enable_warmup=False)
        
        # Create dummy wav
        dummy_wav = "dummy_speaker.wav"
        with open(dummy_wav, "w") as f:
            f.write("dummy content")
            
        try:
            # First call
            with patch('os.path.exists', return_value=True):
                tts.compute_speaker_embedding(dummy_wav)
                assert dummy_wav in tts.speaker_embedding_cache
                
                # Reset mock to verify it's not called again
                mock_model_instance.synthesizer.tts_model.get_conditioning_latents.reset_mock()
                
                # Second call
                tts.compute_speaker_embedding(dummy_wav)
                mock_model_instance.synthesizer.tts_model.get_conditioning_latents.assert_not_called()
        finally:
            if os.path.exists(dummy_wav):
                os.remove(dummy_wav)

    def test_streaming_synthesis_generator(self, mock_tts):
        mock_model_instance = mock_tts.return_value.to.return_value
        # Mock inference to return a dict with 'wav'
        mock_model_instance.synthesizer.tts_model.inference.return_value = {'wav': np.array([0.1, 0.2])}
        # Mock get_conditioning_latents
        mock_model_instance.synthesizer.tts_model.get_conditioning_latents.return_value = ("latent", "embedding")
        
        tts = CoquiTTS(device="cpu", enable_warmup=False)
        
        # Use a long text to trigger streaming
        long_text = "This is a long text. " * 20 
        dummy_wav = "dummy.wav"
        
        with patch('os.path.exists', return_value=True):
            generator = tts.synthesize_stream(long_text, "en", dummy_wav)
            
            # Consume generator
            chunks = list(generator)
            assert len(chunks) > 0
            assert isinstance(chunks[0], np.ndarray)

if __name__ == "__main__":
    pytest.main([__file__])
