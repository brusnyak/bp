import os
import torch
import time
import numpy as np
import soundfile as sf
import tempfile
import io
import logging
from typing import Tuple, Optional, Dict

from backend.tts.piper_tts import PiperTTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridTTS:
    """
    Hybrid TTS engine for BP2.
    Uses Piper for fast Slovak synthesis and OpenVoice V2 for zero-shot voice cloning.
    """
    def __init__(self, device: str = "auto"):
        self.device = device
        if self.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        logging.info(f"HybridTTS: Initializing on device: {self.device}")
        
        # 1. Initialize Piper
        self.piper_engine = PiperTTS(device=self.device)
        self.sample_rate = 16000 # Standard for conversion pass, but Piper is usually 22050
        
        # 2. Initialize OpenVoice V2 Converter
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ckpt_converter = os.path.join(base_dir, 'models', 'openvoice_v2', 'checkpoints_v2', 'converter')
        
        if not os.path.exists(ckpt_converter):
            logging.error(f"HybridTTS: OpenVoice V2 checkpoints not found at {ckpt_converter}")
            raise FileNotFoundError(f"OpenVoice V2 checkpoints not found at {ckpt_converter}")
            
        self.tone_color_converter = ToneColorConverter(
            os.path.join(ckpt_converter, 'config.json'), 
            device=self.device
        )
        self.tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))
        
        # 3. Embedding Caches
        self.target_se_cache: Dict[str, torch.Tensor] = {}
        
    def _get_target_se(self, speaker_wav_path: str) -> torch.Tensor:
        """Cache and retrieve target speaker embedding."""
        if speaker_wav_path in self.target_se_cache:
            return self.target_se_cache[speaker_wav_path]
            
        logging.info(f"HybridTTS: Extracting target SE for {speaker_wav_path}")
        target_se, _ = se_extractor.get_se(speaker_wav_path, self.tone_color_converter, vad=True)
        self.target_se_cache[speaker_wav_path] = target_se
        return target_se

    def synthesize(
        self, 
        text: str, 
        language: str = "sk", 
        speaker_wav_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Synthesize Slovak speech and clone the voice if speaker_wav_path is provided.
        """
        start_time = time.perf_counter()
        
        # 1. Piper Synthesis (Slovak Proxy)
        # Piper returns (waveform, sr, syn_time)
        piper_wav, piper_sr, _ = self.piper_engine.synthesize(text, language=language)
        
        if speaker_wav_path is None:
            # No cloning requested, return Piper output
            total_time = time.perf_counter() - start_time
            return piper_wav, piper_sr, total_time
            
        # 2. OpenVoice Conversion Pass
        logging.info(f"HybridTTS: performing voice cloning to {speaker_wav_path}")
        
        # We need to save piper_wav to a temporary file because OpenVoice converter expects a path
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_piper:
            tmp_piper_path = tmp_piper.name
            sf.write(tmp_piper_path, piper_wav, piper_sr)
            
        try:
            # Extract source SE (from Piper output)
            source_se, _ = se_extractor.get_se(tmp_piper_path, self.tone_color_converter, vad=True)
            
            # Get target SE
            target_se = self._get_target_se(speaker_wav_path)
            
            # Convert
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
                tmp_output_path = tmp_output.name
                
            self.tone_color_converter.convert(
                audio_src_path=tmp_piper_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=tmp_output_path,
                message="@MyShell"
            )
            
            # Load back cloned audio
            cloned_wav, final_sr = sf.read(tmp_output_path)
            
            # Cleanup
            if os.path.exists(tmp_piper_path): os.remove(tmp_piper_path)
            if os.path.exists(tmp_output_path): os.remove(tmp_output_path)
            
            total_time = time.perf_counter() - start_time
            logging.info(f"HybridTTS: End-to-end synthesis + cloning in {total_time:.2f}s")
            
            return np.array(cloned_wav), final_sr, total_time
            
        except Exception as e:
            logging.error(f"HybridTTS: Error during conversion: {e}")
            # Fallback to Piper original
            if os.path.exists(tmp_piper_path): os.remove(tmp_piper_path)
            return piper_wav, piper_sr, time.perf_counter() - start_time

if __name__ == "__main__":
    # Test script
    engine = HybridTTS()
    ref_wav = "/Users/yegor/Documents/STU/BP/test/My test speech_xtts_speaker_clean.wav"
    text = "Vítam vás pri praktickej ukážke hybridného systému syntézy reči."
    
    wav, sr, lat = engine.synthesize(text, speaker_wav_path=ref_wav)
    sf.write("hybrid_test_output.wav", wav, sr)
    print(f"Hybrid synthesis done in {lat:.2f}s")
