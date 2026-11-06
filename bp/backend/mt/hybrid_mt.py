import os
import torch
import logging
from typing import Dict, Tuple, Optional

from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.mt.nllb_mt import NLLBMT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Language pairs that should use NLLB (for better low-resource language support)
NLLB_PAIRS = {
    # Czech/Slovak pairs
    ("en", "cs"), ("en", "sk"), ("cs", "en"), ("sk", "en"),
    ("cs", "sk"), ("sk", "cs"),
    # Add more low-resource pairs as needed
}

# Language pairs that have good Opus-MT support
OPUS_PAIRS = {
    # High-resource pairs
    ("en", "de"), ("de", "en"),
    ("en", "fr"), ("fr", "en"),
    ("en", "es"), ("es", "en"),
    ("en", "it"), ("it", "en"),
    # Slovak/Czech pairs with good Opus-MT support
    ("en", "sk"), ("sk", "en"),
    ("en", "cs"), ("cs", "en"),
}


class HybridMT:
    """
    Hybrid Machine Translation system that intelligently routes between
    CTranslate2-optimized Opus-MT models and NLLB-200 based on language pair.
    """
    
    def __init__(self, device: str = "auto"):
        self.device = device
        if self.device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Storage for different MT models
        self.opus_models: Dict[str, CTranslate2MT] = {}
        self.nllb_model: Optional[NLLBMT] = None
        
        logging.info(f"HybridMT: Initializing on device: {self.device}")
        
    def _get_model_for_pair(self, source_lang: str, target_lang: str) -> str:
        """Determine which model to use for a language pair."""
        pair = (source_lang, target_lang)
        
        # Use NLLB for low-resource or Czech/Slovak pairs
        if pair in NLLB_PAIRS or source_lang in ["cs", "sk"] or target_lang in ["cs", "sk"]:
            # For now, prefer Opus-MT for cs/sk as it has good support
            # and NLLB distilled 600M might not be as fast
            if pair in OPUS_PAIRS:
                return "opus"
            return "nllb"
        
        # Use Opus-MT for high-resource pairs
        return "opus"
    
    def _ensure_opus_model(self, source_lang: str, target_lang: str) -> CTranslate2MT:
        """Get or create an Opus-MT model for the given language pair."""
        key = f"{source_lang}-{target_lang}"
        
        if key not in self.opus_models:
            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
            logging.info(f"HybridMT: Loading Opus-MT model for {key}...")
            try:
                self.opus_models[key] = CTranslate2MT(
                    model_path=model_name, 
                    device=self.device
                )
            except Exception as e:
                logging.warning(f"HybridMT: Failed to load Opus-MT {key}: {e}")
                raise
        
        return self.opus_models[key]
    
    def _ensure_nllb_model(self) -> NLLBMT:
        """Get or create the NLLB model."""
        if self.nllb_model is None:
            logging.info(f"HybridMT: Loading NLLB-200 distilled model...")
            try:
                self.nllb_model = NLLBMT(
                    model_name="facebook/nllb-200-distilled-600M",
                    device=self.device
                )
            except Exception as e:
                logging.error(f"HybridMT: Failed to load NLLB model: {e}")
                raise
        
        return self.nllb_model
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, float]:
        """
        Translate text using the optimal MT model for the language pair.
        
        Args:
            text: Input text
            source_lang: Source language code (e.g., "en")
            target_lang: Target language code (e.g., "sk")
            
        Returns:
            Tuple of (translated_text, translation_time)
        """
        model_type = self._get_model_for_pair(source_lang, target_lang)
        
        if model_type == "nllb":
            model = self._ensure_nllb_model()
            logging.info(f"HybridMT: Using NLLB for {source_lang}->{target_lang}")
            return model.translate(text, source_lang, target_lang)
        else:
            model = self._ensure_opus_model(source_lang, target_lang)
            logging.info(f"HybridMT: Using Opus-MT for {source_lang}->{target_lang}")
            return model.translate(text, source_lang, target_lang)


if __name__ == "__main__":
    # Test the hybrid system
    engine = HybridMT(device="cpu")
    
    test_cases = [
        ("Hello world", "en", "sk"),
        ("Ahoj svet", "sk", "en"),
        ("Hello", "en", "de"),
    ]
    
    for text, src, tgt in test_cases:
        print(f"\nTranslating: '{text}' ({src} -> {tgt})")
        result, latency = engine.translate(text, src, tgt)
        print(f"Result: '{result}' ({latency:.4f}s)")