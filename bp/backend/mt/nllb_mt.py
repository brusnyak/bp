import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from typing import Tuple
import os


class NLLBMT:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
    ):
        """
        Initializes the NLLB Machine Translation model.

        Args:
            model_name (str): The Hugging Face model ID for NLLB.
            device (str): Device to run the model on ("cpu", "cuda", "mps", "auto").
                         "auto" will use MPS if available, then CUDA, then CPU.
        """
        # Determine device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                print("NLLBMT: MPS device detected and will be used.")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("NLLBMT: CUDA device detected and will be used.")
            else:
                self.device = "cpu"
                print("NLLBMT: No MPS/CUDA detected, using CPU.")
        else:
            self.device = device
            print(f"NLLBMT: Using specified device: {self.device}.")

        self.model_name = model_name
        
        # Language code mapping for NLLB (uses FLORES-200 codes)
        self.lang_code_map = {
            "en": "eng_Latn",
            "cs": "ces_Latn",
            "sk": "slk_Latn",
            # Add more mappings as needed
        }
        
        # Initialize model and tokenizer
        print(f"NLLBMT: Loading model '{self.model_name}' on device '{self.device}'...")
        start_time = time.perf_counter()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            
            # For better performance, we can use torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device != "mps":  # MPS doesn't support compile yet
                try:
                    self.model = torch.compile(self.model)
                    print("NLLBMT: Model compiled for faster inference")
                except Exception as e:
                    print(f"NLLBMT: Model compilation failed: {e}")
            
            load_time = time.perf_counter() - start_time
            print(
                f"NLLBMT: Model loaded successfully on {self.device} in {load_time:.2f}s."
            )
        except Exception as e:
            print(f"NLLBMT: Failed to load model: {e}")
            raise

    def _get_lang_code(self, lang: str) -> str:
        """Convert language code to NLLB format."""
        return self.lang_code_map.get(lang, lang)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, float]:
        """
        Translates text from source language to target language using NLLB.

        Args:
            text (str): The input text to translate.
            src_lang (str): Source language code (e.g., "en" for English).
            tgt_lang (str): Target language code (e.g., "sk" for Slovak).

        Returns:
            Tuple[str, float]: A tuple containing the translated text and the translation time in seconds.
        """
        # Get NLLB language codes
        src_code = self._get_lang_code(src_lang)
        tgt_code = self._get_lang_code(tgt_lang)
        
        start_time = time.perf_counter()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Force the target language token
            forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_code]
            
            # Generate translation
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=4,  # Beam search for better quality
                    early_stopping=True
                )
            
            # Decode the translation
            translated_text = self.tokenizer.batch_decode(
                translated_tokens, 
                skip_special_tokens=True
            )[0]
            
            end_time = time.perf_counter()
            translation_time = end_time - start_time
            
            print(
                f"Translated '{text}' ({src_lang}) to '{translated_text}' ({tgt_lang}) in {translation_time:.4f}s"
            )
            return translated_text, translation_time
            
        except Exception as e:
            print(f"Error during translation: {e}")
            raise


if __name__ == "__main__":
    # Simple test
    try:
        mt_model = NLLBMT(model_name="facebook/nllb-200-distilled-600M", device="auto")
        
        # English to Slovak
        text_en = "Hello, how are you today?"
        translated_sk, latency_sk = mt_model.translate(text_en, "en", "sk")
        print(f"EN -> SK Translation: {translated_sk}, Latency: {latency_sk:.4f}s")
        
        # Slovak to English
        text_sk = "Ahoj, ako sa máš?"
        translated_en, latency_en = mt_model.translate(text_sk, "sk", "en")
        print(f"SK -> EN Translation: {translated_en}, Latency: {latency_en:.4f}s")
        
    except Exception as e:
        print(f"Test failed: {e}")