from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
from typing import Tuple

class NLLB_MT:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "auto", compute_type: str = "float32"):
        """
        Initializes the NLLB Machine Translation model.

        Args:
            model_name (str): The name of the NLLB model to use from Hugging Face.
            device (str): Device to run the model on ("cpu", "cuda", "mps", or "auto").
            compute_type (str): Compute type for the model (e.g., "float32", "int8").
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

        print(f"INFO: Loading NLLB_MT model: {model_name} on device: {self.device} with compute_type: {compute_type}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        # TODO: Implement compute_type handling if needed for NLLB
        print(f"NLLB_MT initialized with model_name={model_name}, device={self.device}")

    # A simple mapping for common language codes to NLLB's specific codes
    # This should be expanded for full NLLB support
    LANG_CODE_MAP = {
        "en": "eng_Latn",
        "sk": "slk_Latn",
        "cs": "ces_Latn", # Adding Czech as an example
        # Add more mappings as needed for NLLB
    }

    def _map_lang_code(self, lang: str) -> str:
        return self.LANG_CODE_MAP.get(lang, lang) # Return original if not found

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, float]:
        """
        Translates text from source_lang to target_lang using NLLB.

        Args:
            text (str): The text to translate.
            source_lang (str): The source language code (e.g., "en", "sk").
            target_lang (str): The target language code (e.g., "en", "sk").

        Returns:
            Tuple[str, float]: A tuple containing the translated text and the translation time in seconds.
        """
        nllb_source_lang = self._map_lang_code(source_lang)
        nllb_target_lang = self._map_lang_code(target_lang)

        self.tokenizer.src_lang = nllb_source_lang # Set source language for tokenizer
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        start_time = time.perf_counter()
        # NLLB models use the target language token as the forced_bos_token.
        # The correct way to get the token ID for NLLB is to convert the special language token.
        translated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(f"__{nllb_target_lang}__"))
        end_time = time.perf_counter()
        translation_time = end_time - start_time

        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        print(f"Translated: '{text}' ({source_lang}) to '{translated_text}' ({target_lang}) in {translation_time:.4f}s")
        return translated_text, translation_time

if __name__ == "__main__":
    # Example Usage:
    # Note: NLLB requires specific language codes. For this example, we'll use placeholder codes.
    # A real implementation would need a mapping from common codes (e.g., "en", "sk") to NLLB codes (e.g., "eng_Latn", "slk_Latn").
    
    # For demonstration, let's assume a simple mapping or direct use if the model supports it.
    # You would typically need to load a specific NLLB model that supports the desired language pair.
    # For example, "facebook/nllb-200-distilled-600M" supports many languages.

    nllb_model = NLLB_MT(model_name="facebook/nllb-200-distilled-600M", device="auto")

    # English to Slovak
    text_en = "Hello, how are you doing today?"
    translated_text_sk, time_sk = nllb_model.translate_text(text_en, "en", "sk")
    print(f"EN to SK Translation: {translated_text_sk} (Time: {time_sk:.4f}s)")
