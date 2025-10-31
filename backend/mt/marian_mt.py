from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
from typing import Tuple

class MarianMT:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-sk", device: str = "auto"):
        """
        Initializes the MarianMT Machine Translation model.

        Args:
            model_name (str): The name of the MarianMT model to use from Hugging Face.
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

        print(f"INFO: Loading MarianMT model: {model_name} on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print(f"MarianMT initialized with model_name={model_name}, device={self.device}")

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, float]:
        """
        Translates text from source_lang to target_lang using MarianMT.

        Args:
            text (str): The text to translate.
            source_lang (str): The source language code (e.g., "en", "sk").
            target_lang (str): The target language code (e.g., "en", "sk").

        Returns:
            Tuple[str, float]: A tuple containing the translated text and the translation time in seconds.
        """
        # MarianMT models typically handle language codes directly as part of their model name
        # e.g., "opus-mt-en-sk" implies English source and Slovak target.
        # The tokenizer will automatically handle the source language.

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        start_time = time.perf_counter()
        translated_tokens = self.model.generate(**inputs)
        end_time = time.perf_counter()
        translation_time = end_time - start_time

        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        print(f"Translated: '{text}' ({source_lang}) to '{translated_text}' ({target_lang}) in {translation_time:.4f}s")
        return translated_text, translation_time

if __name__ == "__main__":
    # Example Usage:
    mt_model = MarianMT(model_name="Helsinki-NLP/opus-mt-en-sk", device="auto")

    # English to Slovak
    text_en = "Hello, how are you doing today?"
    translated_text_sk, time_sk = mt_model.translate_text(text_en, "en", "sk")
    print(f"EN to SK Translation: {translated_text_sk} (Time: {time_sk:.4f}s)")

    # Slovak to English (Note: This model is specifically en-sk, so sk-en would require a different model)
    # For sk-en, you would need a model like "Helsinki-NLP/opus-mt-sk-en"
    # text_sk = "Dobrý deň, ako sa dnes máte?"
    # translated_text_en, time_en = mt_model.translate_text(text_sk, "sk", "en")
    # print(f"SK to EN Translation: {translated_text_en} (Time: {time_en:.4f}s)")
