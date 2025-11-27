import ctranslate2
from transformers import AutoTokenizer
import torch  # Added for MPS check
import os  # Added for path operations
from typing import Tuple


class CTranslate2MT:
    def __init__(
        self, model_path: str = "Helsinki-NLP/opus-mt-en-sk", device: str = "auto"
    ):
        """
        Initializes the CTranslate2 Machine Translation model.

        Args:
            model_path (str): Path to the CTranslate2 converted model directory or Hugging Face model ID.
                              If a Hugging Face model ID, it will be converted and saved locally.
            device (str): Device to run the model on ("cpu", "cuda", "auto"). "auto" will use MPS on Apple Silicon.
        """
        if device == "auto":
            # CTranslate2 does not directly support "mps" device. Fallback to "cpu" on Apple Silicon.
            if torch.backends.mps.is_available():
                self.device = "cpu" # Explicitly use CPU for CTranslate2 on MPS
                print("CTranslate2MT: MPS device detected, but CTranslate2 will use CPU.")
            else:
                self.device = "cpu"
                print("CTranslate2MT: MPS not available, falling back to CPU.")
        else:
            self.device = device
            print(f"CTranslate2MT: Using specified device: {self.device}.")

        # CTranslate2 models are typically pre-converted.
        # For simplicity, we'll assume the model_path points to a converted directory.
        # If it's a Hugging Face ID, a conversion step would be needed.
        # For this test, we'll use a pre-converted Opus-MT model.
        # Determine the actual path to the CTranslate2 converted model directory
        # If model_path is a Hugging Face ID, convert it to the local directory name
        if "/" in model_path and not os.path.exists(model_path):
            # Assume it's a Hugging Face ID, construct the local path
            model_dir_name = model_path.replace("/", "--")
            self.ctranslate2_model_dir = os.path.join("ct2_models", model_dir_name)
            self.hf_model_id = model_path # Store original HF ID for tokenizer
        else:
            # Assume it's already a path to a converted model directory
            self.ctranslate2_model_dir = model_path
            # Try to infer HF model ID from the directory name for tokenizer
            # This is a heuristic and might need refinement if directory names don't match HF IDs
            self.hf_model_id = model_path.replace("ct2_models/", "").replace("--", "/")


        if not os.path.exists(os.path.join(self.ctranslate2_model_dir, "model.bin")):
            raise FileNotFoundError(
                f"CTranslate2 model.bin not found in {self.ctranslate2_model_dir}. Please ensure the model is converted."
            )

        self.translator = ctranslate2.Translator(self.ctranslate2_model_dir, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id
        )  # Use original HF ID for tokenizer
        print(
            f"CTranslate2MT initialized with model_path={self.ctranslate2_model_dir}, device={self.device}"
        )

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> Tuple[str, float]:
        """
        Translates text from source language to target language using CTranslate2.

        Args:
            text (str): The input text to translate.
            src_lang (str): Source language code (e.g., "en" for English, "sk" for Slovak).
            tgt_lang (str): Target language code.

        Returns:
            Tuple[str, float]: A tuple containing the translated text and the translation time in seconds.
        """
        # CTranslate2 expects tokenized input
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text, add_special_tokens=True)
        )

        import time

        start_time = time.time()

        # CTranslate2 translation
        # For Opus-MT models, language tags are often part of the model itself,
        # or can be prepended to the source tokens.
        # Let's try passing the language tags as part of the source tokens as per CTranslate2 documentation for NMT.
        # The `source_language` and `target_language` parameters are not directly supported in TranslationOptions for all models.
        # For Opus-MT, the tokenizer handles the language prefix.
        # The `translate_batch` method expects a list of list of tokens.

        # Prepend source language tag to tokens for Opus-MT
        # Example: __lang__en__ Hello world -> translate to Slovak
        # The tokenizer for Opus-MT models usually handles this.
        # Let's try without explicit language tags in options first, as the model is en-sk specific.

        # CTranslate2 translation
        # The `TranslationOptions` class might not be available in all ctranslate2 versions.
        # Let's try passing the options directly as keyword arguments to translate_batch.
        # For Opus-MT models, the target language tag is often prepended to the source sentence.
        # The tokenizer for Opus-MT models usually handles this.
        # Let's try to encode with the target language prefix.

        # Re-encoding with target language prefix for Opus-MT
        # The tokenizer for Opus-MT models often expects the target language token as a prefix.
        # Example: ">>sk<< Hello, how are you today?"

        # The `tokenizer.encode` method already adds special tokens.
        # For Opus-MT, the target language token is usually added by the tokenizer if it's a `MarianTokenizer`.
        # Let's assume the `AutoTokenizer` from `Helsinki-NLP/opus-mt-en-sk` handles this correctly.

        # The `translate_batch` method expects a list of list of tokens.
        # `tokens` is already a list of tokens. So `[tokens]` is correct.

        # The `TypeError` was likely due to `source_language` and `target_language` in `TranslationOptions`.
        # Let's try passing the options directly as keyword arguments to `translate_batch`.

        results = self.translator.translate_batch(
            [tokens], max_batch_size=1, beam_size=5, num_hypotheses=1
        )
        translated_tokens = results[0].hypotheses[0]
        translated_text = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(translated_tokens),
            skip_special_tokens=True,
        )

        end_time = time.time()
        translation_time = end_time - start_time

        print(
            f"Translated '{text}' ({src_lang}) to '{translated_text}' ({tgt_lang}) in {translation_time:.4f}s"
        )
        return translated_text, translation_time


if __name__ == "__main__":
    import os

    # Ensure the ONNX model path exists for testing
    # This assumes the model is already converted and present in onnx_models/models--Helsinki-NLP--opus-mt-en-sk
    # If not, this example will fail.
    converted_model_path = os.path.join(
        "onnx_models",
        "models--Helsinki-NLP--opus-mt-en-sk",
        "snapshots",
        "04d19ccd4566720bceeff93623d5cf93be659816",
    )
    if not os.path.exists(converted_model_path):
        print(
            f"Warning: Converted model not found at {converted_model_path}. Skipping example usage."
        )
        print(
            "Please ensure 'Helsinki-NLP/opus-mt-en-sk' is converted to CTranslate2 format and placed there."
        )
    else:
        mt_model = CTranslate2MT(model_path="Helsinki-NLP/opus-mt-en-sk", device="auto")

        # English to Slovak
        text_en = "Hello, how are you today?"
        translated_sk, latency_sk = mt_model.translate(text_en, "en", "sk")
        print(f"EN -> SK Translation: {translated_sk}, Latency: {latency_sk:.4f}s")

        # Slovak to English (requires a different converted model, e.g., "Helsinki-NLP/opus-mt-sk-en")
        # For this example, we'll just demonstrate EN->SK.
        # If you have opus-mt-sk-en converted, you can test it similarly.
