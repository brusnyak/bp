import os
import ctranslate2
from ctranslate2.converters import TransformersConverter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class CompatTransformersConverter(TransformersConverter):
    """SANDBOX-NOTE (2026-09-28): ctranslate2>=4.6 passes dtype= to from_pretrained,
    but this project pins transformers==4.46.3, which only understands torch_dtype
    (the rename happened in transformers 4.47). The unknown kwarg falls through into
    MarianMTModel.__init__ -> TypeError. Translate it back; conversion output is
    identical, runtime pins untouched."""

    def load_model(self, model_class, model_name_or_path, **kwargs):
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return super().load_model(model_class, model_name_or_path, **kwargs)

def convert_model(model_name: str, output_dir: str, quantization: str = "int8"):
    """
    Converts a Hugging Face Transformers model to CTranslate2 format.

    Args:
        model_name (str): The Hugging Face model ID (e.g., "Helsinki-NLP/opus-mt-en-sk").
        output_dir (str): The directory where the converted CTranslate2 model will be saved.
        quantization (str): Quantization type (e.g., "int8", "float16", "float32").
    """
    print(f"Converting model: {model_name} to CTranslate2 format with {quantization} quantization...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a converter, letting it load the model and tokenizer internally
    converter = CompatTransformersConverter(model_name)

    # Convert and save the model, forcing overwrite if directory exists
    converter.convert(output_dir, quantization=quantization, force=True)

    # Keep the tokenizer next to model.bin so runtime (ctranslate2_mt.py) never needs the network.
    AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir)

    print(f"Model {model_name} successfully converted and saved to {output_dir}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face Transformers model to CTranslate2 format.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="The Hugging Face model ID (e.g., 'Helsinki-NLP/opus-mt-en-sk').")
    parser.add_argument("--quantization", type=str, default="int8",
                        help="Quantization type (e.g., 'int8', 'float16', 'float32'). Default is 'int8'.")

    args = parser.parse_args()

    base_output_dir = "ct2_models"
    os.makedirs(base_output_dir, exist_ok=True)

    model_output_dir = os.path.join(base_output_dir, args.model_name.replace("/", "--"))
    convert_model(args.model_name, model_output_dir, quantization=args.quantization)

    print(f"\nConversion attempt for {args.model_name} completed.")
