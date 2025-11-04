import os
import ctranslate2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
    converter = ctranslate2.converters.TransformersConverter(model_name)

    # Convert and save the model, forcing overwrite if directory exists
    converter.convert(output_dir, quantization=quantization, force=True)

    print(f"Model {model_name} successfully converted and saved to {output_dir}")

if __name__ == "__main__":
    # Define models to convert
    models_to_convert = [
        "Helsinki-NLP/opus-mt-en-sk",
        "Helsinki-NLP/opus-mt-sk-en"
    ]

    base_output_dir = "ct2_models"
    os.makedirs(base_output_dir, exist_ok=True)

    for model_name in models_to_convert:
        # Create a specific output directory for each model
        model_output_dir = os.path.join(base_output_dir, model_name.replace("/", "--"))
        convert_model(model_name, model_output_dir, quantization="int8")

    print("\nAll specified models attempted conversion to CTranslate2.")
