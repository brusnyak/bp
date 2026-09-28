#!/usr/bin/env python3
"""Convert a Hugging Face Whisper checkpoint to a CTranslate2 int8 model usable by faster-whisper.

    python scripts/convert_whisper.py NaiveNeuron/whisper-small-sk ct2_models/whisper-small-sk

Needs the conversion stack (torch + transformers): run it inside the throwaway env from
scripts/convert_models.py or a venv built from requirements-convert.txt. The result is a plain
directory: point BP_SK_STT_MODEL at it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.mt.convert_opus_mt_to_ct2 import CompatTransformersConverter  # handles the transformers 4.x dtype kwarg
from transformers import AutoFeatureExtractor, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
CompatTransformersConverter(src).convert(out, quantization="int8", force=True)
# faster-whisper needs tokenizer.json + preprocessor_config.json next to model.bin; many fine-tunes only ship the
# slow-tokenizer files, so build them from the checkpoint itself (same vocabulary as the model was trained with).
AutoTokenizer.from_pretrained(src).save_pretrained(out)
AutoFeatureExtractor.from_pretrained(src).save_pretrained(out)
print("converted", src, "->", out)
