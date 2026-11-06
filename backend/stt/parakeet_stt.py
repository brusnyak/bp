"""Parakeet TDT v3 STT engine (NVIDIA, CC-BY-4.0).

Optional GPU-first engine. Requires `nemo_toolkit` or `transformers` + `torch`.
Declared-only in registry until installed and validated on Kaggle T4.

Fallback order:
  1. NVIDIA NeMo (nemo.collections.asr.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3"))
  2. Hugging Face transformers (AutoModelForSpeechSeq2Seq + AutoProcessor)
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np


class ParakeetSTT:
    """Parakeet TDT 0.6B v3 speech recognition."""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: Optional[str] = None,
    ):
        self.model_name = model_name
        # Ensure nvidia/ prefix for HF model lookup
        if "/" not in self.model_name:
            self.model_name = f"nvidia/{self.model_name}"
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._engine = None  # "nemo" | "transformers" | None
        self._model = None
        self._processor = None
        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        # Try NeMo first
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore

            self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._engine = "nemo"
            print(f"ParakeetSTT loaded via NeMo: {self.model_name} on {self.device}")
            return
        except Exception as nemo_exc:
            print(f"ParakeetSTT: NeMo unavailable ({nemo_exc}), trying transformers...")

        # Fallback to transformers
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, ParakeetForTDT  # type: ignore

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = ParakeetForTDT.from_pretrained(
                self.model_name, torch_dtype=getattr(torch, self.compute_type, torch.float16)
            ).to(self.device)
            self._engine = "transformers"
            self._torch = torch
            print(f"ParakeetSTT loaded via transformers: {self.model_name} on {self.device}")
            return
        except Exception as tf_exc:
            print(f"ParakeetSTT: transformers unavailable ({tf_exc})")

        raise RuntimeError(
            "ParakeetSTT requires either 'nemo_toolkit' or 'transformers'+'torch'. "
            "Install one: pip install nemo_toolkit['all']  or  pip install transformers torch torchaudio"
        )

    # ------------------------------------------------------------------ transcribe

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        vad_filter: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Tuple[list, float, Optional[str]]:
        """
        Transcribe audio. Returns (segments, transcription_time, detected_language).

        Segments are list-like objects with .text and (optionally) .start/.end/.words,
        mirroring faster-whisper's segment interface so the pipeline can reuse them.
        """
        lang = language or self.language or "en"
        start = time.perf_counter()

        if self._engine == "nemo":
            # NeMo expects a file path; write temp WAV
            import soundfile as sf
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_data, sample_rate)
            try:
                out = self._model.transcribe([tmp_path])[0]
                text = out.text if hasattr(out, "text") else str(out)
            finally:
                os.unlink(tmp_path)
            segments = [_Segment(text)]
            detected = lang

        elif self._engine == "transformers":
            import torch  # type: ignore

            inputs = self._processor(
                audio_data, sampling_rate=sample_rate, return_tensors="pt"
            ).to(self.device)
            # Match input dtype to model dtype (float16 model needs float16 input)
            inputs["input_features"] = inputs["input_features"].to(self._model.dtype)
            with torch.no_grad():
                generated = self._model.generate(inputs["input_features"])
            generated_ids = generated.sequences if hasattr(generated, "sequences") else generated
            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            segments = [_Segment(text)]
            detected = lang

        else:
            raise RuntimeError("ParakeetSTT: no engine loaded")

        dt = time.perf_counter() - start
        return segments, dt, detected


class _Segment:
    """Whisper-compatible segment wrapper."""

    def __init__(self, text: str):
        self.text = text
        self.start = 0.0
        self.end = 0.0
        self.words = None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        import soundfile as sf

        audio, sr = sf.read(sys.argv[1])
        stt = ParakeetSTT()
        segs, dt, lang = stt.transcribe_audio(audio, sr)
        print(f"Lang={lang} Time={dt:.3f}s Text={segs[0].text[:200]}")
