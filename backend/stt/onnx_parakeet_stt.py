"""Parakeet TDT v3 (0.6B, CC-BY-4.0) through onnx-asr: the low-latency streaming engine. Same interface as FasterWhisperSTT.

Why: Whisper pads every input to a fixed 30 s window, so one call costs a flat ~720 ms (large-v3-turbo) or ~170 ms (small) on the
M1 Pro whatever the clip length (measured 2026-11-07: 0.5 s window 787 ms, 3 s window 722 ms). Parakeet's cost scales with the audio:
81 ms for 0.5 s, 96 ms for 1 s, 147 ms for 2 s, 224 ms for 3 s (CPU). That makes 0.3 s decoding steps affordable, which the
incremental route needs to commit words without waiting for a 2 s chunk.
Trade-offs measured: accuracy is below turbo for cs/pl/de-on-T4 and equal for fr/sk; the language is auto-detected (the `language`
argument is ignored; on the owner's non-native Slovak it picked the wrong language, WER 80.7%).
Env: BP_PARAKEET_PROVIDERS (comma list, default CUDA if present else CPU), BP_PARAKEET_ONNX_DIR (model dir), BP_PARAKEET_INT8 (default 1 on CPU).
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np

REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
FRAME_S = 0.08   # TDT encoder frame (token timestamps are multiples of this)


class OnnxParakeetSTT:
    def __init__(self, model_size: str = "", device: str = "auto", compute_type: str = "", beam_size: int = 5):
        import onnx_asr
        import onnxruntime as ort
        from huggingface_hub import snapshot_download

        env = os.environ.get("BP_PARAKEET_PROVIDERS")
        avail = ort.get_available_providers()
        providers = [p.strip() for p in env.split(",")] if env else (["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.environ.get("BP_PARAKEET_ONNX_DIR", os.path.join(root, "onnx_models", "parakeet-tdt-0.6b-v3"))
        # real files, not the HF cache symlinks: onnxruntime rejects external-data paths that escape the model directory
        int8 = os.environ.get("BP_PARAKEET_INT8", "0" if providers[0].startswith("CUDA") else "1") == "1"
        # Only the variant in use: int8 is ~700 MB, fp32 ~2.5 GB. First start downloads it (~1 min), which blocks that
        # session's connection; pre-fetch with `python -c "from backend.stt.onnx_parakeet_stt import OnnxParakeetSTT; OnnxParakeetSTT()"`.
        variant = ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"] if int8 else ["encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"]
        snapshot_download(REPO, local_dir=path, allow_patterns=variant + ["nemo128.onnx", "vocab.txt", "config.json"])
        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path, quantization="int8" if int8 else None, providers=providers).with_timestamps()
        self.model_size = f"parakeet-tdt-0.6b-v3{'-int8' if int8 else ''}"
        self.model.recognize(np.zeros(16000, dtype=np.float32))     # warm-up: graph optimisation + allocator
        print(f"OnnxParakeetSTT initialized: {self.model_size} providers={providers}")

    def transcribe_audio(
        self, audio_data: np.ndarray, sample_rate: int, language: Optional[str] = None, vad_filter: bool = True,
        initial_prompt: Optional[str] = None, word_timestamps: bool = False,
    ) -> Tuple[list, float, Optional[str]]:
        audio = np.ascontiguousarray(audio_data, dtype=np.float32)
        if sample_rate != 16000:
            import scipy.signal as ss
            audio = ss.resample_poly(audio, 16000, sample_rate).astype(np.float32)
        if len(audio) < 1600:                       # < 0.1 s: nothing to recognise
            return [], 0.0, language
        t0 = time.time()
        res = self.model.recognize(audio)
        dt = time.time() - t0
        words = self._words(res.tokens, res.timestamps, len(audio) / 16000)
        text = (res.text or "").strip()
        if not text:
            return [], dt, language
        seg = SimpleNamespace(start=words[0].start if words else 0.0, end=words[-1].end if words else len(audio) / 16000,
                              text=" " + text, words=words or None, no_speech_prob=0.0, avg_logprob=0.0)
        return [seg], dt, language

    @staticmethod
    def _words(tokens, stamps, total_s):
        """Merge sub-word tokens into words: a token starting with a space (or the first) opens a word."""
        words, cur = [], None
        for tok, ts in zip(tokens, stamps):
            if cur is None or tok.startswith(" "):
                if cur is not None:
                    cur.end = max(cur.end, min(ts, cur.end + FRAME_S))
                    words.append(cur)
                cur = SimpleNamespace(word=tok.strip(), start=float(ts), end=float(ts) + FRAME_S, probability=1.0)
            else:
                cur.word += tok
                cur.end = float(ts) + FRAME_S
        if cur is not None:
            words.append(cur)
        for w in words:
            w.end = min(w.end, total_s)
        return [w for w in words if w.word]
