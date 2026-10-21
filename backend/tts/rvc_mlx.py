"""
RVC-MLX voice conversion wrapper — engine-agnostic voice layer.

Train ONE RVC model (Applio, ~1 hr on rented GPU), drop .pth + .index in models/rvc/,
then ANY TTS engine's output → your voice. Solves "per-engine training matrix" permanently.

Requirements (Mac):     pip install rvc-mlx
Requirements (Linux):   pip install onyx[cpu] (or onyx[cuda] for GPU)

MIT licensed: RVC-MLX (Acelogic) + onyx (hixady)
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RVCConfig:
    model_dir: str = "models/rvc"          # contains .pth + .index
    f0_method: str = "rmvpe"               # pitch extraction: rmvpe | harvest | pm
    index_rate: float = 0.5                # feature retrieval blend (0-1)
    filter_radius: int = 3                 # median filter for pitch
    rms_mix_rate: float = 0.25             # loudness envelope blend
    protect: float = 0.33                  # voiceless protection
    device: str = "auto"                   # auto | cpu | cuda | mps


class RVCMLX:
    """RVC-MLX voice conversion for Apple Silicon (MLX backend)."""

    SUPPORTS_CLONING = True
    REQUIRES_SPEAKER_WAV = False  # uses pre-trained .pth, not reference wav
    SUPPORTS_STREAMING = False    # file-based for now
    sample_rate = 44100           # RVC-MLX outputs 44.1kHz

    def __init__(self, config: Optional[RVCConfig] = None):
        self.config = config or RVCConfig()
        self._model = None
        self._model_path = None
        self._index_path = None

    def _ensure_model(self) -> tuple[str, str]:
        """Find and validate .pth + .index files."""
        model_dir = Path(self.config.model_dir)
        pth_files = list(model_dir.glob("*.pth"))
        idx_files = list(model_dir.glob("*.index"))

        if not pth_files:
            raise FileNotFoundError(f"No .pth model found in {model_dir}")
        if not idx_files:
            raise FileNotFoundError(f"No .index file found in {model_dir}")

        # Prefer matching names
        model_path = str(pth_files[0])
        index_path = str(idx_files[0])
        for pth in pth_files:
            stem = pth.stem
            for idx in idx_files:
                if idx.stem.startswith(stem):
                    model_path = str(pth)
                    index_path = str(idx)
                    break

        return model_path, index_path

    def load(self) -> None:
        """Lazy-load the RVC model."""
        if self._model is not None:
            return

        model_path, index_path = self._ensure_model()

        # Import here so module loads even without rvc-mlx installed
        try:
            from rvc_mlx import RVC as MLX_RVC
        except ImportError:
            raise RuntimeError("rvc-mlx not installed. Run: pip install rvc-mlx")

        self._model = MLX_RVC(
            model_path=model_path,
            index_path=index_path,
            device=self.config.device,
        )
        self._model_path = model_path
        self._index_path = index_path

    def synthesize(self, text: str, language: str = "en", speaker_wav_path: Optional[str] = None) -> tuple[np.ndarray, int, float]:
        """Text → TTS → RVC → your voice.
        
        Note: This requires a TTS engine to generate the source audio first.
        In practice, chain: TTS.synthesize() → RVCMLX.convert()
        """
        raise NotImplementedError("Use convert() with pre-generated audio. Chain: TTS → RVC.")

    def convert(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Convert arbitrary audio through RVC voice model.
        
        Returns: (converted_audio, sample_rate) at 44.1kHz
        """
        import time
        self.load()
        t0 = time.perf_counter()

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # RVC-MLX expects float32 [-1, 1]
        audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        # Convert
        out = self._model.convert(
            audio=audio,
            sr=sr,
            f0_method=self.config.f0_method,
            index_rate=self.config.index_rate,
            filter_radius=self.config.filter_radius,
            rms_mix_rate=self.config.rms_mix_rate,
            protect=self.config.protect,
        )

        latency = time.perf_counter() - t0
        return out, 44100, latency


class RVCOnnx:
    """onyx RVC voice conversion for Linux/Windows (ONNX Runtime backend)."""

    SUPPORTS_CLONING = True
    REQUIRES_SPEAKER_WAV = False
    SUPPORTS_STREAMING = False
    sample_rate = 44100

    def __init__(self, config: Optional[RVCConfig] = None):
        self.config = config or RVCConfig()
        self._session = None
        self._model_path = None

    def _ensure_model(self) -> str:
        """Find .onnx model (onyx packs .pth + .index into single .onyx)."""
        model_dir = Path(self.config.model_dir)
        onyx_files = list(model_dir.glob("*.onyx"))
        if not onyx_files:
            # Try to pack from .pth + .index
            pth_files = list(model_dir.glob("*.pth"))
            idx_files = list(model_dir.glob("*.index"))
            if pth_files and idx_files:
                return self._pack_model(str(pth_files[0]), str(idx_files[0]))
            raise FileNotFoundError(f"No .onyx model in {model_dir}. Pack from .pth + .index first.")
        return str(onyx_files[0])

    def _pack_model(self, pth_path: str, index_path: str) -> str:
        """Pack .pth + .index into .onyx using onyx CLI."""
        out_path = pth_path.replace(".pth", ".onyx")
        cmd = ["onyx", "pack", "-o", out_path, "--type", "rvc", "--model", pth_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"onyx pack failed: {result.stderr}")
        return out_path

    def load(self) -> None:
        if self._session is not None:
            return

        model_path = self._ensure_model()

        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime-gpu (or onnxruntime)")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.config.device == "cpu":
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(model_path, providers=providers)
        self._model_path = model_path

    def convert(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Convert audio through onnx RVC model."""
        import time
        self.load()
        t0 = time.perf_counter()

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Resample to 44.1k if needed (onyx expects 44.1k)
        if sr != 44100:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)

        # onnx input format varies by model - this is a simplified version
        # Real implementation needs f0 extraction (RMVPE) + content encoder (ContentVec)
        # For production, use the onyx Python API directly
        raise NotImplementedError("Use onyx Python API for full pipeline. This is a stub.")

    def synthesize(self, text: str, language: str = "en", speaker_wav_path: Optional[str] = None) -> tuple[np.ndarray, int, float]:
        raise NotImplementedError("Use convert() with pre-generated audio.")


def get_rvc_engine(config: Optional[RVCConfig] = None):
    """Factory: returns appropriate RVC engine for current platform."""
    import platform
    system = platform.system().lower()

    if system == "darwin":
        try:
            import rvc_mlx
            return RVCMLX(config)
        except ImportError:
            pass
    
    # Fallback to onyx (cross-platform)
    try:
        import onnxruntime
        return RVCOnnx(config)
    except ImportError:
        pass

    raise RuntimeError("No RVC backend available. Install rvc-mlx (Mac) or onyx (Linux/Windows).")


# Registry entry point for backend/registry.py
def rvc_mlx_factory(**kwargs) -> RVCMLX:
    config = RVCConfig(**kwargs)
    return RVCMLX(config)


def rvc_onnx_factory(**kwargs) -> RVCOnnx:
    config = RVCConfig(**kwargs)
    return RVCOnnx(config)