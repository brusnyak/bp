"""Hardware auto-configuration for BP v2.

Scans the OS, CPU, RAM, and GPU (CUDA / MPS / none) at startup and selects the
optimal STT model size, compute type, MT device, and TTS engine for each stage.

Designed so the same code runs unchanged on:
  - Mac M1/M2/M3 (MPS, ARM)
  - Windows CPU-only (no GPU)
  - NVIDIA laptop (CUDA)
  - Linux server (CUDA or CPU)

Output is written to backend/config/auto_config.json and exposed via /api/hardware.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class HardwareProfile:
    """Detected hardware characteristics."""
    os: str
    arch: str
    cpu_cores: int
    ram_gb: float
    gpu: str                 # "cuda" | "mps" | "cpu" | "none"
    gpu_name: Optional[str]
    vram_gb: Optional[float]
    has_ctranslate2: bool
    has_faster_whisper: bool
    has_piper: bool
    has_torch: bool


@dataclass
class StageConfig:
    """Selected configuration for one pipeline stage."""
    device: str
    compute_type: str
    model_size: str
    engine: str
    notes: str


@dataclass
class AutoConfig:
    """Complete auto-selected pipeline configuration."""
    hardware: HardwareProfile
    stt: StageConfig
    mt: StageConfig
    tts: StageConfig
    vad: StageConfig
    streaming: bool
    max_concurrent_sessions: int
    notes: list[str]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_cpu() -> tuple[int, float]:
    try:
        import os as _os
        cores = _os.cpu_count() or 4
    except Exception:
        cores = 4
    try:
        import psutil  # type: ignore
        ram = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        ram = 8.0  # safe default
    return cores, round(ram, 1)


def _detect_gpu() -> tuple[str, Optional[str], Optional[float]]:
    """Returns (gpu, gpu_name, vram_gb)."""
    # NVIDIA CUDA via ctranslate2
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "NVIDIA CUDA", None  # VRAM unknown without nvidia-ml
    except Exception:
        pass
    # Apple MPS
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps", "Apple Silicon MPS", None
    except Exception:
        pass
    # torch CUDA fallback
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    except Exception:
        pass
    return "cpu", None, None


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def scan_hardware() -> HardwareProfile:
    os_name, arch = platform.system(), platform.machine()
    cpu_cores, ram_gb = _detect_cpu()
    gpu, gpu_name, vram_gb = _detect_gpu()
    return HardwareProfile(
        os=os_name,
        arch=arch,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpu=gpu,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        has_ctranslate2=_has_module("ctranslate2"),
        has_faster_whisper=_has_module("faster_whisper"),
        has_piper=_has_module("piper"),
        has_torch=_has_module("torch"),
    )


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------

def _stt_config(hw: HardwareProfile) -> StageConfig:
    """Pick STT device, compute type, model size."""
    # Prefer Parakeet TDT v3 on GPU-capable machines (faster + more accurate than Whisper)
    if hw.has_torch and hw.gpu in ("cuda", "mps"):
        return StageConfig(
            device=hw.gpu,
            compute_type="float16",
            model_size="parakeet-tdt-0.6b-v3",
            engine="parakeet",
            notes=f"Parakeet TDT v3 on {hw.gpu.upper()} — fast + accurate STT",
        )
    if hw.gpu == "cuda":
        # GPU: small model, float16 for speed/quality balance
        return StageConfig(
            device="cuda",
            compute_type="float16",
            model_size="small",
            engine="faster_whisper",
            notes="GPU CUDA: small model fp16, fast STT",
        )
    if hw.gpu == "mps":
        # Apple Silicon: faster-whisper uses CT2, not MPS — CPU path with int8
        return StageConfig(
            device="cpu",
            compute_type="int8",
            model_size="small",
            engine="faster_whisper",
            notes="Mac M1/M2: CT2 on CPU int8 (fallback when Parakeet unavailable)",
        )
    # CPU-only: small model int8, conservative
    return StageConfig(
        device="cpu",
        compute_type="int8",
        model_size="small",
        engine="faster_whisper",
        notes="CPU-only: small model int8",
    )


def _mt_config(hw: HardwareProfile) -> StageConfig:
    """Pick MT device. Opus-MT CT2 is CPU or CUDA."""
    if hw.gpu == "cuda":
        return StageConfig(
            device="cuda",
            compute_type="int8",
            model_size="base",
            engine="opus-mt",
            notes="GPU CUDA: MT on GPU int8",
        )
    return StageConfig(
        device="cpu",
        compute_type="int8",
        model_size="base",
        engine="opus-mt",
        notes="CPU: MT on CPU int8",
    )


def _tts_config(hw: HardwareProfile) -> StageConfig:
    """Pick TTS engine. Piper is the CPU baseline; OmniVoice/Kokoro need GPU."""
    if hw.gpu == "cuda" and (hw.vram_gb or 0) >= 4:
        return StageConfig(
            device="cuda",
            compute_type="float16",
            model_size="omnivoice",
            engine="omnivoice",
            notes="GPU with >=4GB VRAM: OmniVoice zero-shot cloning",
        )
    # CPU baseline: Piper
    return StageConfig(
        device="cpu",
        compute_type="int8",
        model_size="piper",
        engine="piper",
        notes="CPU baseline: Piper streaming",
    )


def _vad_config(hw: HardwareProfile) -> StageConfig:
    return StageConfig(
        device="cpu",
        compute_type="int8",
        model_size="webrtcvad",
        engine="webrtcvad",
        notes="WebRTC VAD, always CPU",
    )


def select_config(hw: HardwareProfile) -> AutoConfig:
    """Apply selection rules to produce a full AutoConfig."""
    stt = _stt_config(hw)
    mt = _mt_config(hw)
    tts = _tts_config(hw)
    vad = _vad_config(hw)

    streaming = True  # v2 is streaming by default
    notes: list[str] = []

    # Capacity estimate based on RAM + GPU
    if hw.gpu == "cuda" and (hw.vram_gb or 0) >= 8:
        max_sessions = 8
    elif hw.gpu == "cuda":
        max_sessions = 4
    elif hw.ram_gb >= 16:
        max_sessions = 4
    elif hw.ram_gb >= 8:
        max_sessions = 2
    else:
        max_sessions = 1

    if hw.gpu == "none":
        notes.append("No GPU detected — CPU-only path. Expect higher STT latency; consider small model + beam=1.")
    if hw.gpu == "mps":
        notes.append("Apple Silicon — faster-whisper runs on CPU via CT2; MPS not used for STT.")
    if hw.ram_gb < 8:
        notes.append("Low RAM — limit to 1-2 concurrent sessions and small models.")

    return AutoConfig(
        hardware=hw,
        stt=stt,
        mt=mt,
        tts=tts,
        vad=vad,
        streaming=streaming,
        max_concurrent_sessions=max_sessions,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "auto_config.json")


def save_config(cfg: AutoConfig) -> str:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    data = {
        "hardware": asdict(cfg.hardware),
        "stt": asdict(cfg.stt),
        "mt": asdict(cfg.mt),
        "tts": asdict(cfg.tts),
        "vad": asdict(cfg.vad),
        "streaming": cfg.streaming,
        "max_concurrent_sessions": cfg.max_concurrent_sessions,
        "notes": cfg.notes,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return CONFIG_PATH


def load_config() -> Optional[AutoConfig]:
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        hw = HardwareProfile(**data["hardware"])
        return AutoConfig(
            hardware=hw,
            stt=StageConfig(**data["stt"]),
            mt=StageConfig(**data["mt"]),
            tts=StageConfig(**data["tts"]),
            vad=StageConfig(**data["vad"]),
            streaming=data.get("streaming", True),
            max_concurrent_sessions=data.get("max_concurrent_sessions", 2),
            notes=data.get("notes", []),
        )
    except Exception:
        return None


def autoconfigure(force: bool = False) -> AutoConfig:
    """Scan hardware and select config. Force rescan even if cached."""
    if not force:
        cached = load_config()
        if cached is not None:
            return cached
    hw = scan_hardware()
    cfg = select_config(hw)
    save_config(cfg)
    return cfg


_cached_config: Optional[AutoConfig] = None


def get_config() -> Optional[AutoConfig]:
    """Return the cached auto-config without forcing a rescan."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config() or autoconfigure()
    return _cached_config


if __name__ == "__main__":
    cfg = autoconfigure(force=True)
    print(json.dumps({
        "hardware": asdict(cfg.hardware),
        "stt": asdict(cfg.stt),
        "mt": asdict(cfg.mt),
        "tts": asdict(cfg.tts),
        "vad": asdict(cfg.vad),
        "streaming": cfg.streaming,
        "max_concurrent_sessions": cfg.max_concurrent_sessions,
        "notes": cfg.notes,
    }, indent=2))