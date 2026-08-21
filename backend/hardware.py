"""
Shared per-stage hardware-backend detection.

Single source of truth for "what accelerator should this pipeline stage use",
replacing the ad-hoc `torch.backends.mps.is_available()` ternaries that used
to be duplicated in main.py and every backend/tts/*.py engine (mac-only,
wrong default on Windows/Linux). Each pipeline stage has its own priority
chain because the engines behind them support different backends:

  stt / mt          (CTranslate2)         cuda -> rocm -> cpu
  tts_baseline      (Piper/ONNX)          cuda -> rocm -> directml -> coreml -> cpu
  tts_clone         (PyTorch: XTTS,       cuda -> mps -> rocm -> cpu
                      OpenVoice, OmniVoice)

See specs/001-realtime-cross-platform-translation/plan.md for why these
chains differ per stage.
"""
import platform
from typing import Callable, Dict, List, Tuple

Stage = str  # "stt" | "mt" | "tts_baseline" | "tts_clone"


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def _rocm() -> bool:
    import torch
    return platform.system() == "Linux" and getattr(torch.version, "hip", None) is not None


def _mps() -> bool:
    import torch
    return platform.system() == "Darwin" and torch.backends.mps.is_available()


def _onnx_provider(name: str) -> bool:
    try:
        import onnxruntime as ort
        return name in ort.get_available_providers()
    except ImportError:
        return False


def _directml() -> bool:
    return platform.system() == "Windows" and _onnx_provider("DmlExecutionProvider")


def _coreml() -> bool:
    return platform.system() == "Darwin" and _onnx_provider("CoreMLExecutionProvider")


def _cpu() -> bool:
    return True


_CHAINS: Dict[Stage, List[Tuple[str, Callable[[], bool]]]] = {
    "stt": [("cuda", _cuda), ("rocm", _rocm), ("cpu", _cpu)],
    "mt": [("cuda", _cuda), ("rocm", _rocm), ("cpu", _cpu)],
    "tts_baseline": [("cuda", _cuda), ("rocm", _rocm), ("directml", _directml), ("coreml", _coreml), ("cpu", _cpu)],
    "tts_clone": [("cuda", _cuda), ("mps", _mps), ("rocm", _rocm), ("cpu", _cpu)],
}


def detect_backend(stage: Stage, probes: Dict[str, Callable[[], bool]] = None) -> str:
    """Return the first available backend name in `stage`'s priority chain.

    `probes` lets callers (tests) override individual probe functions by name,
    e.g. detect_backend("tts_clone", probes={"cuda": lambda: False, "mps": lambda: True}).
    """
    chain = _CHAINS[stage]
    overrides = probes or {}
    for name, probe in chain:
        if overrides.get(name, probe)():
            return name
    return "cpu"  # unreachable — the cpu probe always returns True
