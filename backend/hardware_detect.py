"""Hardware/OS detection and self-configuration.

Scans the current machine and returns a complete profile:
- OS, arch, CPU, GPU (vendor, VRAM, CUDA/DirectML/MPS)
- Audio stack (BlackHole/VB-Cable/PipeWire, loopback support)
- Python env (torch available? CUDA? MPS? ONNX Runtime providers?)
- Recommended engine selection per stage (STT/MT/TTS/VAD)
- Model sizes and device assignments
- Path B routing commands

Used by `scripts/setup.py` (T012) and the live server to auto-configure.
"""
from __future__ import annotations
import os
import platform
import subprocess
import sys
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal
from pathlib import Path


@dataclass
class GPUInfo:
    vendor: str  # "nvidia" | "amd" | "apple" | "intel" | "unknown"
    name: str
    vram_mb: Optional[int] = None
    cuda_available: bool = False
    mps_available: bool = False
    directml_available: bool = False
    opencl_available: bool = False
    driver_version: Optional[str] = None


@dataclass
class AudioStack:
    os: str  # "macos" | "windows" | "linux"
    loopback_available: bool = False
    virtual_mic_available: bool = False
    virtual_mic_name: Optional[str] = None  # "BlackHole 2ch" | "VB-Cable" | "PipeWire null-sink"
    pipewire: bool = False
    pulseaudio: bool = False
    blackhole_installed: bool = False
    vb_cable_installed: bool = False
    setup_commands: list[str] = field(default_factory=list)


@dataclass
class PythonEnv:
    version: str
    torch_available: bool = False
    torch_cuda: bool = False
    torch_mps: bool = False
    onnxruntime_providers: list[str] = field(default_factory=list)
    ctranslate2_available: bool = False
    faster_whisper_available: bool = False


@dataclass
class EngineRecommendation:
    stage: str
    engine: str
    device: str
    model_size: Optional[str] = None
    notes: str = ""
    licence_ok: bool = True


@dataclass
class HardwareProfile:
    hostname: str
    os: str
    arch: str
    cpu_cores: int
    cpu_model: str
    ram_gb: float
    gpu: GPUInfo
    audio: AudioStack
    python: PythonEnv
    engine_recommendations: list[EngineRecommendation] = field(default_factory=list)
    profile_version: str = "1.0"


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def detect_gpu() -> GPUInfo:
    gpu = GPUInfo(vendor="unknown", name="unknown")
    
    # NVIDIA (nvidia-smi)
    code, out, _ = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if code == 0 and out:
        for line in out.splitlines():
            name, vram, drv = [x.strip() for x in line.split(",")]
            gpu.vendor = "nvidia"
            gpu.name = name
            gpu.vram_mb = int(vram)
            gpu.driver_version = drv
            gpu.cuda_available = True
            break
    
    # AMD (rocm-smi)
    if gpu.vendor == "unknown":
        code, out, _ = _run(["rocm-smi", "--showproductname"])
        if code == 0 and out:
            gpu.vendor = "amd"
            gpu.name = out.splitlines()[-1].strip()
            gpu.opencl_available = True
    
    # Apple Silicon (sysctl)
    if gpu.vendor == "unknown" and platform.system() == "Darwin":
        code, out, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if code == 0 and ("Apple" in out or "M1" in out or "M2" in out or "M3" in out or "M4" in out):
            gpu.vendor = "apple"
            gpu.name = out.strip()
            gpu.mps_available = True
    
    # Intel (lscpu + check for integrated)
    if gpu.vendor == "unknown" and platform.system() == "Linux":
        code, out, _ = _run(["lscpu"])
        if code == 0 and "Intel" in out:
            gpu.vendor = "intel"
            gpu.name = "Intel integrated"
            gpu.opencl_available = True
    
    # Check Python-level GPU backends
    try:
        import torch
        if torch.cuda.is_available():
            gpu.cuda_available = True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu.mps_available = True
        if hasattr(torch.backends, "directml") and torch.backends.directml.is_available():
            gpu.directml_available = True
    except Exception:
        pass
    
    return gpu


def detect_audio() -> AudioStack:
    system = platform.system().lower()
    audio = AudioStack(os=system)
    
    if system == "darwin":
        # Check for BlackHole
        code, out, _ = _run(["system_profiler", "SPAudioDataType"])
        if code == 0 and "BlackHole" in out:
            audio.blackhole_installed = True
            audio.virtual_mic_available = True
            audio.virtual_mic_name = "BlackHole 2ch"
            audio.loopback_available = True
        # Loopback via BlackHole or built-in
        audio.setup_commands = [
            "brew install blackhole-2ch  # virtual mic + loopback",
        ]
    
    elif system == "windows":
        # Check for VB-Cable
        code, out, _ = _run(["powershell", "-Command", "Get-PnpDevice -Class AudioEndpoint | Where-Object {$_.FriendlyName -like '*VB-Cable*'}"])
        if code == 0 and "VB-Cable" in out:
            audio.vb_cable_installed = True
            audio.virtual_mic_available = True
            audio.virtual_mic_name = "VB-Cable"
            audio.loopback_available = True
        audio.setup_commands = [
            "Install VB-Cable from https://vb-audio.com/Cable/  # virtual mic + loopback",
        ]
    
    elif system == "linux":
        # Check PipeWire / PulseAudio
        code, out, _ = _run(["pactl", "info"])
        if code == 0:
            audio.pulseaudio = True
            if "PipeWire" in out:
                audio.pipewire = True
                audio.loopback_available = True
                audio.virtual_mic_available = True
                audio.virtual_mic_name = "PipeWire null-sink"
        audio.setup_commands = [
            "pactl load-module module-null-sink sink_name=bp_virtual_sink  # virtual mic",
            "pactl load-module module-loopback source=bp_virtual_sink.monitor  # loopback",
        ]
    
    return audio


def detect_python_env() -> PythonEnv:
    env = PythonEnv(version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    try:
        import torch
        env.torch_available = True
        env.torch_cuda = torch.cuda.is_available()
        env.torch_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        pass
    
    try:
        import onnxruntime as ort
        env.onnxruntime_providers = ort.get_available_providers()
    except Exception:
        pass
    
    try:
        import ctranslate2
        env.ctranslate2_available = True
    except Exception:
        pass
    
    try:
        import faster_whisper
        env.faster_whisper_available = True
    except Exception:
        pass
    
    return env


def recommend_engines(profile: HardwareProfile) -> list[EngineRecommendation]:
    """Generate engine recommendations based on detected hardware."""
    recs = []
    gpu = profile.gpu
    audio = profile.audio
    py = profile.python
    
    # VAD - always webrtcvad (tiny, CPU, no deps)
    recs.append(EngineRecommendation(
        stage="vad", engine="webrtcvad", device="cpu",
        notes="BSD-3-Clause, ~1 MB, always available"))
    
    # STT - faster-whisper
    if gpu.vendor == "nvidia" and gpu.vram_mb and gpu.vram_mb >= 6000:
        recs.append(EngineRecommendation(
            stage="stt", engine="faster_whisper", device="cuda",
            model_size="small", notes="GPU: small model fits in VRAM"))
    elif gpu.vendor == "apple":
        recs.append(EngineRecommendation(
            stage="stt", engine="faster_whisper", device="coreml",
            model_size="small", notes="Apple Silicon: CoreML backend"))
    else:
        recs.append(EngineRecommendation(
            stage="stt", engine="faster_whisper", device="cpu",
            model_size="base", notes="CPU: base is faster+better WER than small (measured)"))
    
    # SK STT override
    recs.append(EngineRecommendation(
        stage="stt", engine="faster_whisper", device="cpu",
        model_size="small-sk", notes="Slovak fine-tune (local ct2_models/whisper-small-sk)"))
    
    # MT - opus-mt (CT2, always available if weights exist)
    recs.append(EngineRecommendation(
        stage="mt", engine="opus-mt", device="cpu",
        notes="CT2 int8, per-direction weights in ct2_models/"))
    
    if gpu.vendor == "nvidia" and gpu.vram_mb and gpu.vram_mb >= 4000:
        recs.append(EngineRecommendation(
            stage="mt", engine="nllb", device="cuda",
            notes="NLLB-200 (CC-BY-NC, 200 langs) — needs --allow-noncommercial"))
    
    # TTS - Piper baseline (CPU, CoreML on Mac)
    if gpu.vendor == "apple":
        recs.append(EngineRecommendation(
            stage="tts", engine="piper", device="coreml",
            notes="MIT, RTF ~0.03 on CoreML, per-voice .onnx in backend/tts/piper_models/"))
    else:
        recs.append(EngineRecommendation(
            stage="tts", engine="piper", device="cpu",
            notes="MIT, RTF ~0.05 on CPU, per-voice .onnx in backend/tts/piper_models/"))
    
    # Cloning engines (GPU-only)
    if gpu.vendor == "nvidia" and gpu.vram_mb and gpu.vram_mb >= 6000:
        recs.append(EngineRecommendation(
            stage="tts", engine="omnivoice", device="cuda",
            notes="Cloning, 12 steps ~5.8 GB VRAM, SUPPORTS_STREAMING, licence unknown"))
        recs.append(EngineRecommendation(
            stage="tts", engine="xtts", device="cuda",
            licence_ok=False,
            notes="Coqui XTTS v2, non-commercial (CPML), needs --allow-noncommercial"))
    
    if gpu.vendor == "apple":
        recs.append(EngineRecommendation(
            stage="tts", engine="omnivoice", device="mps",
            notes="Cloning on MPS, slower than CUDA but works"))
        recs.append(EngineRecommendation(
            stage="tts", engine="xtts", device="mps",
            licence_ok=False,
            notes="Coqui XTTS v2 on MPS, non-commercial"))
    
    # Voice conversion layer (RVC-MLX on Mac, onyx on Linux/Windows)
    if gpu.vendor == "apple":
        recs.append(EngineRecommendation(
            stage="tts", engine="rvc-mlx", device="coreml",
            notes="RVC-MLX: train once, any TTS → your voice. 10× realtime on M-series."))
    else:
        recs.append(EngineRecommendation(
            stage="tts", engine="onyx", device="cpu",
            notes="ONNX Runtime RVC: cross-platform CPU/CUDA/DirectML, MIT."))
    
    return recs


def detect_all() -> HardwareProfile:
    """Full hardware/OS detection and recommendation."""
    hostname = platform.node()
    os_name = platform.system().lower()
    arch = platform.machine().lower()
    cpu_cores = os.cpu_count() or 1
    
    # CPU model
    cpu_model = "unknown"
    if os_name == "darwin":
        code, out, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if code == 0:
            cpu_model = out.strip()
    elif os_name == "linux":
        code, out, _ = _run(["lscpu"])
        if code == 0:
            for line in out.splitlines():
                if line.startswith("Model name:"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    
    # RAM
    ram_gb = 0
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        if os_name == "darwin":
            code, out, _ = _run(["sysctl", "-n", "hw.memsize"])
            if code == 0:
                ram_gb = round(int(out) / (1024**3), 1)
        elif os_name == "linux":
            code, out, _ = _run(["grep", "MemTotal", "/proc/meminfo"])
            if code == 0:
                ram_gb = round(int(out.split()[1]) / (1024**2), 1)
    
    gpu = detect_gpu()
    audio = detect_audio()
    py = detect_python_env()
    
    profile = HardwareProfile(
        hostname=hostname, os=os_name, arch=arch,
        cpu_cores=cpu_cores, cpu_model=cpu_model, ram_gb=ram_gb,
        gpu=gpu, audio=audio, python=py
    )
    profile.engine_recommendations = recommend_engines(profile)
    return profile


def profile_to_config(profile: HardwareProfile) -> dict:
    """Convert profile to a config dict for setup.py / main.py."""
    config = {
        "hardware": {
            "os": profile.os,
            "arch": profile.arch,
            "cpu_cores": profile.cpu_cores,
            "ram_gb": profile.ram_gb,
            "gpu": asdict(profile.gpu),
        },
        "audio": asdict(profile.audio),
        "engines": {r.stage: r.engine for r in profile.engine_recommendations if r.licence_ok},
        "engine_details": {r.stage: asdict(r) for r in profile.engine_recommendations},
        "session_defaults": {
            "stt_model_size": next((r.model_size for r in profile.engine_recommendations if r.stage == "stt" and not r.notes.startswith("Slovak")), "base"),
            "tts_engine": next((r.engine for r in profile.engine_recommendations if r.stage == "tts" and r.engine == "piper"), "piper"),
        }
    }
    return config


if __name__ == "__main__":
    profile = detect_all()
    print(json.dumps(asdict(profile), indent=2, default=str))