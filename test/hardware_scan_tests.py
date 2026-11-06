"""Tests for backend/hardware_scan.py — hardware auto-configuration."""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

from backend.hardware_scan import (
    scan_hardware, select_config, autoconfigure, load_config, save_config,
    HardwareProfile, StageConfig, AutoConfig,
)


def test_scan_hardware_returns_profile():
    hw = scan_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.os in ("Darwin", "Linux", "Windows")
    assert hw.cpu_cores >= 1
    assert hw.ram_gb > 0
    assert hw.gpu in ("cuda", "mps", "cpu", "none")
    assert isinstance(hw.has_ctranslate2, bool)
    assert isinstance(hw.has_faster_whisper, bool)


def test_select_config_returns_full_stack():
    hw = HardwareProfile(
        os="Darwin", arch="arm64", cpu_cores=8, ram_gb=8.0,
        gpu="cpu", gpu_name=None, vram_gb=None,
        has_ctranslate2=True, has_faster_whisper=True, has_piper=True, has_torch=False,
    )
    cfg = select_config(hw)
    assert isinstance(cfg, AutoConfig)
    assert cfg.stt.engine == "faster_whisper"
    assert cfg.mt.engine == "opus-mt"
    assert cfg.tts.engine == "piper"
    assert cfg.vad.engine == "webrtcvad"
    assert cfg.streaming is True
    assert cfg.max_concurrent_sessions >= 1


def test_select_config_cuda_reduces_sessions():
    hw_cpu = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=4, ram_gb=8.0,
        gpu="cpu", gpu_name=None, vram_gb=None,
        has_ctranslate2=True, has_faster_whisper=True, has_piper=True, has_torch=False,
    )
    hw_gpu = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=4, ram_gb=16.0,
        gpu="cuda", gpu_name="T4", vram_gb=16.0,
        has_ctranslate2=True, has_faster_whisper=True, has_piper=True, has_torch=True,
    )
    cfg_cpu = select_config(hw_cpu)
    cfg_gpu = select_config(hw_gpu)
    assert cfg_gpu.max_concurrent_sessions >= cfg_cpu.max_concurrent_sessions
    assert cfg_gpu.stt.device == "cuda"
    assert cfg_cpu.stt.device == "cpu"


def test_save_load_config_roundtrip(tmp_path, monkeypatch):
    # Point CONFIG_PATH to a temp dir
    import backend.hardware_scan as hs
    tmpdir = tmp_path / "config"
    tmpdir.mkdir()
    monkeypatch.setattr(hs, "CONFIG_DIR", str(tmpdir))
    monkeypatch.setattr(hs, "CONFIG_PATH", str(tmpdir / "auto_config.json"))

    hw = HardwareProfile(
        os="Linux", arch="x86_64", cpu_cores=4, ram_gb=8.0,
        gpu="cpu", gpu_name=None, vram_gb=None,
        has_ctranslate2=True, has_faster_whisper=True, has_piper=True, has_torch=False,
    )
    cfg = select_config(hw)
    path = save_config(cfg)
    assert os.path.exists(path)

    loaded = load_config()
    assert loaded is not None
    assert loaded.stt.model_size == cfg.stt.model_size
    assert loaded.tts.engine == cfg.tts.engine
    assert loaded.max_concurrent_sessions == cfg.max_concurrent_sessions


def test_autoconfigure_idempotent():
    cfg1 = autoconfigure(force=True)
    cfg2 = autoconfigure(force=False)
    assert cfg1.stt.model_size == cfg2.stt.model_size
    assert cfg1.tts.engine == cfg2.tts.engine