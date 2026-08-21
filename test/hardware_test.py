"""Unit tests for backend/hardware.py's selection logic (spec task T010).

Mocks each probe individually so this runs with no real GPU present.
"""
from backend.hardware import detect_backend


def test_cuda_wins_when_available():
    assert detect_backend("tts_clone", probes={"cuda": lambda: True}) == "cuda"


def test_mps_used_on_mac_when_no_cuda():
    probes = {"cuda": lambda: False, "mps": lambda: True, "rocm": lambda: False}
    assert detect_backend("tts_clone", probes=probes) == "mps"


def test_rocm_used_on_linux_amd_when_no_cuda_or_mps():
    probes = {"cuda": lambda: False, "mps": lambda: False, "rocm": lambda: True}
    assert detect_backend("tts_clone", probes=probes) == "rocm"


def test_falls_back_to_cpu_with_no_gpu_present():
    probes = {"cuda": lambda: False, "mps": lambda: False, "rocm": lambda: False}
    assert detect_backend("tts_clone", probes=probes) == "cpu"


def test_tts_baseline_prefers_directml_on_windows_over_coreml():
    probes = {"cuda": lambda: False, "rocm": lambda: False, "directml": lambda: True, "coreml": lambda: False}
    assert detect_backend("tts_baseline", probes=probes) == "directml"


def test_tts_baseline_falls_back_to_coreml_on_mac():
    probes = {"cuda": lambda: False, "rocm": lambda: False, "directml": lambda: False, "coreml": lambda: True}
    assert detect_backend("tts_baseline", probes=probes) == "coreml"


def test_stt_and_mt_chain_has_no_mps_or_directml_step():
    # STT/MT (CTranslate2) only supports cuda/rocm/cpu — mps=True must not select mps
    probes = {"cuda": lambda: False, "rocm": lambda: False}
    assert detect_backend("stt", probes=probes) == "cpu"
    assert detect_backend("mt", probes=probes) == "cpu"
