"""Model-free test: the CUDA preload helper is a safe no-op without GPU libraries and never raises."""
import sys
from backend.cuda_libs import preload_pip_cuda_libs


def test_preload_is_safe_everywhere():
    n = preload_pip_cuda_libs()
    assert isinstance(n, int) and n >= 0
    if not sys.platform.startswith("linux"):
        assert n == 0


def test_preload_twice_is_fine():
    assert preload_pip_cuda_libs() >= 0 and preload_pip_cuda_libs() >= 0
