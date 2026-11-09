"""Make pip-installed CUDA libraries visible to CTranslate2 and onnxruntime (Linux).

`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` puts libcublas/libcudnn under site-packages/nvidia/*/lib, a directory the dynamic loader does
not search, so faster-whisper fails with "libcublas.so.12 not found" on a machine with a GPU but no CUDA toolkit. Loading them by path with
RTLD_GLOBAL before the first engine import fixes that without asking the user to set LD_LIBRARY_PATH. No-op on macOS and Windows.
"""
from __future__ import annotations

import glob
import os
import site
import sys


def is_blas_override(path: str) -> bool:
    """libnvblas exports the CPU BLAS symbols (sgemm_ ...): loaded RTLD_GLOBAL it hijacks numpy and aborts the process with
    "[NVBLAS] CPU Blas library need to be provided" at the first matrix product (found 2026-11-09 on a Kaggle T4: Voxtral's feature
    extractor killed the server). CTranslate2, onnxruntime and torch never need it."""
    return os.path.basename(path).startswith("libnvblas")


def preload_pip_cuda_libs() -> int:
    """Load every nvidia/*/lib/lib*.so* found in site-packages; returns how many loaded. Safe to call twice."""
    if not sys.platform.startswith("linux"):
        return 0
    import ctypes
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    pending = sorted({p for r in roots for p in glob.glob(os.path.join(r, "nvidia", "*", "lib", "lib*.so*")) if not is_blas_override(p)})
    loaded = 0
    for _ in range(3):          # a library may need another one that sorts after it: retry the failures
        still = []
        for so in pending:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                still.append(so)
        if not still or len(still) == len(pending):
            break
        pending = still
    return loaded
