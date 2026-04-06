"""GPU/CPU abstraction layer. Uses CuPy if available, numpy otherwise."""

import numpy as np

try:
    import cupy as cp
    # Test that CUDA runtime actually works (driver + toolkit both needed)
    cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = True
except (ImportError, Exception):
    # CuPy not installed, or CUDA toolkit missing (libnvrtc.so, etc.)
    cp = np
    GPU_AVAILABLE = False


def to_gpu(arr):
    """Move numpy array to GPU."""
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr


def to_cpu(arr):
    """Move GPU array to CPU."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def get_xp(arr):
    """Get the array module (cupy or numpy) for the given array."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np
