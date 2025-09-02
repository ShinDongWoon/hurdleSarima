from typing import Optional


def torch_device(prefer_mps: bool = True):
    try:
        import torch
        if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def gpu_available() -> bool:
    """Return True if a GPU device is available via common libraries."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except Exception:
        pass
    try:
        import cuml  # noqa: F401
        return True
    except Exception:
        return False
