"""
Device-agnostic helpers for running RFantibody on CPU, CUDA or Tecorigin SDAA.

The pipeline was written against ``torch.cuda``. On Tecorigin SDAA ``torch.cuda``
exists only as a stub (``torch.cuda.is_available()`` is ``False``) while the real
API lives in ``torch.sdaa``. This module selects the best available accelerator
once and exposes a small, uniform surface used across the codebase.

Backend selection order (override with ``RFANTIBODY_DEVICE`` = sdaa|cuda|cpu):
    1. SDAA   (``torch.sdaa.is_available()``)
    2. CUDA   (``torch.cuda.is_available()``)
    3. CPU
"""

import os
from contextlib import nullcontext

import torch

try:  # importing torch_sdaa registers torch.sdaa
    import torch_sdaa  # noqa: F401
except Exception:  # pragma: no cover - torch.sdaa may already exist
    pass


def _sdaa_available() -> bool:
    return hasattr(torch, "sdaa") and torch.sdaa.is_available()


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def get_backend() -> str:
    """Return the active backend name: ``'sdaa'``, ``'cuda'`` or ``'cpu'``."""
    forced = os.environ.get("RFANTIBODY_DEVICE", "").lower()
    if forced in ("sdaa", "cuda", "cpu"):
        return forced
    if _sdaa_available():
        return "sdaa"
    if _cuda_available():
        return "cuda"
    return "cpu"


def accelerator_available() -> bool:
    """True when a GPU-like accelerator (SDAA or CUDA) is available."""
    return get_backend() in ("sdaa", "cuda")


def _backend_module():
    """Return the torch submodule implementing the active accelerator API."""
    return getattr(torch, get_backend())


def device_str(index: int = 0) -> str:
    """Return a device string, e.g. ``'sdaa:0'``, ``'cuda:0'`` or ``'cpu'``."""
    backend = get_backend()
    if backend == "cpu":
        return "cpu"
    return f"{backend}:{index}"


def get_device(index: int = 0) -> torch.device:
    return torch.device(device_str(index))


def empty_cache() -> None:
    if accelerator_available():
        _backend_module().empty_cache()


def synchronize() -> None:
    if accelerator_available():
        _backend_module().synchronize()


def manual_seed_all(seed: int) -> None:
    if accelerator_available():
        _backend_module().manual_seed_all(seed)


def current_device() -> int:
    if accelerator_available():
        return _backend_module().current_device()
    return -1


def device_count() -> int:
    if accelerator_available():
        return _backend_module().device_count()
    return 0


def get_device_name(index: int = 0) -> str:
    if not accelerator_available():
        return "CPU"
    try:
        return _backend_module().get_device_name(index)
    except Exception:
        return device_str(index)


def max_memory_allocated(index: int = 0) -> int:
    if not accelerator_available():
        return 0
    return _backend_module().max_memory_allocated(index)


def memory_allocated(index: int = 0) -> int:
    if not accelerator_available():
        return 0
    return _backend_module().memory_allocated(index)


def reset_peak_memory_stats(index: int = 0) -> None:
    if accelerator_available():
        _backend_module().reset_peak_memory_stats(index)


def autocast(enabled: bool = True):
    """Return an autocast context for the active backend (no-op if unsupported)."""
    if not enabled or not accelerator_available():
        return nullcontext()
    try:
        return torch.autocast(device_type=get_backend(), enabled=enabled)
    except Exception:
        return nullcontext()


def device_memory_info(index: int = 0):
    """Return ``(free_bytes, total_bytes)`` for the active accelerator, or None."""
    if not accelerator_available():
        return None
    try:
        return _backend_module().mem_get_info(index)
    except Exception:
        return None