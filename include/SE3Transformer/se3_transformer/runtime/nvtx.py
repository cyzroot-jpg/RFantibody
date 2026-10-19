"""
Portable replacement for ``torch.cuda.nvtx.range``.

NVTX is a CUDA-only profiling annotation; ``torch.cuda.nvtx`` raises on
non-CUDA devices (e.g. Tecorigin SDAA). These helpers are no-ops on any device
and are safe to use as context managers:

    with nvtx_range('name'):
        ...
"""

from contextlib import contextmanager


@contextmanager
def range(msg, *args, **kwargs):  # noqa: A001 - mirror the torch.cuda.nvtx name
    yield


def range_push(msg, *args, **kwargs):
    pass


def range_pop():
    pass


def mark(msg, *args, **kwargs):
    pass