"""GPU-availability gate for the connectomics transforms.

This module exposes a single cross-module helper, :func:`_use_gpu`, which the
transform modules (:mod:`nanocosmos.transforms.label`,
:mod:`nanocosmos.transforms.find_boundaries`) consult to decide whether to run
the cucim/cupy GPU path or the scipy/skimage CPU fallback.

When cucim is installed and a CUDA device is functional, ``_use_gpu()`` returns
True.  The probe is cached per process id, so each worker checks once.
DataLoaders in this package use the ``forkserver`` start method.  A CUDA
context does not survive ``os.fork()``; if the probe fails, the CPU path is
used.  Set ``NANOCOSMOS_FORCE_CPU=1`` to force the CPU path.
"""

import os
from typing import Optional

_pid_gpu_cache: dict = {}

_CUCIM_AVAILABLE: Optional[bool] = None


def _cucim_available() -> bool:
    global _CUCIM_AVAILABLE
    if _CUCIM_AVAILABLE is None:
        try:
            import cucim  # noqa: F401
            _CUCIM_AVAILABLE = True
        except ImportError:
            _CUCIM_AVAILABLE = False
    return _CUCIM_AVAILABLE


def _use_gpu() -> bool:
    """Return True when the GPU (cucim/cupy) code path should be used.

    Checks cucim availability, the ``NANOCOSMOS_FORCE_CPU`` env var (the
    ``NEURONS_FORCE_CPU`` name is still honored), and whether CUDA initializes
    in this process.  The result is cached per PID.
    """
    if not _cucim_available():
        return False
    if os.environ.get("NANOCOSMOS_FORCE_CPU", "") or os.environ.get("NEURONS_FORCE_CPU", ""):
        return False
    pid = os.getpid()
    if pid in _pid_gpu_cache:
        return _pid_gpu_cache[pid]
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        _pid_gpu_cache[pid] = True
        return True
    except Exception:
        _pid_gpu_cache[pid] = False
        return False
