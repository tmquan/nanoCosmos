"""
PyTorch Lightning callbacks bundled with nanocosmos.

Why this package exists
-----------------------
The training loop should stay free of visualisation, memory tracking
and other side-effects; we collect them here as Lightning callbacks
that the training script wires in declaratively from the Hydra config.

Public surface
--------------
* :class:`CudaEmptyCacheCallback` -- empties the PyTorch CUDA caching
  allocator around validation epochs, which is the easiest knob against
  slow-growing reserved memory on multi-day runs.
* :class:`CudaMemoryLoggerCallback` -- logs allocated / reserved /
  fragmentation gauges to TensorBoard under ``cuda_memory/*`` so you
  can tell a leak from a fragmentation drift at a glance.
* :class:`ImageLogger` -- per-head panels for the affinity recipe, one z
  slice per volume, under ``{stage}/{mode}/[{head}/]{panel}``
  (see :mod:`nanocosmos.callbacks.tensorboard.tags`).
* :class:`Joint3DImageLogger` -- joint-recipe logger.  Source, ssl recon
  target, raw-head prediction, and the Wan-decoder diagnostic are
  orthogonal mid-planes (``xy`` / ``xz`` / ``yz``) on the 4 nm grid.
  sft segmentation panels use a foreground-content z slice.  Tags are
  ``{stage}/automatic/{task}/...``.  Every rank captures batches; rank 0
  writes at ``on_validation_epoch_end``, including Lightning's sanity
  check (step-0 panels after sanity validation).

Extending this module
---------------------
Add a new ``nanocosmos/callbacks/<name>.py`` defining a
:class:`pytorch_lightning.callbacks.Callback` subclass and re-export
it here.  Wire it into the training script's ``setup_callbacks``.
"""

from nanocosmos.callbacks.memory import (
    CudaEmptyCacheCallback,
    CudaMemoryLoggerCallback,
)
from nanocosmos.callbacks.tensorboard import ImageLogger, Joint3DImageLogger

__all__ = [
    "CudaEmptyCacheCallback",
    "CudaMemoryLoggerCallback",
    "ImageLogger",
    "Joint3DImageLogger",
]
