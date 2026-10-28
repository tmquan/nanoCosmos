"""
TensorBoard image logger package.

Tags are assembled by :class:`~nanocosmos.callbacks.tensorboard.tags.TagContext`::

    {stage}/{mode}/[{head}/]{panel}

* ``stage`` -- ``"train"`` | ``"val"``
* ``mode``  -- ``"automatic"`` (the mode the training script passes)
* ``head``  -- omitted by :class:`ImageLogger`.  :class:`Joint3DImageLogger`
  sets it to the task name (``ssl`` or ``sft``).
* ``panel`` -- field path, plus ``/{xy,xz,yz}`` on the joint EM panels.

:class:`ImageLogger` (affinity recipe) takes one z slice per volume via
:func:`~nanocosmos.callbacks.tensorboard.viz._to_2d` (the central slice
when no relative depth is passed): ``true/image``, ``true/label``,
``true/sem``, ``true/wan_decoder``, ``pred/sem``, ``pred/raw``,
``pred/label/{pre,mul}``, ``aff/{true,pred}/{offset}``.

:class:`Joint3DImageLogger` logs the EM source (``true/image``), the ssl
recon target (``true/target``), the task-head reconstruction
(``pred/recon``), the Euler-integrated reconstruction (``pred/Euler``),
and the Wan-decoder diagnostic (``pred/wan_decoder``) as orthogonal
mid-planes ``xy`` (mid-z), ``xz`` (mid-y), and ``yz`` (mid-x) on the 4 nm
source grid.  A coarser plane is nearest-upsampled onto that grid
(:func:`~nanocosmos.callbacks.tensorboard.viz._resize_2d`).  sft
segmentation panels still pick one z slice with ``_best_depth_frac``.
Tags look like ``{stage}/automatic/{task}/{kind}/{xy|xz|yz}``.

Affinity names come from :func:`nanocosmos.losses.offset_names`
(for example ``01_pull_z-1``), with a 1-based prefix so alphabetical
TensorBoard order follows offset order.  They live under ``aff/`` so the
image / label / sem panels stay grouped.

Scalar logs from :class:`nanocosmos.modules.base.BaseCircuitModule` use
``{stage}/{mode}/loss/...``, so each field's images sit near its scalars.

Module layout::

    tags.py            -- TagContext (``{stage}/{mode}/[{head}/]{panel}``)
    viz.py             -- orthogonal mid-planes, 2-D resize, depth-slice
                          extraction, per-image min-max, label HSV palette
    heads.py           -- ImageLogger affinity + sem + raw panels, including
                          Mutex Watershed ``pred/label``
    image_logger.py    -- ``ImageLogger``
    joint3d_logger.py  -- ``Joint3DImageLogger``

Every rank caches its batches and runs the epoch-end forward (the model
forward is collective under FSDP).  Rank 0 writes.  With a TensorBoard
logger the other ranks receive Lightning's dummy experiment, whose
``add_*`` methods do not write.  Images go out from
``on_{train,validation}_epoch_end``.  ``on_validation_epoch_end`` also
runs at the end of Lightning's sanity-check validation, so step-0 panels
appear after that pass.  2-D batches are logged as-is.
"""

from nanocosmos.callbacks.tensorboard.image_logger import ImageLogger
from nanocosmos.callbacks.tensorboard.joint3d_logger import Joint3DImageLogger

__all__ = ["ImageLogger", "Joint3DImageLogger"]
