"""Loss package for the affinity + foreground Mutex Watershed head.

The public loss surface is intentionally small:

* :class:`AffinityFGLoss` consumes the model's single
  ``[B, n_aff + 2, *spatial]`` head tensor and supervises the
  per-offset affinities and the scalar foreground probability.  At
  evaluation the affinities are agglomerated into instances by the
  Mutex Watershed (:mod:`nanocosmos.inference.mutex_watershed`).
  :data:`HEAD_CHANNELS` is the default-offset width (16).
* :class:`DiceBCEFocalLoss` is the composite Dice + BCE + Focal
  supervisor used by :class:`AffinityFGLoss` for the foreground head --
  exposed here so external consumers can instantiate it directly with
  the same numerics.
* :class:`Joint3DReconSegLoss` is the joint reconstruction + segmentation
  loss (tasks ``ssl`` and ``sft``) on that same head.  Training configs
  use 30 affinity offsets, so the head is 32 channels (affinities + sem
  + raw).  Pretrain (``configs/nanocosmos-4B-pretrain.yaml``) sets
  ``recon_loss="l1"``, ``weight_rec=1``, ``weight_dif=1``,
  ``weight_ssl=5``, ``weight_seg=0``, ``weight_sft=0``.
  ``weight_dif`` scales the rectified-flow MSE on ``proj_out`` next to
  the L1; the task head stays the one-step readout.
  :data:`HEAD_CHANNELS` remains the
  default-offset width (14 affinities + sem + raw = 16).
* :mod:`nanocosmos.losses._common` owns the canonical channel layout
  (:data:`AFFINITY_OFFSETS`, :data:`AFF_SLICE`, :data:`SEM_SLICE`,
  :data:`RAW_SLICE`), the
  affinity-target / validity-mask builders, field-slicing helpers, and
  shared numerical utilities.  The head emits raw logits (no activation
  in the forward pass).
"""

from nanocosmos.losses.affinity import AffinityFGLoss
from nanocosmos.losses.dice_bce_focal import DiceBCEFocalLoss
from nanocosmos.losses.joint3d import (
    SSL,
    SFT,
    Joint3DReconSegLoss,
)
from nanocosmos.losses._common import (
    AFF_CHANNELS,
    AFF_NAMES,
    AFF_SLICE,
    AFFINITY_OFFSETS,
    HEAD_CHANNELS,
    HEAD_LAYOUT,
    N_AFF,
    N_PULL,
    RAW_SLICE,
    SEM_SLICE,
    affinity_target_from_offsets,
    affinity_validity_mask,
    head_channels_for,
    head_slices,
    offset_names,
    shift_nd,
    shift_replicate,
    slice_head,
    stable_bce_on_probs,
)

__all__ = [
    "AffinityFGLoss",
    "Joint3DReconSegLoss",
    "SSL",
    "SFT",
    "DiceBCEFocalLoss",
    "HEAD_CHANNELS",
    "HEAD_LAYOUT",
    "AFFINITY_OFFSETS",
    "AFF_SLICE",
    "SEM_SLICE",
    "RAW_SLICE",
    "N_AFF",
    "N_PULL",
    "AFF_CHANNELS",
    "AFF_NAMES",
    "offset_names",
    "head_channels_for",
    "head_slices",
    "slice_head",
    "affinity_target_from_offsets",
    "affinity_validity_mask",
    "shift_nd",
    "shift_replicate",
    "stable_bce_on_probs",
]
