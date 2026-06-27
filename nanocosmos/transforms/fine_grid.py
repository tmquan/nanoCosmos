"""
Resample a native-resolution patch onto the fixed fine (small-voxel) grid.

The joint recipe (doc/RESOLUTION_LADDER.md) runs the network on a fixed fine
grid (e.g. 4 nm, 320x256x256).  Each source volume is read at its **native**
voxel size and brought onto that grid by :class:`ToFineGridd`:

* ``image``       -> resampled (trilinear) to the fine grid -- the network input;
* ``label``       -> **left at native** (the segmentation loss pools the fine
  head back down to it);
* ``recon_image`` -> the clean-EM reconstruction target, placed on the
  *coarser of native / fine* (so it is never finer than the grid the model can
  produce): for a coarse rung (FIB 8 nm) it stays native; for a fine rung whose
  native voxel is *smaller* than the grid (COSEM 3.24 nm) it is downsampled to
  the grid.

On the ``sft`` branch the clean native image is captured as ``recon_image``
before the image is resampled (the raw data-consistency target).  On ``dapt``
the degradation transform has already written ``recon_image`` (the clean native
patch); this transform only moves the two onto their grids.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.transforms import MapTransform


def _resample(arr, size: Sequence[int], mode: str):
    """Resample a ``[C, D, H, W]`` tensor/array to ``size`` (D, H, W)."""
    size = tuple(int(s) for s in size)
    is_tensor = isinstance(arr, torch.Tensor)
    is_meta = hasattr(arr, "meta")
    meta = arr.meta if is_meta else None
    applied = arr.applied_operations if is_meta else None
    t = arr if is_tensor else torch.as_tensor(np.asarray(arr))
    if tuple(t.shape[-3:]) == size:
        return arr
    kw = {"mode": "nearest"} if mode == "nearest" else {"mode": "trilinear", "align_corners": False}
    out = F.interpolate(t.float()[None], size=size, **kw)[0].to(t.dtype)
    if is_meta:
        from monai.data import MetaTensor

        return MetaTensor(out, meta=meta, applied_operations=applied)
    return out if is_tensor else out.numpy()


class ToFineGridd(MapTransform):
    """Place ``image`` on the fine grid, keep ``label`` native, grid ``recon_image``.

    Args:
        image_size: Fine-grid spatial size ``(D, H, W)`` for the network input.
        recon_size: Spatial size for the reconstruction target (coarser of
            native / fine).  If ``None``, the recon target is left at native.
        image_key / recon_key / label_key: dict keys.
        set_recon_from_image: capture ``recon_image = image.clone()`` (the clean
            native patch) before resampling the image -- used on the ``sft``
            branch (on ``dapt`` the degradation transform already wrote it).
        task: value written to ``data["task"]`` (``"dapt"`` / ``"sft"``).
    """

    def __init__(
        self,
        image_size: Sequence[int],
        recon_size: Optional[Sequence[int]] = None,
        *,
        image_key: str = "image",
        recon_key: str = "recon_image",
        label_key: str = "label",
        set_recon_from_image: bool = False,
        task: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([image_key], allow_missing_keys=allow_missing_keys)
        self.image_size = tuple(int(s) for s in image_size)
        self.recon_size = tuple(int(s) for s in recon_size) if recon_size is not None else None
        self.image_key = image_key
        self.recon_key = recon_key
        self.label_key = label_key
        self.set_recon_from_image = bool(set_recon_from_image)
        self.task = task

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        img = d[self.image_key]

        # Capture the clean native image as the recon target (sft branch).
        if self.set_recon_from_image and self.recon_key not in d:
            d[self.recon_key] = img.clone() if isinstance(img, torch.Tensor) else np.array(img)

        # Recon target onto its grid (coarser of native / fine).
        if self.recon_key in d and self.recon_size is not None:
            d[self.recon_key] = _resample(d[self.recon_key], self.recon_size, "trilinear")

        # Image onto the fine grid (the network input).
        d[self.image_key] = _resample(img, self.image_size, "trilinear")

        if self.task is not None:
            d["task"] = self.task
        return d


__all__ = ["ToFineGridd"]
