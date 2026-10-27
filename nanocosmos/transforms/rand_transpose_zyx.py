"""Random permutation of the Z, Y, and X axes of a cubic-FOV crop.

Sampling is uniform over the 6 axis orders, identity included.  The finest
native axis (smallest voxel size -- the highest-resolution dimension) therefore
lands on network z, y, or x equally often, instead of staying in the
orientation stored on disk (xy for anisotropic serial-section EM).

The fine field of view must be cubic (``patch_size`` and ``pixel_size`` both
cubic; :class:`~nanocosmos.datamodules.joint3d.Joint3DDataModule` raises
``ValueError`` otherwise), so a permutation is a pure reorientation: spatial
arrays and the native ``(z, y, x)`` voxel-size vector move together, and a
text prompt of the form ``{imaging} · z{Z} y{Y} x{X} nm · {tail}`` is
rewritten to the permuted sizes.

Run this **after** :class:`~nanocosmos.transforms.fine_grid.ToFineGridd` and
:class:`~nanocosmos.transforms.degrade.RandResolutionDegraded`.  The joint
datamodule attaches it to train crops only.  The network image is already on
the cubic fine grid (its shape is unchanged); the recon target, which may
still be anisotropic, is permuted with it so the coarse axis and the prompt
stay aligned.  Section degradation has already been applied on the native
section axis, and that axis moves with the permutation.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

# Output axis i reads input axis ``perm[i]`` (``np.transpose`` / ``Tensor.permute``
# convention on the spatial axes).
_ZYX_PERMS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

_PROMPT_RES = re.compile(
    r"^(?P<prefix>.*?) · z\d+ y\d+ x\d+ nm · (?P<tail>.*)$"
)


def _rewrite_prompt(
    prompt: str,
    perm: Tuple[int, int, int],
    native_resolution: Sequence[float],
) -> Optional[str]:
    """Return ``prompt`` with z/y/x nm rewritten for ``perm``, or None if the
    string is not a joint condition prompt."""
    match = _PROMPT_RES.match(prompt.strip())
    if match is None:
        return None
    vals = [int(native_resolution[i]) for i in perm]
    return (
        f"{match.group('prefix')} · z{vals[0]} y{vals[1]} x{vals[2]} nm · "
        f"{match.group('tail')}"
    )


def _permute_spatial(val, perm: Tuple[int, int, int]):
    """Permute the last three axes of a tensor or ndarray.  ``perm[i]`` is the
    old spatial axis that becomes new spatial axis ``i``."""
    if isinstance(val, torch.Tensor):
        if val.ndim < 3:
            return val
        nd = val.ndim
        axes = tuple(range(nd - 3)) + tuple(nd - 3 + p for p in perm)
        return val.permute(*axes).contiguous()
    if isinstance(val, np.ndarray):
        if val.ndim < 3:
            return val
        nd = val.ndim
        axes = tuple(range(nd - 3)) + tuple(nd - 3 + p for p in perm)
        return np.ascontiguousarray(np.transpose(val, axes))
    return val


class RandTransposeZYXd(MapTransform, Randomizable):
    """Randomly permute Z, Y, and X so the finest native axis is augmented.

    Args:
        keys: Spatial arrays to permute together (image, recon target, label).
        native_resolution: Voxel size ``(z, y, x)`` nm of those arrays **before**
            permutation.  Used to rewrite ``data["prompt"]`` when present.
        allow_missing_keys: Skip keys that are absent from the sample.
    """

    def __init__(
        self,
        keys: KeysCollection,
        native_resolution: Sequence[float],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if len(native_resolution) < 3:
            raise ValueError(
                "native_resolution must be (z, y, x) nm; "
                f"got {list(native_resolution)!r}."
            )
        self.native_resolution = tuple(float(r) for r in native_resolution[:3])
        self._perm: Tuple[int, int, int] = (0, 1, 2)

    def randomize(self, data=None) -> None:
        self._perm = _ZYX_PERMS[int(self.R.randint(0, len(_ZYX_PERMS)))]

    def __call__(self, data: Dict) -> Dict:
        self.randomize()
        d = dict(data)
        perm = self._perm
        if perm != (0, 1, 2):
            for key in self.key_iterator(d):
                d[key] = _permute_spatial(d[key], perm)
            prompt = d.get("prompt")
            if isinstance(prompt, str):
                rewritten = _rewrite_prompt(prompt, perm, self.native_resolution)
                if rewritten is None:
                    raise ValueError(
                        "rand_transpose_zyx could not rewrite the condition "
                        f"prompt {prompt!r} for axis perm {perm}."
                    )
                d["prompt"] = rewritten
        return d


__all__ = ["RandTransposeZYXd"]
