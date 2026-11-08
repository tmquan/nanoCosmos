"""Mean pooling that follows ``F.interpolate(..., mode="nearest")`` windows.

Degrade averages a 4 nm crop into slabs with it, and the joint loss averages
the 4 nm head onto the native grid with it.  Both use the windows a nearest
upsample from the coarse size back to the fine size fills, so a slab is
exactly the planes it is later repeated over.
"""

from __future__ import annotations

from typing import Sequence

import torch


def nearest_source(native: int, fine: int) -> torch.Tensor:
    """Native index each fine voxel copies under ``F.interpolate`` nearest."""
    dst = torch.arange(int(fine), dtype=torch.float32)
    scale = torch.tensor(int(native), dtype=torch.float32) / torch.tensor(int(fine), dtype=torch.float32)
    return torch.floor(dst * scale).to(torch.long).clamp(max=int(native) - 1)


def _nearest_mean_matrix(
    native: int, fine: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """``[native, fine]`` row-mean over the fine voxels each native voxel fills."""
    src = nearest_source(native, fine)
    m = torch.zeros(int(native), int(fine), dtype=torch.float32)
    m[src, torch.arange(int(fine))] = 1.0
    m = m / m.sum(dim=1, keepdim=True)
    return m.to(device=device, dtype=dtype)


def nearest_mean_pool(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Average a fine ``[..., D, H, W]`` volume onto the coarser ``shape``.

    Each coarse voxel is the mean of exactly the fine voxels that a nearest
    upsample back to the fine size fills from it, so the windows do not
    overlap and follow the slabs.  ``adaptive_avg_pool3d`` uses
    ``floor``/``ceil`` windows instead; at a fractional ratio (26 or 34 z
    sections, 171 xy pixels onto 256) each window also takes a voxel of the
    neighbouring section.
    """
    out = x
    for axis, n in zip((-3, -2, -1), shape):
        f = out.shape[axis]
        n = int(n)
        if n == f:
            continue
        if n < 1 or n > f:
            raise ValueError(f"native size {n} must lie in 1..{f} to pool a fine axis.")
        m = _nearest_mean_matrix(n, f, out.device, out.dtype)
        out = torch.movedim(torch.movedim(out, axis, -1) @ m.t(), -1, axis)
    return out


__all__ = ["nearest_mean_pool", "nearest_source"]
