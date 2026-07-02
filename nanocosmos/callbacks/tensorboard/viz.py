"""Low-level image utilities for TensorBoard visualisation.

Contains the small, dependency-light helpers used by the unified-head
panel logger: central-slice extraction, per-image min-max normalisation,
HSV colour LUT, and integer-label → pastel-RGB mapping.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, reduce


def _resize_2d(
    img: torch.Tensor,
    size: Sequence[int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a ``[B, C, H, W]`` panel to ``size=(H', W')`` for display.

    Use ``mode="nearest"`` for discrete maps (instance-label RGB, masks) so
    label colours / boundaries are not blended; ``"bilinear"`` for continuous
    panels (EM image, reconstruction, sem / affinity probabilities).  A no-op
    when the panel is already at ``size``.
    """
    target: Tuple[int, int] = (int(size[0]), int(size[1]))
    if tuple(img.shape[-2:]) == target:
        return img
    if mode == "nearest":
        return F.interpolate(img.float(), size=target, mode="nearest")
    return F.interpolate(img.float(), size=target, mode="bilinear", align_corners=False)


def _to_2d(t: torch.Tensor, frac: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Extract a depth-slice from a 5-D tensor [B,C,D,H,W].

    By default the central slice (``D // 2``) is used.  When ``frac`` is given
    -- a per-sample tensor of *relative* depths in ``[0, 1]`` with shape ``[B]``
    -- each sample's slice is taken at ``round(frac * (D - 1))`` instead.

    ``frac`` is relative (not an absolute index) on purpose: the SAME value
    maps correctly onto tensors living on different grids -- e.g. the fine-grid
    image (deep) and the coarser native-grid label -- so all panels display the
    same physical depth.  Callers use this to show the most informative slice
    (e.g. the z with the most segmentation foreground) rather than a central
    slice that may be empty / fully eroded even when the 3-D target is not.

    Returns *t* unchanged if it is already 4-D [B,C,H,W].
    """
    if t.dim() != 5:
        return t
    D = t.shape[2]
    if frac is None:
        return t[:, :, D // 2]
    idx = (frac.to(t.device).clamp(0.0, 1.0) * max(D - 1, 0)).round().long()
    idx = idx.view(-1, 1, 1, 1, 1).expand(-1, t.shape[1], 1, t.shape[3], t.shape[4])
    return t.gather(2, idx).squeeze(2)


def _normalise(t: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalisation to [0, 1].

    Each image in the batch is normalised independently so that its
    minimum becomes 0 and its maximum becomes 1.

    Note:
        This is a **contrast stretch**, not a simple clip.  Applying
        it to the ground-truth image panel (``true/image``) makes the
        panel visually comparable across samples with different
        intensity distributions, but it also means that
        ``true/image`` can look brighter / more contrasted than the
        ``pred/raw`` panel, which is shown via a ``[-1, 1] -> [0, 1]``
        rescale of the linear reconstruction.  Keep that in mind when
        comparing reconstruction quality visually; the loss scalar
        (``loss/raw``) is computed on the unstretched signal.
    """
    flat = rearrange(t, "b ... -> b (...)")                        # [B, N]
    lo = reduce(flat, "b n -> b 1", "min")
    hi = reduce(flat, "b n -> b 1", "max")
    denom = (hi - lo).clamp(min=1e-5)
    normed = (flat - lo) / denom                                   # [B, N]
    return rearrange(normed, "b (c h w) -> b c h w",
                     c=t.shape[1], h=t.shape[2], w=t.shape[3])


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vectorised HSV→RGB.  All inputs/outputs in [0, 1], shape [N]."""
    h6 = h * 6.0
    sector = h6.long() % 6
    f = h6 - h6.floor()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    lut = [(v, t, p), (q, v, p), (p, v, t),
           (p, q, v), (t, p, v), (v, p, q)]
    r, g, b = torch.zeros_like(h), torch.zeros_like(h), torch.zeros_like(h)
    for i, (ri, gi, bi) in enumerate(lut):
        mask = sector == i
        r = torch.where(mask, ri, r)
        g = torch.where(mask, gi, g)
        b = torch.where(mask, bi, b)
    return torch.stack([r, g, b], dim=-1)


def _label_to_rgb(labels: torch.Tensor) -> torch.Tensor:
    """Map integer instance labels → pastel, deterministic RGB image.

    Background (0) is black.  Non-zero labels are coloured via a
    golden-ratio hue-spaced HSV palette with moderate saturation and
    high value for soft, PCA-like colours.

    Args:
        labels: [B, H, W] long tensor.

    Returns:
        [B, 3, H, W] float tensor in [0, 1].
    """
    B, H, W = labels.shape
    flat = rearrange(labels, "b h w -> (b h w)").long()
    n_labels = flat.max().item() + 1

    GOLDEN_RATIO = 0.618033988749895
    ids = torch.arange(n_labels, device=labels.device, dtype=torch.float32)
    hue = (ids * GOLDEN_RATIO) % 1.0

    gen = torch.Generator(device=labels.device).manual_seed(0)
    rand = torch.rand(n_labels, 2, device=labels.device, generator=gen)
    sat = 0.20 + 0.25 * rand[:, 0]                                # [0.20, 0.45]
    val = 0.75 + 0.25 * rand[:, 1]                                # [0.75, 1.00]

    palette = _hsv_to_rgb(hue, sat, val)                           # [n_labels, 3]
    palette[0] = 0.0                                               # background → black

    rgb = palette[flat]
    return rearrange(rgb, "(b h w) c -> b c h w", b=B, h=H, w=W)


__all__ = [
    "_hsv_to_rgb",
    "_label_to_rgb",
    "_normalise",
    "_resize_2d",
    "_to_2d",
]
