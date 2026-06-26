"""
Realistic large-voxel degradation of small-voxel EM (the free DAPT supervisor).

The nanoCosmos reconstruction objective is *large-voxel -> small-voxel*
super-resolution: take a coarse (large-voxel) volume and synthesise the fine
(small-voxel) ultrastructure (see doc/RESOLUTION_LADDER.md).  On small-voxel
source data (FIB-25 8 nm, or the genuinely 4 nm COSEM / OpenOrganelle volumes)
we can manufacture that training pair **for free** -- no labels needed -- by
degrading the small-voxel patch into a realistic large-voxel acquisition and
keeping the original as the reconstruction target.  This is the self-supervised
DAPT signal that unlocks petascale unlabeled EM.

(Voxel-size convention: *small voxel size* = fine, e.g. 4 nm; *large voxel
size* = coarse, e.g. 30-40 nm.)

:class:`RandResolutionDegraded` maps a single small-voxel image patch
``[C, D, H, W]`` (on the fixed small-voxel network grid) to:

* ``image``        -- the **degraded** volume, back on the *same* small-voxel
  grid (so it is a drop-in network input), blurry / aliased in ``z``;
* ``recon_image``  -- a pristine copy of the input (the L1 reconstruction
  target).

The degradation composes the dominant real ssTEM / SBEM defects, each
independently probability-gated, so the reconstruction does not overfit
clean ``z``-decimation (the make-or-break sim-to-real concern):

1. **``z`` slab integration + decimation** -- ``adaptive_avg_pool3d`` to
   ``d ~= D / zf`` sections (a large physical voxel *averages* the small
   sub-voxels; this is the real large-voxel acquisition, not a point
   subsample).
2. **section misalignment / jitter** -- a small per-section in-plane
   translation via ``grid_sample`` (the registration error between serial
   sections).
3. **missing / duplicated sections** -- blank or replicate whole
   sections (acquisition dropouts; cf. :class:`RandMissingSliced`).
4. **section-dependent noise + contrast** -- per-section Gaussian noise
   and gamma jitter (staining / detector variation).
5. **``z`` up-sampling back to ``D``** -- trilinear, so the model sees the
   large-voxel content on the small-voxel canvas and its job is to restore
   the lost ``z`` detail.

Image-only: there is no label here (DAPT is unlabeled).  When used to
*synthesise* a large-voxel SFT pair from a labelled small-voxel volume, run it
on the image and pool the label separately -- but its primary use is the
label-free DAPT branch.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable


class RandResolutionDegraded(MapTransform, Randomizable):
    """Degrade a small-voxel patch into a realistic large-voxel acquisition.

    Args:
        keys: Image key(s) to degrade.  Default ``("image",)``.
        recon_key: Key under which the pristine small-voxel copy (the
            reconstruction target) is stored.  Default ``"recon_image"``.
        zf_range: ``(min, max)`` voxel-size factor; the sampled ``zf`` sets
            the number of retained sections ``d = round(D / zf)``.  E.g.
            ``(2, 10)`` spans 8 nm (FIB) to 40 nm (CREMI / MICrONS) sections
            on a 4 nm grid.  ``zf`` is sampled log-uniformly so scales are
            symmetric.
        prob: Probability of degrading at all (else passthrough, but the
            ``recon_key`` copy is still written so the recon target always
            exists).
        jitter_prob / max_jitter: Per-section in-plane translation
            (pixels, uniform in ``[-max_jitter, max_jitter]`` per section).
        missing_prob / max_missing: Missing/duplicated-section defect on
            the coarse stack (reuses the zero / replicate fill convention).
        missing_fill: ``"zero"`` | ``"replicate"`` for dropped sections.
        noise_prob / noise_std: Per-section additive Gaussian noise (std
            relative to the ``[0, 1]`` intensity range).
        contrast_prob / contrast_gamma: Per-section gamma jitter
            ``gamma ~ U(contrast_gamma)``.
        z_sections_key: Optional key to record the coarse section count
            ``d`` (handy when re-using this to synthesise large-voxel pairs).
        up_mode: Interpolation for the final ``z`` up-sample back to ``D``
            (``"trilinear"`` default; ``"nearest"`` for a blockier look).
    """

    def __init__(
        self,
        keys: KeysCollection = ("image",),
        recon_key: str = "recon_image",
        zf_range: Tuple[float, float] = (2.0, 10.0),
        prob: float = 1.0,
        jitter_prob: float = 0.5,
        max_jitter: float = 2.0,
        missing_prob: float = 0.3,
        max_missing: int = 2,
        missing_fill: str = "zero",
        noise_prob: float = 0.5,
        noise_std: float = 0.05,
        contrast_prob: float = 0.5,
        contrast_gamma: Tuple[float, float] = (0.8, 1.25),
        z_sections_key: Optional[str] = None,
        up_mode: str = "trilinear",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.recon_key = str(recon_key)
        self.zf_range = (float(zf_range[0]), float(zf_range[1]))
        self.prob = float(prob)
        self.jitter_prob = float(jitter_prob)
        self.max_jitter = float(max_jitter)
        self.missing_prob = float(missing_prob)
        self.max_missing = max(1, int(max_missing))
        if missing_fill not in ("zero", "replicate"):
            raise ValueError(
                f"missing_fill must be 'zero' or 'replicate'; got {missing_fill!r}."
            )
        self.missing_fill = missing_fill
        self.noise_prob = float(noise_prob)
        self.noise_std = float(noise_std)
        self.contrast_prob = float(contrast_prob)
        self.contrast_gamma = (float(contrast_gamma[0]), float(contrast_gamma[1]))
        self.z_sections_key = z_sections_key
        if up_mode not in ("trilinear", "nearest"):
            raise ValueError(f"up_mode must be 'trilinear' or 'nearest'; got {up_mode!r}.")
        self.up_mode = up_mode

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_zf(self) -> float:
        lo, hi = self.zf_range
        return float(np.exp(self.R.uniform(np.log(lo), np.log(hi))))

    def _jitter_sections(self, coarse: torch.Tensor) -> torch.Tensor:
        """Translate each coarse section in-plane by a small random shift.

        ``coarse`` is ``[C, d, H, W]``.  Each of the ``d`` sections is
        sampled with its own ``(dy, dx)`` pixel offset via ``grid_sample``
        (reflection padding), modelling serial-section registration error.
        """
        c, d, h, w = coarse.shape
        # Treat each section as a batch item: [d, C, H, W].
        x = coarse.permute(1, 0, 2, 3).contiguous()
        shifts = self.R.uniform(-self.max_jitter, self.max_jitter, size=(d, 2))
        # Normalised translation (full extent is [-1, 1]); affine_grid theta.
        theta = torch.zeros(d, 2, 3, dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = torch.as_tensor(2.0 * shifts[:, 1] / max(w - 1, 1),
                                          dtype=x.dtype, device=x.device)
        theta[:, 1, 2] = torch.as_tensor(2.0 * shifts[:, 0] / max(h - 1, 1),
                                          dtype=x.dtype, device=x.device)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="reflection",
            align_corners=False,
        )
        return x.permute(1, 0, 2, 3).contiguous()

    def _drop_sections(self, coarse: torch.Tensor) -> torch.Tensor:
        """Blank or replicate up to ``max_missing`` coarse sections."""
        c, d, h, w = coarse.shape
        k = min(int(self.R.randint(1, self.max_missing + 1)), d)
        drop = sorted(int(i) for i in self.R.choice(d, size=k, replace=False))
        surviving = [i for i in range(d) if i not in drop]
        out = coarse.clone()
        for z in drop:
            if self.missing_fill == "replicate" and surviving:
                src = min(surviving, key=lambda s: abs(s - z))
                out[:, z] = coarse[:, src]
            else:
                out[:, z] = 0.0
        return out

    def _section_noise(self, coarse: torch.Tensor) -> torch.Tensor:
        c, d, h, w = coarse.shape
        # Per-section std in [0, noise_std]; broadcast over the plane.
        std = torch.as_tensor(
            self.R.uniform(0.0, self.noise_std, size=(1, d, 1, 1)),
            dtype=coarse.dtype, device=coarse.device,
        )
        return coarse + torch.randn_like(coarse) * std

    def _section_contrast(self, coarse: torch.Tensor) -> torch.Tensor:
        c, d, h, w = coarse.shape
        lo, hi = self.contrast_gamma
        gamma = torch.as_tensor(
            self.R.uniform(lo, hi, size=(1, d, 1, 1)),
            dtype=coarse.dtype, device=coarse.device,
        )
        # Gamma is defined on [0, 1]; clamp to stay in range.
        return coarse.clamp(0.0, 1.0).pow(gamma)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        first = next(iter(self.key_iterator(d)), None)
        if first is None:
            return d

        for key in self.key_iterator(d):
            arr = d[key]
            is_meta = hasattr(arr, "meta")
            meta = arr.meta if is_meta else None
            applied = arr.applied_operations if is_meta else None
            t = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(np.asarray(arr))
            t = t.detach().float()
            if t.ndim != 4:
                # Not a [C, D, H, W] volume; nothing to coarsen.
                d.setdefault(self.recon_key, arr)
                continue

            c, D, H, W = t.shape
            clean = t.clone()

            do = self.R.random() < self.prob and D > 1
            if do:
                zf = self._sample_zf()
                d_coarse = max(1, int(round(D / zf)))
                coarse = F.adaptive_avg_pool3d(
                    clean.unsqueeze(0), (d_coarse, H, W),
                ).squeeze(0)

                if self.max_jitter > 0 and self.R.random() < self.jitter_prob:
                    coarse = self._jitter_sections(coarse)
                if d_coarse > 1 and self.R.random() < self.missing_prob:
                    coarse = self._drop_sections(coarse)
                if self.noise_std > 0 and self.R.random() < self.noise_prob:
                    coarse = self._section_noise(coarse)
                if self.R.random() < self.contrast_prob:
                    coarse = self._section_contrast(coarse)

                up_kw = (
                    {"mode": "nearest"} if self.up_mode == "nearest"
                    else {"mode": "trilinear", "align_corners": False}
                )
                degraded = F.interpolate(
                    coarse.unsqueeze(0), size=(D, H, W), **up_kw,
                ).squeeze(0)
                if self.z_sections_key is not None:
                    d[self.z_sections_key] = int(d_coarse)
            else:
                degraded = clean.clone()
                if self.z_sections_key is not None:
                    d[self.z_sections_key] = int(D)

            degraded = degraded.to(arr.dtype if isinstance(arr, torch.Tensor) else torch.float32)
            if is_meta:
                from monai.data import MetaTensor

                degraded = MetaTensor(degraded, meta=meta, applied_operations=applied)
                clean = MetaTensor(clean, meta=meta, applied_operations=applied)
            d[key] = degraded
            d[self.recon_key] = clean

        return d


__all__ = ["RandResolutionDegraded"]
