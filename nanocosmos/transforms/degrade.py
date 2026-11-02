"""
Realistic large-voxel degradation of small-voxel EM (the free SSL supervisor).

The nanoCosmos reconstruction objective is *large-voxel -> small-voxel*
super-resolution: take a coarse (large-voxel) volume and synthesise the fine
(small-voxel) ultrastructure (see doc/RESOLUTION_LADDER.md).  On small-voxel
source data (FIB-25 8 nm, or the genuinely 4 nm COSEM / OpenOrganelle volumes)
we can manufacture that training pair **for free** -- no labels needed -- by
degrading the small-voxel patch into a realistic large-voxel acquisition and
keeping the original as the reconstruction target.  This is the self-supervised
SSL signal that unlocks petascale unlabeled EM.

(Voxel-size convention: *small voxel size* = fine, e.g. 4 nm; *large voxel
size* = coarse, e.g. 30-40 nm.)

:class:`RandResolutionDegraded` maps one image patch ``[C, D, H, W]`` at the
spatial size it is given.  In the joint pipeline that size is the
**native-resolution crop**, before
:class:`~nanocosmos.transforms.fine_grid.ToFineGridd` nearest-upsamples
``image`` onto the 4 nm grid:

* ``recon_image``  -- the augmented clean crop (the L1 target): per-section
  jitter and gamma contrast, still at the input resolution.
* ``image``        -- that same crop after degrade and input-only noise.
  With ``downsample`` it is shorter in ``z`` (nearest-neighbour decimation).
  The 4 nm upsample is the next transform, not this one.

The order after the read is:

1. **Augment clean** -- per-section jitter, then gamma contrast.  This
   snapshot is the reconstruction target.
2. **Nearest ``z`` downsample** (4 nm only) -- ``d = round(D / zf)`` sections
   of the augmented clean, each a nearest sample, not an average.  The
   datamodule turns this on only for a 4 nm crop.
3. **Augment noise** -- missing / duplicated sections, then per-section
   Gaussian noise.  These touch ``image`` only.  A coarser crop skips the
   downsample and still gets this step, at its own depth.

Image-only: there is no label here (SSL is unlabeled).  When used to
*synthesise* a large-voxel SFT pair from a labelled small-voxel volume, run it
on the image and pool the label separately -- but its primary use is the
label-free SSL branch.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable

from nanocosmos.transforms._sampling import log_uniform


class RandResolutionDegraded(MapTransform, Randomizable):
    """Degrade a small-voxel patch into a realistic large-voxel acquisition.

    Args:
        keys: Image key(s) to degrade.  Default ``("image",)``.
        recon_key: Key under which the augmented clean crop (jitter and
            contrast, the reconstruction target) is stored.  Default
            ``"recon_image"``.
        zf_range: ``(min, max)`` factor on the input z size, used only when
            ``downsample`` is set.  ``d = round(D / zf)``.  On a 4 nm crop,
            ``(2, 10)`` spans 8 nm to 40 nm sections.  Sampled log-uniformly.
        downsample: Nearest-downsample ``z`` of the augmented clean, before
            missing sections and noise.  The datamodule sets this only for a
            4 nm crop.  Off: those input defects still run, at the crop's
            own depth.
        prob: Probability of degrading at all (else passthrough, but the
            ``recon_key`` copy is still written so the recon target always
            exists).
        jitter_prob / max_jitter: Per-section in-plane translation
            (pixels, uniform in ``[-max_jitter, max_jitter]`` per section).
        missing_prob / max_missing: Missing/duplicated-section defect.
        missing_fill: ``"zero"`` | ``"replicate"`` for dropped sections.
        noise_prob / noise_std: Per-section additive Gaussian noise (std
            relative to the ``[0, 1]`` intensity range).
        contrast_prob / contrast_gamma: Per-section gamma jitter
            ``gamma ~ U(contrast_gamma)``.
        z_sections_key: Optional key recording the depth after downsample
            (the input depth when downsample is off).
    """

    def __init__(
        self,
        keys: KeysCollection = ("image",),
        recon_key: str = "recon_image",
        zf_range: Tuple[float, float] = (2.0, 10.0),
        downsample: bool = True,
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
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.recon_key = str(recon_key)
        self.zf_range = (float(zf_range[0]), float(zf_range[1]))
        self.downsample = bool(downsample)
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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_zf(self) -> float:
        lo, hi = self.zf_range
        return log_uniform(self.R, lo, hi)

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
        """Blank or replicate up to ``max_missing`` coarse sections.

        Deliberately kept separate from :class:`RandMissingSliced` (in
        ``missing_slice.py``): that transform applies the same missing-section
        defect to a full-resolution ``[C, D, H, W]`` image at augmentation
        time, whereas this operates on the coarse decimated stack inside the
        SSL degrade model, so the two are not merged into one helper.
        """
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
                # Augment clean first.  Jitter and contrast are the values
                # the loss compares to; degrade and noise are derived from
                # this snapshot and do not write back into it.
                if self.max_jitter > 0 and self.R.random() < self.jitter_prob:
                    clean = self._jitter_sections(clean)
                if self.R.random() < self.contrast_prob:
                    clean = self._section_contrast(clean)

                coarse = clean
                d_coarse = D
                if self.downsample:
                    zf = self._sample_zf()
                    d_coarse = max(1, int(round(D / zf)))
                    if d_coarse != D:
                        coarse = F.interpolate(
                            clean.unsqueeze(0), size=(d_coarse, H, W),
                            mode="nearest",
                        ).squeeze(0)

                if d_coarse > 1 and self.R.random() < self.missing_prob:
                    coarse = self._drop_sections(coarse)
                if self.noise_std > 0 and self.R.random() < self.noise_prob:
                    coarse = self._section_noise(coarse)

                degraded = coarse
                if self.z_sections_key is not None:
                    d[self.z_sections_key] = int(coarse.shape[1])
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
