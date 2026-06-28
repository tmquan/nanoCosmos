"""
Joint super-resolution-reconstruction + segmentation loss (the nanoCosmos recipe).

The model is a **voxel-size super-resolver** (see doc/RESOLUTION_LADDER.md):
one shared backbone emits a single head on the **fixed small-voxel network
grid** (the finest voxel size, ``D == H == W``, e.g. ``160`` at 4 nm)::

    head : [B, N_AFF + 2, D, H, W]      # aff (logits) | sem (logit) | raw (linear)

(Voxel-size convention used throughout: *small voxel size* = fine / high
detail, e.g. 4 nm; *large voxel size* = coarse, e.g. 30-40 nm.)

The ``raw`` channel is the **reconstructed small-voxel EM** (not the input
reconstruction it was in the plain affinity recipe); ``aff`` / ``sem`` are
the **small-voxel segmentation**.  *Every supervision term pools the small-
voxel prediction down to wherever its ground truth lives* -- the pool factor
is read straight from the GT tensor's spatial shape, so it can never disagree
with the target grid (factor 1 == the smallest-voxel rung).

Two branches, routed per **batch** by ``targets["task"]`` (batches must be
task-homogeneous -- see the round-robin multi-task sampler in the datamodule;
this also gives per-branch batch-size / weighting control).  The roles
**stack** by dataset (a labeled small-voxel rung like FIB-25 is in *both*):

``ssl``  -- self-supervised reconstruction, gated by *voxel size*:
    a degraded (large-voxel) input is reconstructed back to the clean small-
    voxel EM, ``L1(pool(raw -> recon_image grid), recon_image)``.  Sources:
    the smallest-voxel rung (COSEM 4 nm) and any other fine-enough volume,
    **including UNLABELED FIB** (the vast unproofread FIB-25 volume -- domain-
    matched neuropil SSL).  No segmentation supervision.

``sft``  -- segmentation, gated by *labels*: any labeled rung.  The small-
    voxel head is pooled to the **label grid** (``labels.shape[-3:]``) via
    :func:`adaptive_avg_pool3d` -- the physically-faithful "a large voxel
    averages the small sub-voxels" downsample on every axis (z *and* xy), the
    anti-aliased counterpart of a single ``grid_sample`` tap, accepting any
    (fractional) factor -- then :class:`~nanocosmos.losses.AffinityFGLoss`.
    **Plus a data-consistency term on the ``raw`` head**: the 32-ch head
    always carries ``raw``, so on sft batches we also pool the predicted
    small-voxel ``raw`` down to the ORIGINAL large-voxel EM and match it
    (``recon_image`` here is that large-voxel EM).  This keeps the raw head
    trained on sft and enforces that the super-resolved reconstruction agrees
    with the measured data when downsampled ("do no harm").  Optional
    (skipped if no ``recon_image``).

All branches **backprop jointly** into the shared backbone (no detach): the
segmentation gradient shapes the reconstruction and vice-versa.

Output dict (keys gated by the active branch / configured weights)::

    loss              # global weighted total for this (homogeneous) batch
    loss/recon        # reconstruction (ssl: main SR; sft: raw data-consistency)
    loss/aff          # affinity segmentation (sft)
    loss/sem          # foreground segmentation (sft)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanocosmos.losses.affinity import AffinityFGLoss
from nanocosmos.losses._common import regression_loss_fn

# Canonical branch names.  A batch carries exactly one of these in
# ``targets["task"]`` (the datamodule's sampler yields task-homogeneous
# batches).  See doc/RESOLUTION_LADDER.md:
#   * SSL -- self-supervised reconstruction of the clean small-voxel EM from a
#     degraded (large-voxel) input (gated by *voxel size*: any rung fine
#     enough; labels optional).  Owns the ``raw`` reconstruction term.
#   * SFT  -- segmentation on any *labeled* rung (gated by *labels*): pool the
#     small-voxel head down to the label grid, then the affinity + sem loss.  A
#     rung with labels does SFT regardless of where it sits on the ladder; the
#     pool factor is derived from the labels (1 = the smallest-voxel case).
SSL = "ssl"
SFT = "sft"
_TASKS = (SSL, SFT)


def _as_channeled(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Ensure a singleton channel axis so ``x`` has ``ndim`` dims.

    Accepts a recon target as either ``[B, D, H, W]`` or
    ``[B, 1, D, H, W]`` and returns the ``[B, 1, D, H, W]`` form.
    """
    if x.dim() == ndim - 1:
        return x.unsqueeze(1)
    return x


class Joint3DReconSegLoss(nn.Module):
    """Two-branch joint EM reconstruction + segmentation loss.

    Args:
        seg: Either an already-built :class:`AffinityFGLoss` or a kwargs
            mapping forwarded to it.  ``weight_raw`` on the inner loss is
            **forced to 0** -- the EM reconstruction term is owned by this
            module (so it can be applied / skipped per branch and target an
            arbitrary EM grid, not the model input).  Everything else
            (offsets, ``weight_aff`` / ``weight_sem`` and all the rebalancing
            knobs) is passed straight through.
        recon_loss: Regression family for the reconstruction term
            (``l1`` / ``mse`` / ``smooth_l1``; default ``l1``).
        weight_rec: Scalar weight on the reconstruction term.
        weight_seg: Scalar weight on the segmentation term.
        weight_ssl / weight_sft: Per-branch multipliers on that branch's
            *total* (up/down-weight a whole branch independently of how often
            its batches are drawn).
        recon_image_key: Batch key holding the clean EM reconstruction target
            (default ``"recon_image"``).  On ``ssl`` it is the clean small-
            voxel EM (required); on ``sft`` it is the original large-voxel EM
            for the raw data-consistency term (optional).
        task_key: Batch key holding the branch name (default ``"task"``).

    The pool factor is **derived from the ground-truth shape** -- the small-
    voxel head is pooled to ``labels.shape[-3:]`` (and ``raw`` to
    ``recon_image.shape[-3:]``) -- so no explicit voxel-count key is needed and
    the pool can never disagree with the target grid (``pool factor == 1`` when
    the GT is already on the small-voxel grid).
    """

    def __init__(
        self,
        seg: Union[AffinityFGLoss, Mapping[str, Any], None] = None,
        *,
        recon_loss: str = "l1",
        weight_rec: float = 1.0,
        weight_seg: float = 1.0,
        weight_ssl: float = 1.0,
        weight_sft: float = 1.0,
        recon_image_key: str = "recon_image",
        task_key: str = "task",
    ) -> None:
        super().__init__()

        if isinstance(seg, AffinityFGLoss):
            self.seg = seg
        else:
            seg_kw = dict(seg or {})
            # The reconstruction term is owned here, not by the inner affinity
            # loss (whose ``raw`` term reconstructs the *input*).
            seg_kw["weight_raw"] = 0.0
            self.seg = AffinityFGLoss(**seg_kw)

        self._recon_fn = regression_loss_fn(recon_loss)
        self.recon_loss = recon_loss
        self.weight_rec = float(weight_rec)
        self.weight_seg = float(weight_seg)
        self.weight_ssl = float(weight_ssl)
        self.weight_sft = float(weight_sft)

        self.recon_image_key = str(recon_image_key)
        self.task_key = str(task_key)

        # Delegate the canonical channel layout so every consumer
        # (Lightning module, image logger, Mutex Watershed) keys off this
        # loss exactly as it does off ``AffinityFGLoss``.
        self.offsets = self.seg.offsets
        self.n_pull = self.seg.n_pull
        self.n_aff = self.seg.n_aff
        self.aff_slice = self.seg.aff_slice
        self.sem_slice = self.seg.sem_slice
        self.raw_slice = self.seg.raw_slice
        self.head_channels = self.seg.head_channels
        self.num_channels = self.seg.num_channels

    # ------------------------------------------------------------------
    # Target / key helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_targets(
        self,
        labels: torch.Tensor,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build the cached affinity target at the **supervised grid**.

        Delegates to :meth:`AffinityFGLoss.build_targets`.  ``labels`` is on
        the rung's native grid (small-voxel for FIB/COSEM, large-voxel for
        SNEMI3D / CREMI / MICrONS); the head is pooled to match in
        :meth:`forward`, so the target is always built from the labels as-is.
        """
        return self.seg.build_targets(labels, batch)

    def canonical_loss_keys(self) -> List[str]:
        """Deterministic loss-dict keys for the cross-rank eval reduction.

        Returns the union over branches (gated only by configured weights,
        never by batch content) so the key set is identical on every
        DDP/FSDP rank regardless of which branch a rank happens to draw.
        """
        keys: List[str] = ["loss"]
        if self.weight_rec > 0:
            keys.append("loss/recon")
        if self.weight_seg > 0 and self.seg.weight_aff > 0:
            keys.append("loss/aff")
        if self.weight_seg > 0 and self.seg.weight_sem > 0:
            keys.append("loss/sem")
        return keys

    def _resolve_task(self, task: Any) -> str:
        """Normalise ``targets['task']`` to a single branch name.

        Accepts a plain string or a per-sample sequence/tensor of strings;
        a per-sample form must be homogeneous (the multi-task sampler
        guarantees this).  Anything mixed is a configuration error.
        """
        if isinstance(task, str):
            name = task
        elif isinstance(task, (list, tuple)):
            uniq = set(map(str, task))
            if len(uniq) != 1:
                raise ValueError(
                    f"Joint3DReconSegLoss expects task-homogeneous batches; "
                    f"got mixed tasks {sorted(uniq)}.  Use the round-robin "
                    f"multi-task sampler so each batch is a single branch."
                )
            name = next(iter(uniq))
        else:
            raise TypeError(
                f"targets['{self.task_key}'] must be a str or a homogeneous "
                f"sequence of str; got {type(task).__name__}."
            )
        if name not in _TASKS:
            raise ValueError(
                f"Unknown task {name!r}; expected one of {_TASKS}."
            )
        return name

    # ------------------------------------------------------------------
    # Branch losses
    # ------------------------------------------------------------------

    def _recon_term(
        self, head: torch.Tensor, recon_image: torch.Tensor,
    ) -> torch.Tensor:
        """``L1`` (or configured) on the ``raw`` channel vs the clean EM.

        The small-voxel ``raw`` channel is pooled to the reconstruction
        target's own grid (``recon_image.shape[-3:]``) before the regression,
        so the term works whether the target sits on the small-voxel grid
        (COSEM 4 nm; pool is a no-op) or a larger-voxel native grid (FIB 8 nm
        SSL, or the original large-voxel EM in the sft data-consistency term).
        """
        target = _as_channeled(recon_image, head.dim()).to(torch.float32)
        raw = self._pool_to(head[:, self.raw_slice], target.shape[-3:])
        return self._recon_fn(raw.float(), target.detach())

    def _seg_terms(
        self,
        head: torch.Tensor,
        labels: torch.Tensor,
        cached: Optional[Dict[str, torch.Tensor]],
        sem_label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the inner affinity+sem loss on ``head`` at the label grid.

        ``sem_label`` (the boundary-eroded foreground from the datamodule's
        ``boundary_target: semantic`` mode) supervises the sem head only; the
        affinity target always uses the pristine instance ``labels``.
        """
        sub_targets: Dict[str, Any] = {"labels": labels}
        if cached is not None:
            sub_targets["_cached_targets"] = cached
        if sem_label is not None:
            sub_targets["sem_label"] = sem_label
        out = self.seg(head, sub_targets)
        # Drop the inner total (``loss``); this module rebuilds the total.
        return {k: v for k, v in out.items() if k != "loss"}

    @staticmethod
    def _pool_to(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """Average-pool a small-voxel volume down to the native ``(d, h, w)``.

        ``adaptive_avg_pool3d`` integrates over each (possibly fractional)
        window on **every** axis -- the faithful "a large voxel averages the
        small sub-voxels" downsample, and the anti-aliased counterpart of a
        single ``grid_sample`` tap.  Pools on all three axes so larger-voxel-
        *xy* rungs (MICrONS / FIB at 8 nm) pool in-plane too, not just in z.
        A no-op when ``x`` is already at ``shape`` (the smallest-voxel rung /
        pool factor 1).
        """
        target = tuple(int(s) for s in shape)
        if tuple(x.shape[-3:]) == target:
            return x
        return F.adaptive_avg_pool3d(x, target)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        head: torch.Tensor,
        targets: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Route the (task-homogeneous) batch to its branch and aggregate.

        Args:
            head: ``[B, N_AFF + 2, D, H, W]`` small-voxel head (aff/sem
                logits, raw linear) on the fixed small-voxel network grid.
            targets: Must carry ``task``.  ``ssl`` carries ``recon_image``
                (the clean EM reconstruction target, on the small-voxel grid
                for COSEM or a larger-voxel native grid for FIB SSL).  ``sft``
                carries ``labels`` at the native grid -- the pool factor is
                read from the label shape, so no voxel-count key is needed --
                and optionally ``recon_image`` (the original large-voxel EM)
                for the ``raw`` data-consistency term.  ``_cached_targets``
                (from :meth:`build_targets`) is used when present.

        Returns:
            ``ssl`` -> ``{"loss", "loss/recon"}``;
            ``sft``  -> ``{"loss", "loss/aff", "loss/sem", ["loss/recon"]}``.
        """
        if head.shape[1] != self.head_channels:
            raise ValueError(
                f"Joint3DReconSegLoss expects a {self.head_channels}-channel "
                f"head (N_AFF + 2); got {head.shape[1]}."
            )
        task = self._resolve_task(targets[self.task_key])
        cached: Optional[Dict[str, torch.Tensor]] = targets.get("_cached_targets")
        out: Dict[str, torch.Tensor] = {}
        total = head.new_zeros(())

        recon_image = targets.get(self.recon_image_key)

        if task == SSL:
            # ---- self-supervised reconstruction of the clean EM ----
            # The whole branch: reconstruct the clean (small-voxel) EM from a
            # degraded (large-voxel) input.  ``recon_image`` is REQUIRED (it is
            # the only supervision on this label-free branch).
            if recon_image is None:
                raise KeyError(
                    f"Joint3DReconSegLoss[ssl] needs targets"
                    f"['{self.recon_image_key}'] (the clean EM reconstruction "
                    f"target)."
                )
            if self.weight_rec > 0:
                l_recon = self._recon_term(head, recon_image)
                out["loss/recon"] = l_recon
                total = total + self.weight_rec * l_recon
            branch_w = self.weight_ssl
        else:
            # ---- segmentation on a labeled rung ----
            # Pool the small-voxel head down to the LABEL grid (factor derived
            # from the label shape; a no-op when labels are already small-voxel).
            labels = targets["labels"]
            if self.weight_seg > 0:
                seg_head = self._pool_to(head, labels.shape[-3:])
                seg = self._seg_terms(seg_head, labels, cached, targets.get("sem_label"))
                for k, v in seg.items():
                    out[k] = v
                    total = total + self.weight_seg * v
            # Data-consistency on the ``raw`` head (the 32-ch head always
            # carries it).  Without this the raw channel is UNTRAINED on sft
            # batches.  Pool the predicted small-voxel ``raw`` down to the
            # ORIGINAL large-voxel EM and match it: the super-resolved
            # reconstruction must agree with what was actually measured when
            # downsampled ("do no harm" / SR data-consistency).  ``recon_image``
            # here is the original large-voxel EM; optional (skipped if the
            # datamodule omits it).
            if recon_image is not None and self.weight_rec > 0:
                l_recon = self._recon_term(head, recon_image)
                out["loss/recon"] = l_recon
                total = total + self.weight_rec * l_recon
            branch_w = self.weight_sft

        out["loss"] = branch_w * total
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(seg={self.seg!r}, "
            f"recon_loss={self.recon_loss}, weight_rec={self.weight_rec}, "
            f"weight_seg={self.weight_seg}, weight_ssl={self.weight_ssl}, "
            f"weight_sft={self.weight_sft})"
        )


__all__ = ["Joint3DReconSegLoss", "SSL", "SFT"]
