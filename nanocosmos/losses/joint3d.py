"""
Joint super-resolution-reconstruction + segmentation loss (the nanoCosmos recipe).

The model is a **voxel-size super-resolver** (see doc/RESOLUTION_LADDER.md):
one shared backbone emits a single head on the fixed 4 nm network grid.
The patch shape is the configured ``patch_size`` (pretrain
``[400, 400, 400]``, joint 4B ``[400, 256, 256]``, 16B ``[200, 256, 256]``)::

    head : [B, n_aff + 2, D, H, W]      # aff (logits) | sem (logit) | raw (linear)

(Voxel-size convention used throughout: *small voxel size* = fine / high
detail, e.g. 4 nm; *large voxel size* = coarse, e.g. 30-40 nm.)

The ``raw`` channel is the reconstructed small-voxel EM.  ``aff`` / ``sem``
are the small-voxel segmentation.  Every supervision term pools the
small-voxel prediction down to the ground-truth tensor's spatial shape, so
the pool factor follows that grid (factor 1 when the target is already on
the small-voxel grid).

Two branches, routed per **batch** by ``targets["task"]`` (batches must be
task-homogeneous -- the datamodule's round-robin sampler, which also sets
per-branch batch size).  A labeled rung is drawn in both branches; FIB-25
(8 nm) is one such rung.

``ssl``  -- self-supervised reconstruction, gated by voxel size:
    a degraded input is matched to the clean EM,
    ``L1(pool(raw -> recon_image grid), recon_image)``.  ``recon_image``
    sits on the coarser of the native voxel size and the 4 nm grid (COSEM
    ~4 nm lands on the fine grid; unlabeled FIB-25 and other 8 nm neuropil
    stay on their native grid).  No segmentation supervision.

``sft``  -- segmentation, gated by labels: any labeled rung.  The
    small-voxel head is pooled to the **label grid** (``labels.shape[-3:]``)
    with :func:`adaptive_avg_pool3d` (each output voxel averages the small
    voxels in its window, on z and on xy, including fractional factors),
    then :class:`~nanocosmos.losses.AffinityFGLoss`.  The head is
    ``n_aff + 2`` channels (30 affinity offsets + sem + raw = 32 in the
    training configs) and always carries ``raw``.  On sft the predicted
    ``raw`` is also pooled to ``recon_image``: the clean native EM captured
    before the input is resampled onto the 4 nm grid, then placed on the
    coarser of native and fine.  That term is skipped when ``recon_image``
    is absent.

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
#   * SSL -- self-supervised reconstruction of the clean EM (on the coarser
#     of native and the 4 nm grid) from a degraded input (gated by voxel
#     size: any rung fine enough; labels optional).  Owns the ``raw`` term.
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

    Training configs use a 32-channel head (30 affinity offsets + sem + raw)
    and ``recon_loss="l1"``.  Pretrain
    (``configs/nanocosmos-4B-pretrain.yaml``) sets ``weight_rec=1``,
    ``weight_ssl=5``, ``weight_seg=0``, ``weight_sft=0``.  The constructor
    defaults are the signature values.

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
        weight_dif: Weight of the rectified-flow velocity MSE relative to
            the reconstruction L1, before the branch multiplier. ``1`` adds
            the flow term at the same scale as ``weight_rec * L1``. The MSE
            itself is computed by the wrapper (``proj_out``); this class only
            stores the weight so every rank agrees on the logged key.
        diffusion_steps: Euler steps for the ``pred/Euler`` reconstruction panel.
            Training still draws one random ``t``; this is the comparison
            integral only.
        recon_image_key: Batch key holding the clean EM reconstruction target
            (default ``"recon_image"``).  On ``ssl`` it is required and lives
            on the coarser of the native voxel size and the 4 nm grid.  On
            ``sft`` it is the clean native EM captured before the input is
            resampled to the fine grid, then placed on that same coarser
            grid (optional; the data-consistency term is skipped when the
            key is absent).
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
        weight_dif: float = 1.0,
        diffusion_steps: int = 1,
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
        self.weight_dif = float(weight_dif)
        self.diffusion_steps = int(diffusion_steps)

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

        Delegates to :meth:`AffinityFGLoss.build_targets`.  ``labels`` stays
        on the rung's native grid; the head is pooled to that shape in
        :meth:`forward`, so the target is built from the labels as-is.
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
        if self.weight_dif > 0:
            keys.append("loss/rflow")
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
        so the term works when the target is already on the 4 nm grid (COSEM;
        the pool is a no-op) and when it stays on a coarser native grid
        (FIB 8 nm SSL, and the sft native-EM target).
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
            if sem_label.shape[-3:] != labels.shape[-3:]:
                raise ValueError(
                    "sem_label spatial shape "
                    f"{tuple(sem_label.shape[-3:])} must match labels "
                    f"{tuple(labels.shape[-3:])} (sem_label stays on the "
                    "native label grid; the head is pooled to match)."
                )
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
            head: ``[B, n_aff + 2, D, H, W]`` small-voxel head (aff/sem
                logits, raw linear) on the fixed 4 nm network grid.
            targets: Must carry ``task``.  ``ssl`` carries ``recon_image``
                (the clean EM target, on the 4 nm grid for COSEM or on the
                native grid when that is coarser, e.g. FIB 8 nm).  ``sft``
                carries ``labels`` at the native grid -- the pool factor is
                read from the label shape, so no voxel-count key is needed --
                and optionally ``recon_image`` (clean native EM on the coarser
                of native and fine) for the ``raw`` data-consistency term, and
                optionally ``sem_label`` (boundary-eroded foreground on the
                label grid; supervises the sem head only --
                ``boundary_target: semantic``).
                ``_cached_targets`` (from :meth:`build_targets`) is used when
                present.

        Returns:
            ``ssl`` -> ``{"loss", "loss/recon"}``;
            ``sft``  -> ``{"loss", "loss/aff", "loss/sem", ["loss/recon"]}``.
        """
        if head.shape[1] != self.head_channels:
            raise ValueError(
                f"Joint3DReconSegLoss expects a {self.head_channels}-channel "
                f"head (n_aff + 2); got {head.shape[1]}."
            )
        task = self._resolve_task(targets[self.task_key])
        cached: Optional[Dict[str, torch.Tensor]] = targets.get("_cached_targets")
        out: Dict[str, torch.Tensor] = {}
        total = head.new_zeros(())

        recon_image = targets.get(self.recon_image_key)

        if task == SSL:
            # ---- self-supervised reconstruction of the clean EM ----
            # Match ``raw`` (pooled to the recon grid) to ``recon_image``.
            # ``recon_image`` is REQUIRED (it is the only supervision on this
            # label-free branch).
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
            # Data-consistency on the ``raw`` head (the unified head always
            # carries it).  Pool the predicted small-voxel ``raw`` to
            # ``recon_image`` (clean native EM on the coarser of native and
            # fine) and match it, so the super-resolved volume agrees with
            # the measured data after that downsample.  Optional (skipped
            # when the datamodule omits ``recon_image``).
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
            f"weight_sft={self.weight_sft}, weight_dif={self.weight_dif}, "
            f"diffusion_steps={self.diffusion_steps})"
        )


__all__ = ["Joint3DReconSegLoss", "SSL", "SFT"]
