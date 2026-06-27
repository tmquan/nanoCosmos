"""
Joint reconstruction + segmentation Lightning module (the nanoCosmos recipe).

``JointModule`` reuses the **whole** Cosmos-3 Nano training/eval machinery
(model build, freeze schedule, optimiser param-group split, gradient guard,
the epoch-end reduce/log loop) from
:class:`~nanocosmos.modules.cosmos_3_nano.module.Cosmos3Nano3DModule`, and
swaps in :class:`~nanocosmos.losses.JointReconSegLoss`.  Only three things
differ from the plain affinity recipe, all task-routing:

* :meth:`_loss_offsets` -- the offset list lives under ``loss.seg.offsets``
  (nested), so the head width is derived from there.
* :meth:`_prepare_targets` -- builds the per-batch ``targets`` for the active
  branch: ``dapt`` carries ``recon_image`` and **no labels**; ``sft`` carries
  native ``labels`` (+ cached affinity target) and optionally ``recon_image``
  (the original large-voxel EM for the ``raw`` data-consistency term).
* :meth:`_accumulate_metrics` -- on ``sft`` the small-voxel head is pooled to
  the native label grid before the foreground / Mutex-Watershed metrics; on
  ``dapt`` (no labels) segmentation metrics are skipped.

``training_step`` / ``validation_step`` / the epoch-end reduce are inherited
verbatim: they call ``_prepare_targets`` -> ``self.model(images)`` ->
``self.criterion(head, targets)``, and :class:`JointReconSegLoss` does the
branch routing + small-voxel→native pooling internally.

Batches must be **task-homogeneous** (one branch per step) -- the joint
datamodule's round-robin sampler guarantees this.
"""

from typing import Any, Dict

import torch
from einops import rearrange

from nanocosmos.losses import DAPT, JointReconSegLoss
from nanocosmos.modules.cosmos_3_nano.module import Cosmos3Nano3DModule


class JointModule(Cosmos3Nano3DModule):
    """Cosmos-3 Nano backbone trained with the joint recon + seg loss."""

    _loss_cls = JointReconSegLoss

    # ------------------------------------------------------------------
    # Config / target plumbing
    # ------------------------------------------------------------------

    def _loss_offsets(self, loss_config: Dict[str, Any]):
        # The joint loss nests the affinity config under ``seg``.
        return (loss_config.get("seg") or {}).get("offsets")

    @staticmethod
    def _batch_task(batch: Dict[str, Any]) -> str:
        """Resolve the (task-homogeneous) batch's branch name to a str."""
        task = batch.get("task")
        if isinstance(task, str):
            return task
        if isinstance(task, (list, tuple)) and task:
            uniq = set(map(str, task))
            if len(uniq) != 1:
                raise ValueError(
                    f"JointModule expects task-homogeneous batches; got mixed "
                    f"{sorted(uniq)}. Use the round-robin multi-task sampler."
                )
            return next(iter(uniq))
        raise KeyError(
            "JointModule batch is missing a 'task' field ('dapt' | 'sft'); "
            "the joint datamodule must tag each batch."
        )

    def _maybe_scale_recon(self, recon: torch.Tensor) -> torch.Tensor:
        """Match the raw head's range: scale [0,1] -> [-1,1] when vae_input_pm1."""
        if getattr(self.model, "vae_input_pm1", False):
            return recon * 2.0 - 1.0
        return recon

    @torch.no_grad()
    def _prepare_targets(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        ndim_with_channel = self._SPATIAL_DIMS + 2
        task = self._batch_task(batch)
        targets: Dict[str, Any] = {"task": task}

        # Reconstruction target (clean EM): required on dapt, optional on sft
        # (the raw data-consistency term).  Pooled to its own grid in the loss.
        recon = batch.get("recon_image")
        if recon is not None:
            targets["recon_image"] = self._maybe_scale_recon(recon)

        if task == DAPT:
            return targets  # label-free branch

        # ---- sft: native labels + cached affinity target ----
        labels = batch["label"]
        if labels.dim() == ndim_with_channel:
            labels = rearrange(labels, self._SQUEEZE_PATTERN)
        targets["labels"] = labels

        sem_label = batch.get("sem_label")
        if sem_label is not None:
            if sem_label.dim() == ndim_with_channel:
                sem_label = rearrange(sem_label, self._SQUEEZE_PATTERN)
            targets["sem_label"] = sem_label

        targets["_cached_targets"] = self.criterion.build_targets(labels, targets)
        return targets

    # ------------------------------------------------------------------
    # Eval metrics (pool small-voxel head -> native label grid for sft)
    # ------------------------------------------------------------------

    def _accumulate_metrics(
        self,
        head: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        prefix: str,
        bs: float,
    ) -> None:
        # dapt is reconstruction-only (no labels) -> no segmentation metrics.
        if targets.get("task") == DAPT or "labels" not in targets:
            return
        # The head is on the fine (small-voxel) grid; the labels are native.
        # Pool the head down to the label grid (a no-op when they match) so the
        # foreground + Mutex-Watershed metrics are scored at the supervised
        # resolution -- exactly the grid the seg loss used.
        labels = targets["labels"]
        pooled = self.criterion._pool_to(head, labels.shape[-3:])
        self._accumulate_semantic_metrics(pooled, targets, prefix, bs)
        self._accumulate_instance_metrics(pooled, targets, prefix, bs)


__all__ = ["JointModule"]
