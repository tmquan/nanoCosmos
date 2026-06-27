"""TensorBoard visualisation for the joint reconstruction + segmentation recipe.

:class:`JointImageLogger` extends :class:`ImageLogger` for the two-branch
:class:`~nanocosmos.modules.JointModule`.  It reuses the parent's batch-capture
/ epoch-end / autocast machinery and only overrides the per-epoch render so it
can branch on ``batch["task"]``:

* **both branches** -- reconstruction panels::

      {stage}/automatic/true/image        the (degraded, on dapt) network input
      {stage}/automatic/pred/recon        the raw-head small-voxel reconstruction
      {stage}/automatic/true/recon_target the clean EM target (recon_image)

* **sft only** -- the fine head is pooled to the native label grid (matching
  the loss) and the usual segmentation panels are emitted::

      {stage}/automatic/true/label        native instance labels
      {stage}/automatic/pred/sem          foreground (sigmoid)
      {stage}/automatic/pred/label        Mutex-Watershed instances
      {stage}/automatic/aff/pred/{offset} a few affinity channels

The ``dapt`` branch is label-free, so it shows only the reconstruction panels.
"""

from typing import Any, Dict

import torch
from einops import rearrange, repeat

from nanocosmos.callbacks.tensorboard.heads import _add_aff_panels, aff_panel_indices
from nanocosmos.callbacks.tensorboard.image_logger import ImageLogger
from nanocosmos.callbacks.tensorboard.tags import TagContext
from nanocosmos.callbacks.tensorboard.viz import _label_to_rgb, _normalise, _to_2d
from nanocosmos.losses import AFFINITY_OFFSETS, N_PULL, offset_names


def _gray3(panel_2d: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] (or [B,H,W]) -> normalised 3-channel grayscale for TB."""
    if panel_2d.dim() == 3:
        panel_2d = panel_2d.unsqueeze(1)
    return repeat(_normalise(panel_2d.float()), "b 1 h w -> b 3 h w")


class JointImageLogger(ImageLogger):
    """Task-aware TensorBoard logger for :class:`JointModule`."""

    def _run_visualization(self, tb, trainer, pl_module, batch, *, stage: str):
        task = self._resolve_task(batch.get("task"))
        device_type = str(pl_module.device).split(":")[0]
        autocast_enabled = device_type == "cuda"
        amp_dtype = self._resolve_autocast_dtype(trainer, pl_module)

        with torch.no_grad(), torch.amp.autocast(
            device_type=device_type, enabled=autocast_enabled, dtype=amp_dtype,
        ):
            images = batch["image"].to(pl_module.device)
            if images.dim() == self.spatial_dims + 1:
                images = rearrange(images, "b ... -> b 1 ...")
            n = min(images.shape[0], self.max_images)
            fwd_module = getattr(trainer, "model", None) or pl_module
            head = torch.cat(
                [fwd_module(images[i:i + 1]) for i in range(n)], dim=0,
            ).float()

        if tb is None:  # non-rank-0: forward done (collective), no logging
            return

        crit = pl_module.criterion
        ctx = TagContext(stage=stage, mode=self.mode)

        # ---- reconstruction panels (both branches) ----
        tb.add_images(ctx.tag("true/image"), _gray3(_to_2d(images[:n])), global_step=pl_module.current_epoch)
        raw = head[:, crit.raw_slice][:n]               # [n,1,D,H,W] linear
        tb.add_images(ctx.tag("pred/recon"), _gray3(_to_2d(raw)), global_step=pl_module.current_epoch)
        recon_t = batch.get("recon_image")
        if recon_t is not None:
            recon_t = recon_t.to(pl_module.device)
            if recon_t.dim() == self.spatial_dims + 1:
                recon_t = rearrange(recon_t, "b ... -> b 1 ...")
            tb.add_images(
                ctx.tag("true/recon_target"), _gray3(_to_2d(recon_t[:n])),
                global_step=pl_module.current_epoch,
            )

        # ---- segmentation panels (sft only) ----
        if task == "sft" and "label" in batch:
            self._log_seg_panels(tb, ctx, pl_module, head, batch, n)

    def _log_seg_panels(self, tb, ctx, pl_module, head, batch, n):
        epoch = pl_module.current_epoch
        crit = pl_module.criterion
        labels = batch["label"].to(pl_module.device)
        if labels.dim() == self.spatial_dims + 2:
            labels = rearrange(labels, "b 1 ... -> b ...")
        labels = labels[:n]

        # Pool the fine head to the native label grid (matches the loss).
        pooled = crit._pool_to(head[:n], labels.shape[-3:])

        # True instance labels.
        labels_2d = rearrange(
            _to_2d(rearrange(labels, "b ... -> b 1 ...")), "b 1 ... -> b ...",
        )
        tb.add_images(ctx.tag("true/label"), _label_to_rgb(labels_2d.long()), global_step=epoch)

        # Foreground (sem) prediction.
        sem = pooled[:, crit.sem_slice].sigmoid()
        tb.add_images(ctx.tag("pred/sem"), _gray3(_to_2d(sem)), global_step=epoch)

        # Affinity channels (curated subset) + Mutex-Watershed instances.
        agg = getattr(pl_module, "agglomerator", None)
        offsets = getattr(agg, "offsets", AFFINITY_OFFSETS) if agg else AFFINITY_OFFSETS
        n_pull = getattr(agg, "n_pull", N_PULL) if agg else N_PULL
        aff = pooled[:, crit.aff_slice].sigmoid().float()
        names = offset_names(offsets, n_pull)
        mask_2d = torch.ones_like(_to_2d(aff[:, :1]))
        _add_aff_panels(
            tb, ctx, aff, aff_panel_indices(len(offsets), n_pull, max_push=4),
            names=names, mask_2d=mask_2d, epoch=epoch, tag_prefix="aff/pred",
        )

        if agg is not None and self.spatial_dims == 3:
            if getattr(agg, "gate_with_sem", True):
                thr = getattr(agg, "sem_gate_threshold", 0.5)
                sem_fg = sem[:, 0] > thr
            else:
                sem_fg = None
            seg_3d = agg(aff, sem_fg)
            seg_2d = rearrange(
                _to_2d(rearrange(seg_3d, "b ... -> b 1 ...")), "b 1 ... -> b ...",
            )
            tb.add_images(ctx.tag("pred/label"), _label_to_rgb(seg_2d.long()), global_step=epoch)

    @staticmethod
    def _resolve_task(task: Any) -> str:
        if isinstance(task, str):
            return task
        if isinstance(task, (list, tuple)) and task:
            return str(task[0])
        return "sft"


__all__ = ["JointImageLogger"]
