"""TensorBoard visualisation for the joint reconstruction + segmentation recipe.

:class:`Joint3DImageLogger` extends :class:`ImageLogger` for the two-branch
:class:`~nanocosmos.modules.Joint3DModule`.  Every epoch it captures the first
``ssl`` batch AND the first ``sft`` batch and renders BOTH, each under a
task-namespaced tag ``{stage}/{mode}/{task}/...``:

* **both branches** -- reconstruction panels::

      {stage}/automatic/{ssl,sft}/true/image         the (degraded, on ssl) input
      {stage}/automatic/{ssl,sft}/pred/recon         raw-head small-voxel recon
      {stage}/automatic/ssl/true/recon_target        clean EM target (recon_image)
      # ssl only: on sft recon_target is a clone of true/image, so it is skipped

* **sft only** -- the fine head is pooled to the native label grid (matching
  the loss) and the usual segmentation panels are emitted::

      {stage}/automatic/sft/true/label        native instance labels
      {stage}/automatic/sft/true/sem          sem target (eroded sem_label > 0)
      {stage}/automatic/sft/pred/sem          foreground (sigmoid)
      {stage}/automatic/sft/pred/label        Mutex-Watershed instances
      {stage}/automatic/sft/aff/pred/{offset} a few affinity channels

The ``ssl`` branch is label-free, so it shows only the reconstruction panels.
"""

from typing import Any, Dict

import torch
from einops import rearrange, repeat

from nanocosmos.callbacks.tensorboard.heads import _add_aff_panels, aff_panel_indices
from nanocosmos.callbacks.tensorboard.image_logger import ImageLogger
from nanocosmos.callbacks.tensorboard.tags import TagContext
from nanocosmos.callbacks.tensorboard.viz import (
    _label_to_rgb,
    _normalise,
    _resize_2d,
    _to_2d,
)
from nanocosmos.losses import AFFINITY_OFFSETS, N_PULL, offset_names


def _gray3(panel_2d: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] (or [B,H,W]) -> normalised 3-channel grayscale for TB."""
    if panel_2d.dim() == 3:
        panel_2d = panel_2d.unsqueeze(1)
    return repeat(_normalise(panel_2d.float()), "b 1 h w -> b 3 h w")


class Joint3DImageLogger(ImageLogger):
    """Task-aware TensorBoard logger for :class:`Joint3DModule`.

    Logs **both** branches every epoch: it captures the first ``ssl`` batch and
    the first ``sft`` batch seen, and renders each under a task-namespaced tag
    (``{stage}/{mode}/{ssl|sft}/...``).  ``ssl`` shows reconstruction panels
    only; ``sft`` adds the pooled segmentation panels.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # First batch of EACH task per epoch, keyed by task name.  Replaces the
        # parent's single first-batch capture so both ssl + sft panels log.
        self._train_batches: Dict[str, Any] = {}
        self._val_batches: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Per-task batch capture (all ranks).  The round-robin sampler's schedule
    # is rank-identical, so every rank captures the same task set -> the
    # epoch-end forwards stay collective-consistent.
    # ------------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        t = self._resolve_task(batch.get("task"))
        if t not in self._train_batches:
            self._train_batches[t] = self._detach_batch(batch)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0,
    ) -> None:
        t = self._resolve_task(batch.get("task"))
        if t not in self._val_batches:
            self._val_batches[t] = self._detach_batch(batch)

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._run_all_tasks(trainer, pl_module, self._train_batches, stage="train")
        self._train_batches = {}

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._run_all_tasks(trainer, pl_module, self._val_batches, stage="val")
        self._val_batches = {}

    def _run_all_tasks(self, trainer, pl_module, batches, *, stage: str) -> None:
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs != 0:
            return
        tb = self._get_tb(trainer)  # real on rank 0, None elsewhere
        was_training = pl_module.training
        pl_module.eval()
        try:
            # Sorted task order -> identical forward sequence on every rank.
            for task in sorted(batches):
                self._run_visualization(
                    tb, trainer, pl_module, batches[task], stage=stage,
                )
        finally:
            if was_training:
                pl_module.train()

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
        # Namespace every panel by task -> {stage}/{mode}/{ssl|sft}/... so the
        # ssl and sft renders coexist in TensorBoard instead of overwriting.
        ctx = TagContext(stage=stage, mode=self.mode, head=task)
        epoch = pl_module.current_epoch

        # All panels are upsampled to the FINEST in-plane size (the fine network
        # grid, e.g. 256x256) so every TB image -- including the SFT seg panels
        # that are pooled to a coarser native grid -- renders at the same size.
        target_hw = tuple(_to_2d(images[:n]).shape[-2:])

        # ---- reconstruction panels (both branches) ----
        tb.add_images(ctx.tag("true/image"),
                      _resize_2d(_gray3(_to_2d(images[:n])), target_hw),
                      global_step=epoch)
        raw = head[:, crit.raw_slice][:n]               # [n,1,D,H,W] linear
        tb.add_images(ctx.tag("pred/recon"),
                      _resize_2d(_gray3(_to_2d(raw)), target_hw),
                      global_step=epoch)
        # The clean recon target only differs from true/image on the ssl branch
        # (where the input was degraded).  On sft the recon target is a clone of
        # the input, so logging it would just duplicate true/image -> skip it.
        recon_t = batch.get("recon_image")
        if recon_t is not None and task == "ssl":
            recon_t = recon_t.to(pl_module.device)
            if recon_t.dim() == self.spatial_dims + 1:
                recon_t = rearrange(recon_t, "b ... -> b 1 ...")
            tb.add_images(
                ctx.tag("true/recon_target"),
                _resize_2d(_gray3(_to_2d(recon_t[:n])), target_hw),
                global_step=epoch,
            )

        # ---- segmentation panels (sft only) ----
        if task == "sft" and "label" in batch:
            self._log_seg_panels(tb, ctx, pl_module, head, batch, n, target_hw)

    def _log_seg_panels(self, tb, ctx, pl_module, head, batch, n, target_hw=None):
        epoch = pl_module.current_epoch
        crit = pl_module.criterion
        labels = batch["label"].to(pl_module.device)
        if labels.dim() == self.spatial_dims + 2:
            labels = rearrange(labels, "b 1 ... -> b ...")
        labels = labels[:n]

        # Pool the fine head to the native label grid (matches the loss).
        pooled = crit._pool_to(head[:n], labels.shape[-3:])

        # True instance labels.  (nearest upsample -> no label-colour blending.)
        labels_2d = rearrange(
            _to_2d(rearrange(labels, "b ... -> b 1 ...")), "b 1 ... -> b ...",
        )
        tb.add_images(
            ctx.tag("true/label"),
            _resize_2d(_label_to_rgb(labels_2d.long()), target_hw, mode="nearest")
            if target_hw else _label_to_rgb(labels_2d.long()),
            global_step=epoch,
        )

        # Ground-truth foreground the sem head is trained on.  Uses the eroded
        # ``sem_label`` (boundary_target: semantic) when present so the panel
        # matches the actual target -- thin membrane gaps; else the instance
        # foreground.
        sem_src = batch.get("sem_label")
        if sem_src is not None:
            sem_src = sem_src.to(pl_module.device)
            if sem_src.dim() == self.spatial_dims + 2:
                sem_src = rearrange(sem_src, "b 1 ... -> b ...")
            sem_src = sem_src[:n]
        else:
            sem_src = labels
        gt_sem = rearrange((sem_src > 0).float(), "b ... -> b 1 ...")
        gt_sem_panel = _gray3(_to_2d(gt_sem))
        if target_hw:
            gt_sem_panel = _resize_2d(gt_sem_panel, target_hw)
        tb.add_images(ctx.tag("true/sem"), gt_sem_panel, global_step=epoch)

        # Foreground (sem) prediction.
        sem = pooled[:, crit.sem_slice].sigmoid()
        sem_panel = _gray3(_to_2d(sem))
        if target_hw:
            sem_panel = _resize_2d(sem_panel, target_hw)
        tb.add_images(ctx.tag("pred/sem"), sem_panel, global_step=epoch)

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
            size=target_hw,
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
            seg_rgb = _label_to_rgb(seg_2d.long())
            if target_hw:
                seg_rgb = _resize_2d(seg_rgb, target_hw, mode="nearest")
            tb.add_images(ctx.tag("pred/label"), seg_rgb, global_step=epoch)

    @staticmethod
    def _resolve_task(task: Any) -> str:
        if isinstance(task, str):
            return task
        if isinstance(task, (list, tuple)) and task:
            return str(task[0])
        return "sft"


__all__ = ["Joint3DImageLogger"]
