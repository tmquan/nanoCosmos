"""TensorBoard visualisation for the joint reconstruction + segmentation recipe.

:class:`Joint3DImageLogger` extends :class:`ImageLogger` for the two-branch
:class:`~nanocosmos.modules.Joint3DModule`.  Every epoch it captures the first
``ssl`` batch AND the first ``sft`` batch and renders BOTH, each under a
task-namespaced tag ``{stage}/{mode}/{task}/...``:

* **both branches** -- reconstruction panels, each as the three orthogonal
  mid-planes (``xy`` / ``xz`` / ``yz``) on the 4 nm grid.  A source or target
  that is coarser than 4 nm is nearest-upsampled onto that grid before the
  plane is drawn::

      {stage}/automatic/{ssl,sft}/true/image/{xy,xz,yz}    source (degraded on ssl)
      {stage}/automatic/{ssl,sft}/pred/recon/{xy,xz,yz}  raw head on the degraded latent
      {stage}/automatic/{ssl,sft}/pred/Euler/{xy,xz,yz} raw head after Euler
                                                     integration of proj_out
                                                     (logged when weight_dif > 0)
      {stage}/automatic/{ssl,sft}/pred/wan_decoder/{xy,xz,yz}
                                                     DiT latent through the frozen
                                                     preserved original_conv_out
                                                     (diagnostic; skipped when the
                                                     wrapper has no pretrained VAE)
      {stage}/automatic/ssl/true/target/{xy,xz,yz}   clean EM target (recon_image)
      # ssl only.  sft also carries recon_image (native EM) but that panel
      # is not logged.
      {stage}/automatic/{ssl,sft}/volume             text: source volume name(s)

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
    _orthogonal_midplanes,
    _resize_2d,
    _to_2d,
)
from nanocosmos.losses import AFFINITY_OFFSETS, N_PULL, offset_names


def _gray3(panel_2d: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] (or [B,H,W]) -> normalised 3-channel grayscale for TB."""
    if panel_2d.dim() == 3:
        panel_2d = panel_2d.unsqueeze(1)
    return repeat(_normalise(panel_2d.float()), "b 1 h w -> b 3 h w")


def _log_ortho_planes(tb, ctx, epoch: int, tag: str, vol: torch.Tensor, fine_size) -> None:
    """Log the xy / xz / yz mid-planes of ``vol`` on the 4 nm ``fine_size`` grid.

    ``vol`` is ``[B, C, D, H, W]``.  Multi-channel volumes (the Wan decoder's
    RGB render) are averaged to one grayscale plane before the contrast stretch.
    """
    for name, plane in _orthogonal_midplanes(vol, fine_size).items():
        if plane.shape[1] != 1:
            plane = plane.mean(dim=1, keepdim=True)
        tb.add_images(ctx.tag(f"{tag}/{name}"), _gray3(plane), global_step=epoch)


class Joint3DImageLogger(ImageLogger):
    """Task-aware TensorBoard logger for :class:`Joint3DModule`.

    Logs **both** branches every epoch: it captures the first ``ssl`` batch and
    the first ``sft`` batch seen, and renders each under a task-namespaced tag
    (``{stage}/{mode}/{ssl|sft}/...``).  ``ssl`` shows reconstruction panels
    only; ``sft`` adds the pooled segmentation panels.

    Panels are captured per task on every rank.  Rank 0 writes them from
    ``on_train_epoch_end`` / ``on_validation_epoch_end``.  The validation hook
    also runs at the end of Lightning's sanity check, so step-0 panels appear
    after sanity validation.
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
        tb = self._get_tb(trainer)  # SummaryWriter on rank 0; dummy elsewhere
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

            # Frozen Wan-decoder diagnostic: decode the model's latent through
            # the preserved ``original_conv_out`` (the pretrained Wan conv,
            # frozen; the decoder's live ``conv_out`` is ``Identity``).
            # Diagnostic only -- ``original_conv_out`` is never optimised.
            # Run per-image on ALL ranks (collective-consistent with the head
            # forward above; ``wan_decoder_output`` is a ``@torch.no_grad``
            # method).  Returns ``None`` on wrappers without a pretrained VAE
            # (e.g. the random-init standalone DiT), in which case the panel is
            # simply skipped.
            wan_fn = getattr(getattr(pl_module, "model", None), "wan_decoder_output", None)
            wan_rgb = None
            if callable(wan_fn):
                wan_parts = [wan_fn(images[i:i + 1]) for i in range(n)]
                if all(p is not None for p in wan_parts):
                    wan_rgb = torch.cat(wan_parts, dim=0).float()

            # pred/Euler: integrate proj_out from the degraded latent, then
            # the same task head. pred/recon (``head`` above) stays the
            # degraded-latent readout. Both run on every rank.
            few_raw = None
            crit_cfg = getattr(pl_module, "criterion", None)
            few_fn = getattr(getattr(pl_module, "model", None), "few_step_reconstruct", None)
            rflow_w = float(getattr(crit_cfg, "weight_dif", 0.0) or 0.0)
            diffusion_steps = int(getattr(crit_cfg, "diffusion_steps", 1) or 1)
            if callable(few_fn) and rflow_w > 0 and diffusion_steps > 0:
                few_parts = [
                    few_fn(images[i:i + 1], steps=diffusion_steps) for i in range(n)
                ]
                few_head = torch.cat(few_parts, dim=0).float()
                raw_slice = getattr(crit_cfg, "raw_slice", None)
                few_raw = few_head[:, raw_slice] if raw_slice is not None else few_head[:, -1:]

        if tb is None:  # no image logger: forward done, nothing to write
            return

        crit = pl_module.criterion
        # Namespace every panel by task -> {stage}/{mode}/{ssl|sft}/... so the
        # ssl and sft renders coexist in TensorBoard instead of overwriting.
        ctx = TagContext(stage=stage, mode=self.mode, head=task)
        epoch = pl_module.current_epoch

        # Per-sample display depth for the sft segmentation panels: the z-slice
        # with the most semantic foreground (a RELATIVE depth in [0, 1]).  The
        # fixed central slice can be empty / fully eroded on sparse or
        # thin-structure crops (e.g. MICrONS neurites, or a single near-solid
        # instance) even when the 3-D target is non-empty, showing up as a
        # misleading all-black true/sem panel.  _best_depth_frac falls back PER
        # SAMPLE from sem_label to the pristine label whenever a given sample's
        # sem_label is empty on every slice.  A relative depth maps onto both
        # the fine grid and the coarser native label grid.  Source and target
        # EM panels are the orthogonal mid-planes, not this slice.  ssl carries
        # no seg target -> None.
        frac_z = (
            self._best_depth_frac(batch, pl_module, n) if task == "sft" else None
        )

        # Record which source volume(s) the panels came from, so a black/odd
        # panel can be traced to a specific volume.  TensorBoard image cards
        # can't carry dynamic titles, so this logs a sibling text card under
        # the same task namespace (``{stage}/{mode}/{task}/volume``).
        vols = batch.get("volume")
        if vols is not None:
            if isinstance(vols, str):
                vols = [vols]
            shown = [str(v) for v in list(vols)[:n]]
            # De-duplicate while preserving order (volume-balance mode draws a
            # batch from one volume; resolution/subset mode may mix several).
            uniq = list(dict.fromkeys(shown))
            tb.add_text(ctx.tag("volume"), ", ".join(uniq), global_step=epoch)

        # 4 nm field of view: the network input.  Source and target mid-planes
        # are drawn on this grid; a coarser recon target is upsampled per plane.
        # Seg panels still share the fine in-plane size so labels line up with xy.
        fine_size = tuple(int(s) for s in images.shape[-3:])
        target_hw = (fine_size[1], fine_size[2])

        # ---- reconstruction panels (both branches) ----
        # Orthogonal mid-planes: xy (fixed z), xz (fixed y), yz (fixed x).
        _log_ortho_planes(tb, ctx, epoch, "true/image", images[:n], fine_size)
        raw = head[:, crit.raw_slice][:n]               # [n,1,D,H,W] linear
        _log_ortho_planes(tb, ctx, epoch, "pred/recon", raw, fine_size)
        if few_raw is not None:
            _log_ortho_planes(
                tb, ctx, epoch, "pred/Euler", few_raw[:n], fine_size,
            )
        # Wan-decoder diagnostic (frozen ``original_conv_out``).
        # RGB [n,3,D,H,W] ~[-1,1]; each plane is collapsed to grayscale.
        if wan_rgb is not None:
            _log_ortho_planes(
                tb, ctx, epoch, "pred/wan_decoder", wan_rgb[:n], fine_size,
            )
        # ssl only.  ``recon_image`` is the clean EM on its own grid, which is
        # often coarser than the 4 nm source; ``_log_ortho_planes`` nearest-
        # upsamples each plane onto that grid.  sft also stores ``recon_image`` (the
        # native EM, captured before the input is resampled) and this logger
        # does not draw it.
        recon_t = batch.get("recon_image")
        if recon_t is not None and task == "ssl":
            recon_t = recon_t.to(pl_module.device)
            if recon_t.dim() == self.spatial_dims + 1:
                recon_t = rearrange(recon_t, "b ... -> b 1 ...")
            _log_ortho_planes(tb, ctx, epoch, "true/target", recon_t[:n], fine_size)

        # ---- segmentation panels (sft only) ----
        if task == "sft" and "label" in batch:
            self._log_seg_panels(
                tb, ctx, pl_module, head, batch, n, target_hw, frac_z=frac_z,
            )

    def _per_z_foreground(self, batch, key, pl_module, n):
        """``[n, D]`` per-z foreground voxel counts for ``batch[key]``, or
        ``None`` if the key is absent / not a 3-D volume tensor."""
        t = batch.get(key)
        if t is None:
            return None
        s = t.to(pl_module.device)
        if s.dim() == self.spatial_dims + 2:      # [B,1,D,H,W] -> [B,D,H,W]
            s = rearrange(s, "b 1 ... -> b ...")
        if s.dim() != self.spatial_dims + 1:      # expect [B,D,H,W]
            return None
        fg = (s[:n] > 0).float()
        if fg.shape[1] <= 1:
            return None
        return fg.flatten(2).sum(-1)              # [n, D]

    def _best_depth_frac(self, batch, pl_module, n):
        """Per-sample relative depth (``[0, 1]``) of the z-slice with the most
        semantic foreground, used to slice-pick the sft panels.

        Prefers the eroded ``sem_label`` (the actual sem target) per sample,
        but falls back to the instance ``label`` **per sample** (not just when
        the key is entirely absent) whenever ``sem_label`` has zero foreground
        on every slice for that particular sample -- e.g. a thin/sheet-like
        single instance whose boundary erosion wipes out most cross-sections.
        Without this, ``argmax`` over an all-zero row silently picks index 0,
        an arbitrary slice that is usually empty for `true/sem` even though
        the (pristine) label clearly has content nearby -- the "still black
        semantic on a single-instance crop" symptom. Returns ``None`` (=>
        central slice) when no usable target is present or every slice is
        empty for every sample.
        """
        per_z_sem = self._per_z_foreground(batch, "sem_label", pl_module, n)
        per_z_label = self._per_z_foreground(batch, "label", pl_module, n)
        per_z = per_z_sem if per_z_sem is not None else per_z_label
        if per_z is None:
            return None
        depth = per_z.shape[1]

        if per_z_sem is not None and per_z_label is not None:
            empty = ~torch.any(per_z_sem > 0, dim=1)
            if bool(empty.any()):
                per_z = per_z.clone()
                per_z[empty] = per_z_label[empty]

        if not torch.any(per_z > 0):
            return None                           # empty target -> central slice
        return per_z.argmax(dim=1).float() / (depth - 1)

    def _log_seg_panels(
        self, tb, ctx, pl_module, head, batch, n, target_hw=None, frac_z=None,
    ):
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
            _to_2d(rearrange(labels, "b ... -> b 1 ..."), frac_z),
            "b 1 ... -> b ...",
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
        gt_sem_panel = _gray3(_to_2d(gt_sem, frac_z))
        if target_hw:
            gt_sem_panel = _resize_2d(gt_sem_panel, target_hw)
        tb.add_images(ctx.tag("true/sem"), gt_sem_panel, global_step=epoch)

        # Foreground (sem) prediction.
        sem = pooled[:, crit.sem_slice].sigmoid()
        sem_panel = _gray3(_to_2d(sem, frac_z))
        if target_hw:
            sem_panel = _resize_2d(sem_panel, target_hw)
        tb.add_images(ctx.tag("pred/sem"), sem_panel, global_step=epoch)

        # Affinity channels (curated subset) + Mutex-Watershed instances.
        agg = getattr(pl_module, "agglomerator", None)
        offsets = getattr(agg, "offsets", AFFINITY_OFFSETS) if agg else AFFINITY_OFFSETS
        n_pull = getattr(agg, "n_pull", N_PULL) if agg else N_PULL
        aff = pooled[:, crit.aff_slice].sigmoid().float()
        names = offset_names(offsets, n_pull)
        mask_2d = torch.ones_like(_to_2d(aff[:, :1], frac_z))
        # Every pull offset, plus up to max_push=7 evenly spaced push offsets.
        _add_aff_panels(
            tb, ctx, aff, aff_panel_indices(len(offsets), n_pull, max_push=7),
            names=names, mask_2d=mask_2d, epoch=epoch, tag_prefix="aff/pred",
            size=target_hw, frac=frac_z,
        )

        if agg is not None and self.spatial_dims == 3:
            if getattr(agg, "gate_with_sem", True):
                thr = getattr(agg, "sem_gate_threshold", 0.5)
                sem_fg = sem[:, 0] > thr
            else:
                sem_fg = None
            seg_3d = agg(aff, sem_fg)
            seg_2d = rearrange(
                _to_2d(rearrange(seg_3d, "b ... -> b 1 ..."), frac_z),
                "b 1 ... -> b ...",
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
