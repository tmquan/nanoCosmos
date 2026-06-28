"""
Multi-task datamodule for the joint reconstruction + segmentation recipe.

Feeds :class:`~nanocosmos.modules.Joint3DModule` the batch contract it expects
(see doc/JOINT_TRAINING.md): ``task`` (``"ssl"`` | ``"sft"``), the fine-grid
``image``, the native-grid ``label`` (sft), and the ``recon_image`` target.

Geometry (see doc/RESOLUTION_LADDER.md).  The network runs on a fixed fine
grid (``patch_size`` @ ``pixel_size`` nm, e.g. 320x256x256 @ 4 nm).  Each
volume is read at its **native** voxel size over the same physical field of
view, then :class:`~nanocosmos.transforms.ToFineGridd` resamples the image onto
the fine grid while the label stays native and the recon target sits on the
coarser of native/fine.

Branches & batching.  Volumes are grouped by ``(task, native_resolution)`` --
samples in a group share a tensor shape, so a batch can only be drawn from one
group.  A round-robin batch sampler interleaves groups (weighted by each
branch's ``sample_weight``), yielding **task- and shape-homogeneous** batches,
which is exactly what ``Joint3DReconSegLoss`` / ``Joint3DModule`` require.

Config schema (``cfg.data``)::

    patch_size: [320, 256, 256]      # fine grid (z, y, x)
    pixel_size: [4, 4, 4]            # fine voxel size nm (cubic)
    degrade: {zf_range: [...], ...}  # RandResolutionDegraded kwargs (ssl)
    branches:
      ssl: {batch_size, sample_weight, volumes: [{vol, root, native_resolution}]}
      sft:  {batch_size, sample_weight, volumes: [{vol, seg, root, native_resolution}]}
    val_volumes: [{vol, seg, root, task, native_resolution}]   # optional
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
)
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from nanocosmos.datasets.lazy import LazyVolDataset
from nanocosmos.transforms import (
    FindBoundariesd,
    Labeld,
    RandResolutionDegraded,
    ToFineGridd,
)

logger = logging.getLogger(__name__)

_DEGRADE_KEYS = (
    "zf_range", "prob", "jitter_prob", "max_jitter", "missing_prob",
    "max_missing", "missing_fill", "noise_prob", "noise_std", "contrast_prob",
    "contrast_gamma", "up_mode",
)


def _scaled(fine_patch: Sequence[int], fine_nm: float, res: Sequence[float]) -> Tuple[int, ...]:
    """Native voxel count covering the fine FOV: round(fine * fine_nm / res)."""
    return tuple(max(1, int(round(fine_patch[d] * fine_nm / float(res[d]))))
                 for d in range(len(fine_patch)))


class _RoundRobinBatchSampler(Sampler):
    """Yield task/shape-homogeneous batches, round-robin across groups.

    Each group occupies a contiguous index range ``[offset, offset+length)`` in
    the parent :class:`ConcatDataset`.  A group contributes
    ``length // batch_size`` batches per epoch; the per-group batch counts are
    interleaved (shuffled on train, sequential on val).  ``sample_weight`` is
    folded into each group's ``length`` (= num_samples) by the datamodule.
    """

    def __init__(
        self,
        groups: List[Tuple[int, int, int]],   # (offset, length, batch_size)
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        self.groups = groups
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return sum(length // bs for _, length, bs in self.groups if bs > 0)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        schedule: List[int] = []
        for gi, (_off, length, bs) in enumerate(self.groups):
            schedule += [gi] * (length // bs)
        if self.shuffle:
            rng.shuffle(schedule)
        cursors = [0] * len(self.groups)
        for gi in schedule:
            off, length, bs = self.groups[gi]
            if self.shuffle:
                yield [off + rng.randrange(length) for _ in range(bs)]
            else:
                c = cursors[gi]
                idx = [off + ((c + j) % length) for j in range(bs)]
                cursors[gi] = (c + bs) % length
                yield idx


class Joint3DDataModule(pl.LightningDataModule):
    """Round-robin multi-task datamodule for the joint recipe."""

    def __init__(
        self,
        data_root: str = "data",
        patch_size: Sequence[int] = (320, 256, 256),
        pixel_size: Sequence[float] = (4.0, 4.0, 4.0),
        branches: Optional[Dict[str, Any]] = None,
        degrade: Optional[Dict[str, Any]] = None,
        val_volumes: Optional[List[Dict[str, Any]]] = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        num_samples: int = 8000,
        val_num_samples: int = 16,
        val_batch_size: int = 1,
        min_foreground: float = 0.0,
        find_boundaries: float = 0.0,
        boundary_target: str = "semantic",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.fine_patch = tuple(int(s) for s in patch_size)
        # Cubic fine grid: use the z spacing as the (single) fine voxel size.
        self.fine_nm = float(pixel_size[0])
        self.branches = dict(branches or {})
        self.degrade = {k: v for k, v in dict(degrade or {}).items() if k in _DEGRADE_KEYS}
        self.val_volumes = val_volumes
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and num_workers > 0
        self.prefetch_factor = int(prefetch_factor)
        self.num_samples = int(num_samples)
        self.val_num_samples = int(val_num_samples)
        self.val_batch_size = int(val_batch_size)
        self.min_foreground = float(min_foreground)
        # sem-head boundary supervision (sft only).  ``find_boundaries`` = per-
        # sample probability of eroding membrane voxels so the sem head targets
        # thin membranes instead of (near-degenerate) full foreground.
        # ``boundary_target``: "semantic" -> a separate eroded ``sem_label`` for
        # the sem head only (instance ``label`` stays pristine for affinity);
        # "both" -> erode the shared ``label`` (sem AND affinity targets).
        self.find_boundaries = float(find_boundaries)
        self.boundary_target = str(boundary_target)
        if self.boundary_target not in ("both", "semantic"):
            raise ValueError(
                f"boundary_target must be 'both' or 'semantic'; got {boundary_target!r}."
            )
        self.seed = int(seed)

        self._train_groups: List[Tuple[int, int, int]] = []
        self._val_groups: List[Tuple[int, int, int]] = []
        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None

    # ------------------------------------------------------------------
    # Per-group transform pipelines
    # ------------------------------------------------------------------

    def _group_transform(self, task: str, native_res: Sequence[float]) -> Compose:
        recon_size = _scaled(self.fine_patch, self.fine_nm,
                             [max(float(r), self.fine_nm) for r in native_res])
        if task == "ssl":
            return Compose([
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                RandResolutionDegraded(keys=["image"], recon_key="recon_image", **self.degrade),
                ToFineGridd(
                    image_size=self.fine_patch, recon_size=recon_size,
                    set_recon_from_image=False, task="ssl",
                ),
                EnsureTyped(keys=["image", "recon_image"]),
            ])
        # sft pipeline.  Optional boundary erosion makes the sem head target thin
        # membranes instead of near-degenerate full foreground.  Erosion runs on
        # the NATIVE label grid using this group's native resolution, so
        # FindBoundariesd's anisotropy guard (xy-only when z is >2x coarser)
        # applies per dataset.
        sft_tf: List[Any] = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Labeld(keys=["label"], spatial_dims=3),
        ]
        out_keys = ["image", "label", "recon_image"]
        if self.find_boundaries > 0:
            if self.boundary_target == "both":
                sft_tf.append(FindBoundariesd(
                    keys=["label"], prob=self.find_boundaries, pixel_size=native_res,
                ))
            else:  # "semantic": eroded sem_label only; instance label stays pristine
                sft_tf += [
                    CopyItemsd(keys=["label"], times=1, names=["sem_label"]),
                    FindBoundariesd(
                        keys=["sem_label"], prob=self.find_boundaries,
                        pixel_size=native_res,
                    ),
                ]
                out_keys.append("sem_label")
        sft_tf += [
            ToFineGridd(
                image_size=self.fine_patch, recon_size=recon_size,
                set_recon_from_image=True, task="sft",
            ),
            EnsureTyped(keys=out_keys),
        ]
        return Compose(sft_tf)

    def _build_group(
        self,
        task: str,
        native_res: Tuple[float, ...],
        volumes: List[Dict[str, Any]],
        num_samples: int,
        deterministic: bool,
    ) -> LazyVolDataset:
        native_patch = _scaled(self.fine_patch, self.fine_nm, native_res)
        return LazyVolDataset(
            root_dir=self.data_root,
            volumes=[{k: v for k, v in vol.items() if k in ("vol", "seg", "root")}
                     for vol in volumes],
            patch_size=native_patch,
            transform=self._group_transform(task, native_res),
            num_samples=num_samples,
            min_foreground=(self.min_foreground if task == "sft" else 0.0),
            deterministic=deterministic,
        )

    @staticmethod
    def _group_by_res(volumes: List[Dict[str, Any]]) -> Dict[Tuple[float, ...], List[Dict[str, Any]]]:
        groups: Dict[Tuple[float, ...], List[Dict[str, Any]]] = {}
        for vol in volumes:
            res = tuple(float(r) for r in vol["native_resolution"])
            groups.setdefault(res, []).append(vol)
        return groups

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        datasets: List[LazyVolDataset] = []
        group_specs: List[Tuple[int, int, int]] = []
        offset = 0
        for task, bcfg in self.branches.items():
            bs = int(bcfg.get("batch_size", 2))
            weight = float(bcfg.get("sample_weight", 1.0))
            vols = [dict(v) for v in bcfg.get("volumes", [])]
            n_group = max(bs, int(round(self.num_samples * weight)))
            for res, gvols in self._group_by_res(vols).items():
                ds = self._build_group(task, res, gvols, n_group, deterministic=False)
                datasets.append(ds)
                group_specs.append((offset, len(ds), bs))
                offset += len(ds)
                logger.info(
                    "Joint train group: task=%s res=%s vols=%d native_patch=%s "
                    "len=%d bs=%d", task, res, len(gvols),
                    _scaled(self.fine_patch, self.fine_nm, res), len(ds), bs,
                )
        if not datasets:
            raise ValueError("Joint3DDataModule: no train volumes configured under data.branches.")
        self.train_dataset = ConcatDataset(datasets)
        self._train_groups = group_specs

        # Validation: explicit val_volumes (flat, each tagged with task +
        # native_resolution), else the sft branch volumes (labeled -> metrics).
        val_vols = self.val_volumes
        if not val_vols:
            sft = self.branches.get("sft", {})
            val_vols = [dict(v, task="sft") for v in sft.get("volumes", [])]
        v_datasets: List[LazyVolDataset] = []
        v_specs: List[Tuple[int, int, int]] = []
        v_off = 0
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for v in val_vols:
            by_task.setdefault(v.get("task", "sft"), []).append(v)
        for task, vols in by_task.items():
            for res, gvols in self._group_by_res(vols).items():
                ds = self._build_group(task, res, gvols, self.val_num_samples, deterministic=True)
                v_datasets.append(ds)
                v_specs.append((v_off, len(ds), self.val_batch_size))
                v_off += len(ds)
        if v_datasets:
            self.val_dataset = ConcatDataset(v_datasets)
            self._val_groups = v_specs

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def _loader(self, dataset, groups, shuffle):
        return DataLoader(
            dataset,
            batch_sampler=_RoundRobinBatchSampler(groups, shuffle=shuffle, seed=self.seed),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="forkserver" if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, self._train_groups, shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return self._loader(self.val_dataset, self._val_groups, shuffle=False)


__all__ = ["Joint3DDataModule"]
