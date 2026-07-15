"""
Multi-task datamodule for the joint reconstruction + segmentation recipe.

Feeds :class:`~nanocosmos.modules.Joint3DModule` the batch contract it expects
(see doc/JOINT_TRAINING.md): ``task`` (``"ssl"`` | ``"sft"``), the fine-grid
``image``, the native-grid ``label`` (sft), and the ``recon_image`` target.

Geometry (see doc/RESOLUTION_LADDER.md).  The network runs on a fixed fine
grid (``patch_size`` @ ``pixel_size`` nm, e.g. [200, 256, 256] @ 4 nm).  Each
volume is read at its **native** voxel size over the same physical field of
view, then :class:`~nanocosmos.transforms.ToFineGridd` resamples the image onto
the fine grid while the label stays native and the recon target sits on the
coarser of native/fine.

Branches & batching.  Volumes are bucketed into round-robin groups per the
``balance`` policy -- ``"resolution"`` (one group per ``(task, native_resolution)``;
legacy default), ``"subset"`` (one group per domain), or ``"volume"`` (one group
per volume; each volume scheduled equally).  Every group is single-task and
single-resolution, so a batch drawn from one group is always **task- and
shape-homogeneous**, which is exactly what ``Joint3DReconSegLoss`` /
``Joint3DModule`` require.  A round-robin batch sampler interleaves groups,
weighted by each branch's ``sample_weight`` and (for subset/volume balance) by
``subset_weights`` / per-volume ``sample_weight``.

Config schema (``cfg.data``)::

    patch_size: [200, 256, 256]      # fine grid (z, y, x)
    pixel_size: [4, 4, 4]            # fine voxel size nm (cubic)
    degrade: {zf_range: [...], ...}  # RandResolutionDegraded kwargs (ssl)
    sft_min_foreground: 0.8          # sft gate: BOTH label fg AND image non-zero
    ssl_min_foreground: 0.8          # image non-zero gate (ssl volumes)
    ssl_min_std: 0.05                # ssl contrast gate (reject flat resin crops)
    balance: resolution              # group/schedule unit: "resolution" (default)
                                     #   | "subset" | "volume"
    subset_weights: {cremi3d: 50}    # (balance: subset) per-subset schedule weight
    find_boundaries: 1.0             # per-sample boundary-erosion probability
                                     #   (applied to filled / unset sft volumes)
    boundary_target: semantic        # "semantic" (sem_label only) | "both"
    branches:
      ssl: {batch_size, sample_weight, volumes: [{vol, root, native_resolution}]}
      sft:  {batch_size, sample_weight, volumes: [
               {vol, seg, root, native_resolution, label_convention?}]}
      #   label_convention (optional, sft): ``gapped`` | ``filled``
      #     gapped -- membranes / extracellular already label==0; skip
      #               FindBoundariesd (gaps already present).
      #     filled -- space-filling / abutting instances; apply
      #               FindBoundariesd so the sem head sees thin gaps.
      #   per-volume ``sample_weight`` (optional) scales its share when
      #   balance: volume (e.g. overfit one volume).
    val_volumes: [{vol, seg, root, task, native_resolution, label_convention?}]
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
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

from nanocosmos.datamodules.condition_text import (  # noqa: E402
    LABEL_CONVENTIONS,
    prompt_from_volume_spec,
    validate_label_convention,
)

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
    """Round-robin multi-task datamodule for the joint recipe.

    Note: unlike the single-dataset datamodules, this one intentionally does
    NOT call ``save_hyperparameters()`` -- the joint data config (branches,
    degrade schema, crop gates) is reproduced from the Hydra config snapshot
    saved under the run directory rather than from checkpoint
    ``datamodule_hyper_parameters``. Model checkpoints are unaffected either way.
    """

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
        sft_min_foreground: float = 0.0,
        sft_min_instances: int = 0,
        sft_max_inst_frac: float = 0.0,
        ssl_min_foreground: float = 0.0,
        ssl_min_std: float = 0.0,
        ssl_min_autocorr: float = 0.0,
        find_boundaries: float = 0.0,
        boundary_target: str = "semantic",
        balance: str = "resolution",
        subset_weights: Optional[Dict[str, float]] = None,
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
        # sft crop gate: require BOTH label foreground >= t AND image non-zero >= t
        # (rejects background-heavy / unlabeled crops and zero-padded EM).  Falls
        # back to the legacy label-only ``min_foreground`` when unset.
        self.sft_min_foreground = float(sft_min_foreground) or float(min_foreground)
        # Instance-diversity gate for the sft branch: ``sft_min_foreground`` only
        # checks the label's NON-ZERO fraction, so a crop entirely filled by ONE
        # giant instance (e.g. a single MICrONS dendrite trunk spanning the whole
        # patch) passes trivially yet gives AffinityFGLoss no inter-instance
        # "push" signal -- only "pull".  ``sft_min_instances`` rejects crops with
        # fewer than N distinct nonzero instance ids; ``sft_max_inst_frac``
        # rejects crops where the single largest instance exceeds this fraction
        # of the labeled foreground.  Both 0 = disabled (default).
        self.sft_min_instances = int(sft_min_instances)
        self.sft_max_inst_frac = float(sft_max_inst_frac)
        # Image-nonzero gate for the label-less ssl branch: rejects mostly-empty /
        # zero-padded crops (e.g. MitoEM2 nnU-Net irregular crops padded to a box,
        # empty COSEM/FLYEM regions) that would otherwise show as black panels.
        self.ssl_min_foreground = float(ssl_min_foreground)
        # Contrast gate for the ssl branch: rejects flat, structureless crops
        # (e.g. resin / embedding medium around a COSEM cell) whose normalised
        # intensity std is below this value.  The non-zero gate cannot catch
        # these because resin is a non-zero mid-grey; only a variance test does.
        # 0 = disabled.
        self.ssl_min_std = float(ssl_min_std)
        # Structure gate for the ssl branch: reject NOISE-dominated crops
        # (detector / resin grain) that pass the std/content gate because random
        # noise has high local variance.  Lag-1 spatial autocorrelation on the
        # per-volume [0, 1] scale: real ultrastructure ~0.6-0.8, white noise ~0.
        # 0 = disabled.
        self.ssl_min_autocorr = float(ssl_min_autocorr)
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
        # How volumes are bucketed into round-robin groups (= the unit that is
        # scheduled equally per epoch).  Each batch is drawn from ONE group, so
        # every mode still yields task- and shape-homogeneous batches.
        #   "resolution" -- one group per (task, native_resolution) [default,
        #                   legacy].  Within a group, volumes are picked
        #                   voxel-count-weighted, so big volumes dominate.
        #   "subset"     -- one group per (task, subset); every subset is
        #                   scheduled equally (x ``subset_weights``).  Balances
        #                   domains regardless of size / crop count.
        #   "volume"     -- one group per volume; every volume is scheduled
        #                   equally (x its ``sample_weight``).  A batch is then
        #                   always drawn from a single volume.
        # ``subset_weights`` / per-volume ``sample_weight`` multiply a group's
        # schedule length, so a subset/volume can be deliberately over-sampled
        # (e.g. to overfit CREMI / SNEMI for a segmentation sanity check).
        self.balance = str(balance)
        if self.balance not in ("resolution", "subset", "volume"):
            raise ValueError(
                "balance must be 'resolution', 'subset' or 'volume'; "
                f"got {balance!r}."
            )
        self.subset_weights = {str(k): float(v) for k, v in (subset_weights or {}).items()}
        self.seed = int(seed)

        self._train_groups: List[Tuple[int, int, int]] = []
        self._val_groups: List[Tuple[int, int, int]] = []
        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None

    # ------------------------------------------------------------------
    # Per-group transform pipelines
    # ------------------------------------------------------------------

    def _group_transform(
        self,
        task: str,
        native_res: Sequence[float],
        *,
        label_convention: Optional[str] = None,
    ) -> Compose:
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
        # applies per dataset.  ``gapped`` volumes already have membrane /
        # extracellular as label==0, so FindBoundariesd is skipped for them.
        sft_tf: List[Any] = [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Labeld(keys=["label"], spatial_dims=3),
        ]
        out_keys = ["image", "label", "recon_image"]
        apply_boundaries = (
            self.find_boundaries > 0 and label_convention != "gapped"
        )
        if apply_boundaries:
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
        *,
        label_convention: Optional[str] = None,
    ) -> LazyVolDataset:
        native_patch = _scaled(self.fine_patch, self.fine_nm, native_res)
        # Attach fixed text-condition prompts (imaging · z/y/x nm · tail).
        lazy_vols: List[Dict[str, Any]] = []
        for vol in volumes:
            entry = {k: v for k, v in vol.items() if k in ("vol", "seg", "root")}
            try:
                entry["prompt"] = prompt_from_volume_spec(vol, task=task)
            except ValueError as exc:
                # Keep backward-compat for configs without imaging: (2B/16B).
                logger.debug("Skipping prompt for %s: %s", vol.get("vol"), exc)
            lazy_vols.append(entry)
        return LazyVolDataset(
            root_dir=self.data_root,
            volumes=lazy_vols,
            patch_size=native_patch,
            transform=self._group_transform(
                task, native_res, label_convention=label_convention,
            ),
            num_samples=num_samples,
            # sft: gate on BOTH label and image at sft_min_foreground.
            # ssl: image-only gate (label-less) at ssl_min_foreground.
            min_foreground=(self.sft_min_foreground if task == "sft" else 0.0),
            image_min_foreground=(
                self.sft_min_foreground if task == "sft"
                else (self.ssl_min_foreground if task == "ssl" else 0.0)
            ),
            # Contrast + structure gates only for the ssl branch (label-less recon).
            image_min_std=(self.ssl_min_std if task == "ssl" else 0.0),
            image_min_autocorr=(self.ssl_min_autocorr if task == "ssl" else 0.0),
            # Instance-diversity gates only for the sft branch (labeled).
            min_instances=(self.sft_min_instances if task == "sft" else 0),
            max_inst_frac=(self.sft_max_inst_frac if task == "sft" else 0.0),
            deterministic=deterministic,
        )

    @staticmethod
    def _group_by_res(
        volumes: List[Dict[str, Any]],
    ) -> Dict[Tuple[float, ...], List[Dict[str, Any]]]:
        groups: Dict[Tuple[float, ...], List[Dict[str, Any]]] = {}
        for vol in volumes:
            res = tuple(float(r) for r in vol["native_resolution"])
            groups.setdefault(res, []).append(vol)
        return groups

    @staticmethod
    def _group_by_res_and_convention(
        volumes: List[Dict[str, Any]],
    ) -> Dict[Tuple[Tuple[float, ...], Optional[str]], List[Dict[str, Any]]]:
        """Bucket by ``(native_resolution, label_convention)``.

        ``label_convention`` is validated when present; missing -> ``None``
        (legacy configs).  Mixing ``gapped`` and ``filled`` at the same
        resolution must not share a transform (FindBoundariesd gating).
        """
        groups: Dict[
            Tuple[Tuple[float, ...], Optional[str]], List[Dict[str, Any]]
        ] = {}
        for vol in volumes:
            res = tuple(float(r) for r in vol["native_resolution"])
            conv = validate_label_convention(
                vol.get("label_convention"), vol=str(vol.get("vol")),
            )
            groups.setdefault((res, conv), []).append(vol)
        return groups

    @staticmethod
    def _derive_subset(vol_name: str) -> str:
        """Best-effort subset (domain) name from a volume stem.

        Covers the on-disk naming conventions in ``doc/DATASETS.md``.  A
        volume may override this with an explicit ``subset:`` field in its
        config spec.
        """
        n = vol_name
        for suf in ("_volume", "_segmentation"):
            if n.endswith(suf):
                n = n[: -len(suf)]
        # SNEMI3D challenge volumes: train_*/test_* stems
        if n.startswith("train_"):
            return "train"
        if n.startswith("test_"):
            return "test"
        # Neurons / neurite cylinder
        if n.startswith("neurons") or n.startswith("neurite"):
            return "neurons"
        # MICrONS
        if n.startswith("minnie65"):
            return "minnie65"
        # FlyWire FAFB v783
        if n.startswith("flywire_"):
            return "flywire"
        # CREMI: cremi3d_sample_A / A+ -> cremi3d (train) / cremi3d_test (image-only)
        m = re.match(r"^(cremi3d)_sample_[A-Za-z]+(\+?)", n)
        if m:
            return "cremi3d_test" if m.group(2) == "+" else "cremi3d"
        # FLYEM3D members (hemibrain / malecns carry the name; rest is FIB-25)
        if n.startswith("flyem3d_hemibrain"):
            return "flyem3d_hemibrain"
        if n.startswith("flyem3d_malecns"):
            return "flyem3d_malecns"
        if re.match(r"^flyem3d_\d+nm_ssl", n):
            return "flyem3d_fib25_ssl"
        if n.startswith("flyem3d"):
            return "flyem3d_fib25"
        # COSEM: jrc_<id>_<rx>x<ry>x<rz>nm_... -> jrc_<id>
        m = re.match(r"^(jrc_[A-Za-z0-9\-]+?)_\d+x\d", n)
        if m:
            return m.group(1)
        # MitoEM2: mitoem2_<subset>_train01 / test01
        m = re.match(r"^(mitoem2_[a-z]+)_(?:train|test)\d+", n)
        if m:
            return m.group(1)
        # Fallback: drop a trailing _x{X}_y{Y}_z{Z} coordinate block.
        m = re.match(r"^(.*?)_x\d+_y\d+_z\d+", n)
        return m.group(1) if m else n

    def _iter_train_groups(
        self, task: str, vols: List[Dict[str, Any]],
    ) -> List[Tuple[str, Tuple[float, ...], List[Dict[str, Any]], float, Optional[str]]]:
        """Bucket a branch's volumes into ``(desc, res, vols, multiplier, convention)``
        round-robin groups per the ``balance`` policy.  ``multiplier`` scales
        the group's schedule length (1.0 = the branch's base share)."""
        out: List[
            Tuple[str, Tuple[float, ...], List[Dict[str, Any]], float, Optional[str]]
        ] = []
        if self.balance == "resolution":
            for (res, conv), gvols in self._group_by_res_and_convention(vols).items():
                tag = f"res={res}" + (f" conv={conv}" if conv else "")
                out.append((tag, res, gvols, 1.0, conv))
        elif self.balance == "subset":
            keyed: Dict[
                Tuple[str, Tuple[float, ...], Optional[str]], List[Dict[str, Any]]
            ] = {}
            for v in vols:
                sub = str(v.get("subset") or self._derive_subset(v["vol"]))
                res = tuple(float(r) for r in v["native_resolution"])
                conv = validate_label_convention(
                    v.get("label_convention"), vol=str(v.get("vol")),
                )
                keyed.setdefault((sub, res, conv), []).append(v)
            for (sub, res, conv), gvols in keyed.items():
                mult = self.subset_weights.get(sub, 1.0)
                tag = f"subset={sub}" + (f" conv={conv}" if conv else "")
                out.append((tag, res, gvols, mult, conv))
        else:  # "volume": one group per volume, equally likely (x sample_weight)
            for v in vols:
                res = tuple(float(r) for r in v["native_resolution"])
                conv = validate_label_convention(
                    v.get("label_convention"), vol=str(v.get("vol")),
                )
                mult = float(v.get("sample_weight", 1.0))
                out.append((f"vol={v['vol']}", res, [v], mult, conv))
        return out

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
            branch_groups = self._iter_train_groups(task, vols)
            # Normalise multipliers by the branch mean so a group's length is
            # ``num_samples * sample_weight`` when all weights are equal -- i.e.
            # weights are RELATIVE ratios, and the branch's total virtual-epoch
            # budget stays fixed regardless of how the weights are set (a high
            # ``subset_weights`` simply reallocates the budget toward that
            # subset rather than inflating the epoch).
            mults = [m for _, _, _, m, _ in branch_groups]
            mean_mult = (sum(mults) / len(mults)) if mults else 1.0
            if mean_mult <= 0:
                mean_mult = 1.0
            for desc, res, gvols, mult, conv in branch_groups:
                n_group = max(bs, int(round(self.num_samples * weight * mult / mean_mult)))
                try:
                    ds = self._build_group(
                        task, res, gvols, n_group, deterministic=False,
                        label_convention=conv,
                    )
                except ValueError:
                    # All volumes in the group were skipped (e.g. smaller than
                    # the native patch on some axis).  Skip the empty group
                    # rather than aborting the whole run.
                    logger.warning(
                        "Joint train group skipped (no usable volumes): "
                        "task=%s %s", task, desc,
                    )
                    continue
                datasets.append(ds)
                group_specs.append((offset, len(ds), bs))
                offset += len(ds)
                logger.info(
                    "Joint train group: task=%s %s vols=%d native_patch=%s "
                    "len=%d bs=%d mult=%.2f", task, desc, len(gvols),
                    _scaled(self.fine_patch, self.fine_nm, res), len(ds), bs, mult,
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
            for (res, conv), gvols in self._group_by_res_and_convention(vols).items():
                ds = self._build_group(
                    task, res, gvols, self.val_num_samples, deterministic=True,
                    label_convention=conv,
                )
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
