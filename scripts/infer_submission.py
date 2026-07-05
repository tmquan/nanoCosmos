#!/usr/bin/env python
"""Full-volume, chunked (blockwise) inference for challenge submission.

``scripts/infer_volume.py`` runs one bounded region through the network in a
single shot -- fine for a quick demo, but a real CREMI A+/B+/C+ (padded,
200 x 3072 x 3072 native) or even SNEMI3D AC3 (100 x 1024 x 1024 native)
volume, resampled onto the network's fine grid, is **far** too large to hold
as one tensor: e.g. CREMI's 40 nm z becomes a 10x upsample, so its full
fine-grid volume is ~2000 x 3072 x 3072 -- a single-channel accumulator alone
would need multiple TB, let alone the ``HEAD_CHANNELS``-wide one
``sliding_window_inference`` allocates.

This script instead tiles the FULL volume into GPU-sized 3-D **blocks**
(``--fine-core-size``), each with a **context** margin on every side
(``--fine-context``) so the network sees real surrounding tissue at every
block's border, not a hard zero-padded edge:

1. For each block: read the (core + context) NATIVE region, resample to the
   fine grid, run the same Gaussian-blended sliding-window inference +
   Mutex-Watershed agglomeration as ``infer_volume.py`` (affinities always at
   the fine grid; MWS runs exactly once per block, on that block's fully
   blended affinity field).
2. Crop away the context margin, keeping only the block's **core** -- this
   also removes the usual sliding-window edge artefacts at the block border.
3. Relabel the core's instance ids with a running offset so ids never collide
   between blocks.
4. Resample the (now id-relabelled) core segmentation down to **native**
   resolution (nearest-neighbour) and write it into the pre-allocated
   full-native-resolution output ``.h5`` (chunked on disk -- never held fully
   in GPU memory).
5. Once every block is written, save the challenge submission file in the
   right format for the target challenge (``--submission-format``, default
   ``auto``):
   * **CREMI** -- if the volume carries the ``cropped_region_offset_zyx`` /
     ``cropped_region_shape_zyx`` attributes (written by
     ``scripts/download_cremi3d.py --padded``), crop the full
     padded-resolution output down to exactly the region CREMI expects and
     write ``volumes/labels/neuron_ids`` (+ a ``resolution`` attribute) into
     a plain ``.hdf``.
   * **SNEMI3D** (e.g. AC3, which is not a padded download and has no such
     attributes) -- no cropping (AC3 already *is* the exact test region); the
     official format is completely different from CREMI's -- a **zip file**
     containing exactly one ``test-input.h5`` with dataset ``main`` (pixels
     with equal values = one 3-D object), per
     https://snemi3d.grand-challenge.org/.
   ``auto`` picks CREMI when the crop attributes are present, SNEMI3D
   otherwise; pass ``cremi`` / ``snemi3d`` explicitly to override.

KNOWN LIMITATION -- read before trusting a submission
-------------------------------------------------------
Mutex Watershed runs **independently per block**. The generous context
margin means most real structures are fully captured within one block's
core+context and agglomerate correctly, but there is **no cross-block
region-adjacency-graph merge**: an instance that genuinely spans a block seam
will get relabelled as two *different* ids on either side of that seam. This
is the same "blockwise inference has no cross-block merge" gap noted in
nanoCosmos's own design docs -- true seamless whole-volume merging (the
production LSD/`waterz`/`daisy`-style approach) is a separate, larger
project. Use a generous ``--fine-context`` (the default already ~half a
network patch) and a coarse ``--fine-core-size`` (fewer, bigger blocks) to
minimise how often this matters; it is not eliminated.

Examples
--------
    # CREMI A+ (padded, 40x4x4 nm) -- cropped back to the submission region:
    python scripts/infer_submission.py \\
        --config-name nanocosmos-2B --ckpt <ckpt> \\
        --vol cremi3d_sample_A+_padded_volume --root data/CREMI3D \\
        --native-resolution 40 4 4 \\
        --out-dir outputs/submission/cremi_A+

    # SNEMI3D AC3 (non-padded, 30x6x6 nm) -- identical invocation, no crop-back;
    # AC3 carries no cropped_region_* attrs so this auto-writes the SNEMI3D
    # zip format (test-input.h5 / dataset 'main') instead of CREMI's:
    python scripts/infer_submission.py \\
        --config-name nanocosmos-2B --ckpt <ckpt> \\
        --vol AC3_inputs --root data/SNEMI3D \\
        --native-resolution 30 6 6 \\
        --out-dir outputs/submission/snemi3d_AC3

    # Just plan the block grid (no inference) to estimate the run:
    python scripts/infer_submission.py ... --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import build_module  # noqa: E402
from infer_volume import (  # noqa: E402
    _get_h5_dataset,
    _norm_range,
)


# ----------------------------------------------------------------------
# Volume metadata (shape without loading data; cropped_region_* attrs)
# ----------------------------------------------------------------------

def _get_native_shape(path: Path) -> Tuple[int, int, int]:
    with h5py.File(str(path), "r", locking=False) as f:
        return tuple(int(s) for s in _get_h5_dataset(f).shape[-3:])


def _read_submission_crop_attrs(path: Path) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """``(offset_zyx, shape_zyx)`` if the volume carries the padded-CREMI
    crop-back attributes (see ``scripts/download_cremi3d.py --padded``);
    ``None`` otherwise (submit the full volume as-is)."""
    with h5py.File(str(path), "r", locking=False) as f:
        ds = _get_h5_dataset(f)
        if "cropped_region_offset_zyx" not in ds.attrs:
            return None
        offset = tuple(int(v) for v in ds.attrs["cropped_region_offset_zyx"])
        shape = tuple(int(v) for v in ds.attrs["cropped_region_shape_zyx"])
        return offset, shape


def _read_native_region(path: Path, origin: Sequence[int], size: Sequence[int]) -> np.ndarray:
    with h5py.File(str(path), "r", locking=False) as f:
        ds = _get_h5_dataset(f)
        shape = ds.shape[-3:]
        slices = tuple(
            slice(max(0, o), min(shape[d], o + size[d])) for d, o in enumerate(origin)
        )
        return np.asarray(ds[slices], dtype=np.float32)


# ----------------------------------------------------------------------
# Fine <-> native grid conversion (mirrors infer_volume._fine_grid_shape)
# ----------------------------------------------------------------------

def _native_to_fine(coord: Sequence[float], native_res: Sequence[float], fine_nm: float) -> Tuple[int, int, int]:
    return tuple(int(round(coord[d] * float(native_res[d]) / fine_nm)) for d in range(3))


def _fine_to_native(coord: Sequence[float], native_res: Sequence[float], fine_nm: float) -> Tuple[int, int, int]:
    return tuple(int(round(coord[d] * fine_nm / float(native_res[d]))) for d in range(3))


# ----------------------------------------------------------------------
# Block grid
# ----------------------------------------------------------------------

class Block:
    """One tile of the blockwise inference plan (all bounds in FINE-grid voxels)."""

    __slots__ = ("core_lo", "core_hi", "padded_lo", "padded_hi")

    def __init__(self, core_lo, core_hi, padded_lo, padded_hi):
        self.core_lo = core_lo
        self.core_hi = core_hi
        self.padded_lo = padded_lo
        self.padded_hi = padded_hi

    def __repr__(self) -> str:
        return f"Block(core={self.core_lo}-{self.core_hi}, padded={self.padded_lo}-{self.padded_hi})"


def _plan_blocks(fine_shape: Sequence[int], core_size: Sequence[int], context: Sequence[int]) -> List[Block]:
    """Tile ``fine_shape`` into non-overlapping CORE blocks of ``core_size``,
    each padded with ``context`` voxels of extra margin on every side
    (clamped to the volume bounds)."""
    n = [max(1, -(-fine_shape[d] // core_size[d])) for d in range(3)]  # ceil div
    blocks = []
    for i, j, k in itertools.product(range(n[0]), range(n[1]), range(n[2])):
        idx = (i, j, k)
        core_lo = tuple(idx[d] * core_size[d] for d in range(3))
        core_hi = tuple(min(fine_shape[d], core_lo[d] + core_size[d]) for d in range(3))
        padded_lo = tuple(max(0, core_lo[d] - context[d]) for d in range(3))
        padded_hi = tuple(min(fine_shape[d], core_hi[d] + context[d]) for d in range(3))
        blocks.append(Block(core_lo, core_hi, padded_lo, padded_hi))
    return blocks


# ----------------------------------------------------------------------
# Per-block inference
# ----------------------------------------------------------------------

def _run_block(
    module, block: Block, vol_path: Path, native_resolution: Sequence[float], fine_nm: float,
    vmin: float, vmax: float, patch_size: Sequence[int], device: torch.device, sem_threshold: float,
) -> Optional[np.ndarray]:
    """Run inference on one block; return its CORE instance segmentation at
    the fine grid (ids start at 1, local to this block -- the caller
    relabels), or ``None`` if the block's native region is empty/degenerate."""
    from nanocosmos.losses import slice_head
    from nanocosmos.inference.sliding_window import sliding_window_inference

    padded_lo, padded_hi = block.padded_lo, block.padded_hi
    padded_fine_size = tuple(padded_hi[d] - padded_lo[d] for d in range(3))
    if min(padded_fine_size) <= 0:
        return None

    native_lo = _fine_to_native(padded_lo, native_resolution, fine_nm)
    native_hi = _fine_to_native(padded_hi, native_resolution, fine_nm)
    native_size = tuple(max(1, native_hi[d] - native_lo[d]) for d in range(3))
    region = _read_native_region(vol_path, native_lo, native_size)
    if region.size == 0:
        return None

    image01 = np.clip((region - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    t = torch.from_numpy(image01)[None, None]
    fine_image = F.interpolate(t, size=padded_fine_size, mode="trilinear", align_corners=False)[0]

    stride = tuple(max(1, p // 2) for p in patch_size)
    head = sliding_window_inference(
        module, fine_image, patch_size=patch_size, stride=stride,
        aggregation="gaussian", batch_size=1, device=device, progress=False,
    )
    fields = slice_head(head[None])
    sem = fields["sem"].sigmoid()
    aff = fields["aff"].sigmoid().float()
    sem_fg = (sem[:, 0] > sem_threshold) if getattr(module.agglomerator, "gate_with_sem", True) else None
    seg_padded = module.agglomerator(aff, sem_fg)[0]  # [D, H, W], local to this padded block

    # Crop away the context margin -> keep only this block's CORE.
    off = tuple(block.core_lo[d] - padded_lo[d] for d in range(3))
    sz = tuple(block.core_hi[d] - block.core_lo[d] for d in range(3))
    seg_core = seg_padded[off[0]:off[0] + sz[0], off[1]:off[1] + sz[1], off[2]:off[2] + sz[2]]
    return seg_core.cpu().numpy().astype(np.int64)


# ----------------------------------------------------------------------
# Core pipeline
# ----------------------------------------------------------------------

def infer_submission(
    cfg,
    ckpt_path: str,
    vol_path: Path,
    native_resolution: Sequence[float],
    out_dir: Path,
    fine_core_size: Sequence[int],
    fine_context: Sequence[int],
    device: str = "cuda",
    sem_threshold: float = 0.5,
    dry_run: bool = False,
    submission_format: str = "auto",
) -> Optional[Path]:
    """Run blockwise inference over the full volume and write a challenge
    submission file.

    Args:
        submission_format: ``"auto"`` (default) picks the format from whether
            the volume carries ``cropped_region_*`` attrs (CREMI padded
            downloads only -- see ``scripts/download_cremi3d.py --padded``):
            present -> ``"cremi"`` (``volumes/labels/neuron_ids`` HDF5,
            cropped back to the official region); absent -> ``"snemi3d"``
            (a ZIP containing ``test-input.h5`` with dataset ``main``, no
            cropping -- matches https://snemi3d.grand-challenge.org/). Pass
            ``"cremi"`` / ``"snemi3d"`` explicitly to override the detection.
    """
    fine_nm = float(cfg.data.pixel_size[0])
    patch_size = tuple(int(s) for s in cfg.data.patch_size)

    native_shape = _get_native_shape(vol_path)
    fine_shape = _native_to_fine(native_shape, native_resolution, fine_nm)
    blocks = _plan_blocks(fine_shape, fine_core_size, fine_context)
    max_padded = tuple(max(b.padded_hi[d] - b.padded_lo[d] for b in blocks) for d in range(3))
    acc_gb = np.prod(max_padded) * 32 * 4 / 1e9  # HEAD_CHANNELS=32, float32

    print(f"Volume: {vol_path.name}  native shape {native_shape} @ {tuple(native_resolution)} nm")
    print(f"Fine grid: {fine_shape} @ {fine_nm} nm  ->  {len(blocks)} blocks "
          f"(core {tuple(fine_core_size)}, context {tuple(fine_context)})")
    print(f"Largest per-block accumulator: ~{acc_gb:.1f} GB (HEAD_CHANNELS x float32)")
    if dry_run:
        for b in blocks:
            print(f"  {b}")
        return None

    region0 = _read_native_region(vol_path, (0, 0, 0), native_shape)
    vmin, vmax = _norm_range(vol_path, region0)
    del region0

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    module = build_module(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    module.to(device_t).eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"{vol_path.stem}_pred_label_native_full.h5"
    with h5py.File(str(full_path), "w") as f:
        full_ds = f.create_dataset(
            "main", shape=native_shape, dtype=np.int64,
            chunks=tuple(min(s, 256) for s in native_shape), compression="gzip", compression_opts=4,
        )
        next_id = 1
        t0 = time.time()
        for bi, block in enumerate(blocks):
            seg_core_fine = _run_block(
                module, block, vol_path, native_resolution, fine_nm,
                vmin, vmax, patch_size, device_t, sem_threshold,
            )
            elapsed = time.time() - t0
            if seg_core_fine is None or not seg_core_fine.any():
                print(f"  block {bi + 1}/{len(blocks)}  {block}  (empty, skipped)  [{elapsed:.0f}s]")
                continue
            n_local = int(seg_core_fine.max())
            seg_core_fine = np.where(seg_core_fine > 0, seg_core_fine + next_id - 1, 0)
            next_id += n_local

            native_lo = _fine_to_native(block.core_lo, native_resolution, fine_nm)
            native_hi = _fine_to_native(block.core_hi, native_resolution, fine_nm)
            native_size = tuple(max(1, native_hi[d] - native_lo[d]) for d in range(3))
            seg_core_native = F.interpolate(
                torch.from_numpy(seg_core_fine)[None, None].float(), size=native_size, mode="nearest",
            )[0, 0].numpy().astype(np.int64)

            sl = tuple(slice(native_lo[d], native_lo[d] + native_size[d]) for d in range(3))
            full_ds[sl] = seg_core_native
            print(f"  block {bi + 1}/{len(blocks)}  {block}  {n_local} local ids  [{elapsed:.0f}s]")
        full_ds.attrs["resolution_zyx_nm"] = np.asarray(native_resolution, dtype=np.float64)
        full_ds.attrs["source"] = (
            f"nanocosmos blockwise inference from {ckpt_path}; MWS per block "
            f"(core {tuple(fine_core_size)}, context {tuple(fine_context)} @ {fine_nm} nm); "
            "see infer_submission.py docstring for the cross-block-merge limitation."
        )
        print(f"Full native-resolution segmentation ({next_id - 1} total ids) -> {full_path}")

    # ---- crop back to the official submission region, if applicable ----
    crop = _read_submission_crop_attrs(vol_path)
    fmt = submission_format
    if fmt == "auto":
        fmt = "cremi" if crop is not None else "snemi3d"
    print(f"Submission format: {fmt}"
          f"{' (auto-detected from cropped_region_* attrs)' if submission_format == 'auto' else ''}")

    if fmt == "snemi3d":
        # SNEMI3D / Grand Challenge format: a ZIP containing exactly one file
        # named ``test-input.h5`` with a dataset ``main`` -- NOT the CREMI
        # ``volumes/labels/neuron_ids`` layout. AC3 is the test volume as-is
        # (not a padded download), so no crop-back is applied here.
        if crop is not None:
            print("  (note: this volume DOES carry cropped_region_* attrs but "
                  "--submission-format snemi3d was requested/auto-detected as "
                  "not applicable -- using the full volume, uncropped, as SNEMI3D expects.)")
        import zipfile

        h5_name = "test-input.h5"
        tmp_h5 = out_dir / h5_name
        with h5py.File(str(full_path), "r") as fsrc, h5py.File(str(tmp_h5), "w") as fdst:
            fdst.create_dataset("main", data=fsrc["main"][:].astype(np.uint32))
        submission_path = out_dir / f"{vol_path.stem}_submission.zip"
        with zipfile.ZipFile(str(submission_path), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(str(tmp_h5), arcname=h5_name)
        tmp_h5.unlink()
        print(f"Submission zip written: {submission_path}  (contains {h5_name}, dataset 'main')")
        return submission_path

    if crop is None:
        print("No cropped_region_* attrs on this volume -- submitting the full volume as-is "
              "in CREMI format (volumes/labels/neuron_ids).")
        submission_path = out_dir / f"{vol_path.stem}_submission.hdf"
        with h5py.File(str(full_path), "r") as fsrc, h5py.File(str(submission_path), "w") as fdst:
            data = fsrc["main"][:]
            ds = fdst.create_dataset("volumes/labels/neuron_ids", data=data.astype(np.uint64))
            ds.attrs["resolution"] = np.asarray(native_resolution, dtype=np.float64)
    else:
        offset, shape = crop
        print(f"Cropping to the official submission region: offset={offset} shape={shape}")
        with h5py.File(str(full_path), "r") as f:
            sl = tuple(slice(offset[d], offset[d] + shape[d]) for d in range(3))
            cropped = f["main"][sl]
        submission_path = out_dir / f"{vol_path.stem.replace('_padded', '')}_submission.hdf"
        with h5py.File(str(submission_path), "w") as fdst:
            ds = fdst.create_dataset("volumes/labels/neuron_ids", data=cropped.astype(np.uint64))
            ds.attrs["resolution"] = np.asarray(native_resolution, dtype=np.float64)

    n_ids = int(len(np.unique(h5py.File(str(submission_path), "r")["volumes/labels/neuron_ids"])))
    print(f"Submission file written: {submission_path}  ({n_ids} unique ids incl. background)")
    return submission_path


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-name", default="nanocosmos-2B")
    p.add_argument("--overrides", nargs="*", default=[])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vol", required=True, help="Volume basename (without _volume.h5).")
    p.add_argument("--root", required=True)
    p.add_argument("--native-resolution", type=float, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--fine-core-size", type=int, nargs=3, default=[800, 512, 512], metavar=("D", "H", "W"),
        help="Per-block CORE size in fine-grid voxels (default: 2x the network patch).",
    )
    p.add_argument(
        "--fine-context", type=int, nargs=3, default=[200, 128, 128], metavar=("D", "H", "W"),
        help="Extra context margin per block, each side, in fine-grid voxels "
             "(default: half the network patch).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--sem-threshold", type=float, default=0.5)
    p.add_argument(
        "--submission-format", choices=("auto", "cremi", "snemi3d"), default="auto",
        help="'auto' (default) picks CREMI format (volumes/labels/neuron_ids, "
             "cropped to the official region) if the volume carries "
             "cropped_region_* attrs (padded CREMI downloads), else SNEMI3D "
             "format (a zip containing test-input.h5, dataset 'main', uncropped "
             "-- see https://snemi3d.grand-challenge.org/). Override explicitly "
             "with 'cremi' / 'snemi3d' if needed.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the block plan (count, sizes, accumulator memory estimate) and exit -- no inference.",
    )
    return p.parse_args()


def _load_cfg(config_name: str, overrides):
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.config_name, args.overrides)
    vol_path = Path(args.root) / f"{args.vol}.h5"
    if not vol_path.exists():
        raise SystemExit(f"Volume not found: {vol_path}")

    infer_submission(
        cfg,
        ckpt_path=args.ckpt,
        vol_path=vol_path,
        native_resolution=args.native_resolution,
        out_dir=Path(args.out_dir),
        fine_core_size=args.fine_core_size,
        fine_context=args.fine_context,
        device=args.device,
        sem_threshold=args.sem_threshold,
        dry_run=args.dry_run,
        submission_format=args.submission_format,
    )


if __name__ == "__main__":
    main()
