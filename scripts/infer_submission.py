#!/usr/bin/env python
"""Full-volume, chunked (blockwise) inference for challenge submission.

``scripts/infer_volume.py`` runs one bounded region through the network in a
single shot -- fine for a quick demo, but a real CREMI A+/B+/C+ (padded,
200 x 3072 x 3072 native) or even SNEMI3D AC3 (100 x 1024 x 1024 native)
volume, resampled onto the network's fine grid, is **far** too large to hold
as one tensor.

This script runs inference in two phases:

Phase A -- Gaussian-weighted blending of raw sem+aff logits
-------------------------------------------------------------
The FULL fine-grid volume is tiled with heavily-**overlapping** windows, each
exactly ``--window-size`` (defaults to the network's native ``patch_size``,
e.g. 400 x 256 x 256 @ 4 nm), advancing by a stride of
``--stride-frac * window-size`` (default 0.25, i.e. 75% overlap between
neighbouring windows -- this is the same idea as
``nanocosmos/inference/sliding_window.py``'s per-patch Gaussian blending,
just applied at the whole-volume scale). For every window: run the network
once (a single forward pass, since the window equals the native patch), take
the raw (pre-sigmoid) affinity + semantic channels (drop the raw-recon
channel -- not needed for segmentation), multiply by a 3-D Gaussian weight
(peak at the window center, tapering to the edges), and accumulate both the
weighted logits and the weight itself into an on-disk HDF5 accumulator
covering the whole (padded) fine grid. If the real volume doesn't divide
evenly into an integer number of window strides, the fine grid is rounded UP
to the next multiple and the extra region is zero-padded (never a smaller/
ragged window) -- see ``_read_native_region``.

Once every window has been accumulated, the blended field at each voxel is
``sum(weight_i * logit_i) / sum(weight_i)`` -- exactly the weighted average
in the diagram (final normalised by total participating weight), done lazily
per chunk in Phase B rather than as a separate whole-volume pass.

Phase B -- Mutex Watershed on the blended field
-------------------------------------------------
Mutex Watershed needs a fully-formed local affinity neighbourhood, and is
itself memory-hungry, so it still runs **chunked**: the blended field is
tiled into ``--fine-core-size`` CORE chunks with ``--fine-context`` margin
(no network inference here -- this only reads the already-blended field, so
these chunks can be considerably larger than Phase A's window without OOMing
on model activations). Per chunk: sigmoid the blended logits, run MWS once,
crop away the context margin (keep only the CORE), relabel with a running id
offset so ids never collide between chunks, resample down to **native**
resolution (nearest-neighbour) and write into the pre-allocated full-native
output ``.h5`` (chunked on disk -- never held fully in memory).

Once every chunk is written, save the challenge submission file in the right
format for the target challenge (``--submission-format``, default ``auto``):
  * **CREMI** -- if the volume carries the ``cropped_region_offset_zyx`` /
    ``cropped_region_shape_zyx`` attributes (written by
    ``scripts/download_cremi3d.py --padded``), crop the full padded-
    resolution output down to exactly the region CREMI expects and write
    ``volumes/labels/neuron_ids`` (+ a ``resolution`` attribute) into a
    plain ``.hdf``.
  * **SNEMI3D** (e.g. AC3, which is not a padded download and has no such
    attributes) -- no cropping (AC3 already *is* the exact test region); the
    official format is a **zip file** containing exactly one
    ``test-input.h5`` with dataset ``main``, per
    https://snemi3d.grand-challenge.org/.
  ``auto`` picks CREMI when the crop attributes are present, SNEMI3D
  otherwise; pass ``cremi`` / ``snemi3d`` explicitly to override.

KNOWN LIMITATION -- read before trusting a submission
-------------------------------------------------------
Phase A's Gaussian blending removes seam artefacts from the AFFINITY field
itself (neighbouring windows heavily overlap and agree), but Mutex Watershed
in Phase B still runs **independently per chunk** on that blended field --
there is **no cross-chunk region-adjacency-graph merge**: an instance that
genuinely spans a chunk seam will get relabelled as two *different* ids on
either side. Use a generous ``--fine-context`` and a coarse
``--fine-core-size`` (fewer, bigger MWS chunks -- cheap now that Phase B does
no network inference) to minimise how often this matters; it is not
eliminated. True seamless whole-volume merging (the production
LSD/`waterz`/`daisy`-style approach) is a separate, larger project.

DISK / I/O COST -- read before running on a big volume
---------------------------------------------------------
The Phase A accumulator holds ``(N_AFF + 1)`` float32 channels over the
**whole padded fine grid**, e.g. for SNEMI3D AC3 (fine grid ~800x1536x1536)
with the default 30-offset model that's ~230 GB on disk (plus ~7.5 GB for
the weight map); a heavily-overlapping stride (small ``--stride-frac``)
multiplies both the number of windows and the read-modify-write I/O against
that accumulator accordingly. The printed block-plan (also shown by
``--dry-run``) reports the exact numbers before you commit to a run --
increase ``--stride-frac`` (less overlap) or shrink ``--window-size`` if
it's impractical for your storage.

Examples
--------
    # CREMI A+ (padded, 40x4x4 nm) -- cropped back to the submission region:
    python scripts/infer_submission.py \\
        --config-name nanocosmos-2B --ckpt <ckpt> \\
        --vol cremi3d_sample_A+_padded_volume --root data/CREMI3D \\
        --native-resolution 40 4 4 \\
        --out-dir outputs/submission/cremi_A+

    # SNEMI3D AC3 (non-padded, 30x6x6 nm), full-field-of-view windows
    # (400x256x256 @ 4nm, matching the network's native patch) with 1/4-
    # stride (75%) overlap between windows -- identical invocation otherwise;
    # AC3 carries no cropped_region_* attrs so this auto-writes the SNEMI3D
    # zip format (test-input.h5 / dataset 'main') instead of CREMI's:
    python scripts/infer_submission.py \\
        --config-name nanocosmos-2B --ckpt <ckpt> \\
        --vol AC3_inputs --root data/SNEMI3D \\
        --native-resolution 30 6 6 \\
        --window-size 400 256 256 --stride-frac 0.25 \\
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
    """Read a native-space region of exactly ``size`` starting at ``origin``.

    ``origin`` may be negative and/or ``origin + size`` may extend past the
    volume -- out-of-bounds voxels are zero-padded so the returned array
    always has the requested ``size`` (never silently shrunk, which would
    otherwise get stretched to the wrong scale by the caller's resample)."""
    with h5py.File(str(path), "r", locking=False) as f:
        ds = _get_h5_dataset(f)
        shape = ds.shape[-3:]
        if all(o == 0 for o in origin) and tuple(size) == tuple(shape):
            return np.asarray(ds[...], dtype=np.float32)
        src_lo = tuple(max(0, o) for o in origin)
        src_hi = tuple(max(src_lo[d], min(shape[d], origin[d] + size[d])) for d in range(3))
        out = np.zeros(tuple(size), dtype=np.float32)
        if any(src_hi[d] <= src_lo[d] for d in range(3)):
            return out
        dst_lo = tuple(src_lo[d] - origin[d] for d in range(3))
        dst_hi = tuple(dst_lo[d] + (src_hi[d] - src_lo[d]) for d in range(3))
        src_slices = tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
        dst_slices = tuple(slice(dst_lo[d], dst_hi[d]) for d in range(3))
        out[dst_slices] = np.asarray(ds[src_slices], dtype=np.float32)
        return out


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
    """One tile of a blockwise-inference plan (all bounds in FINE-grid
    voxels). ``padded_lo``/``padded_hi`` may lie outside ``[0, grid_shape)``
    at the volume's edges -- callers must clip (see ``_clip_window``)."""

    __slots__ = ("core_lo", "core_hi", "padded_lo", "padded_hi")

    def __init__(self, core_lo, core_hi, padded_lo, padded_hi):
        self.core_lo = core_lo
        self.core_hi = core_hi
        self.padded_lo = padded_lo
        self.padded_hi = padded_hi

    def __repr__(self) -> str:
        return f"Block(core={self.core_lo}-{self.core_hi}, padded={self.padded_lo}-{self.padded_hi})"


def _plan_blocks(
    grid_shape: Sequence[int],
    core_size: Sequence[int],
    context_lo: Sequence[int],
    context_hi: Sequence[int],
) -> List[Block]:
    """Tile ``grid_shape`` into CORE blocks of ``core_size`` (the last block
    along an axis is clamped/smaller if ``core_size`` doesn't divide evenly),
    each surrounded by ``context_lo``/``context_hi`` voxels of margin on
    every side (may extend past ``[0, grid_shape)`` -- the caller zero-pads,
    see ``_clip_window``). When ``grid_shape`` IS an exact multiple of
    ``core_size`` (as arranged by ``infer_submission`` for Phase A), every
    block's core -- and hence its full padded window -- is exactly the same
    constant size, including at the edges."""
    n = [max(1, -(-grid_shape[d] // core_size[d])) for d in range(3)]  # ceil div
    blocks = []
    for i, j, k in itertools.product(range(n[0]), range(n[1]), range(n[2])):
        idx = (i, j, k)
        core_lo = tuple(idx[d] * core_size[d] for d in range(3))
        core_hi = tuple(min(grid_shape[d], core_lo[d] + core_size[d]) for d in range(3))
        padded_lo = tuple(core_lo[d] - context_lo[d] for d in range(3))
        padded_hi = tuple(core_hi[d] + context_hi[d] for d in range(3))
        blocks.append(Block(core_lo, core_hi, padded_lo, padded_hi))
    return blocks


def _clip_window(
    lo: Sequence[int], hi: Sequence[int], grid_shape: Sequence[int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """Clip the window ``[lo, hi)`` to the valid ``[0, grid_shape)`` range.

    Returns ``(clipped_lo, clipped_hi, src_lo, src_hi)``: ``clipped_lo/hi``
    are destination bounds inside ``grid_shape``; ``src_lo/hi`` are the
    matching bounds inside a window-local array of shape ``hi - lo`` (i.e.
    the region to copy from/into). ``clipped_hi <= clipped_lo`` means the
    window doesn't overlap the grid at all."""
    clipped_lo = tuple(max(0, lo[d]) for d in range(3))
    clipped_hi = tuple(min(grid_shape[d], hi[d]) for d in range(3))
    src_lo = tuple(clipped_lo[d] - lo[d] for d in range(3))
    src_hi = tuple(src_lo[d] + max(0, clipped_hi[d] - clipped_lo[d]) for d in range(3))
    return clipped_lo, clipped_hi, src_lo, src_hi


# ----------------------------------------------------------------------
# Phase A -- Gaussian-weighted blending of raw sem+aff logits
# ----------------------------------------------------------------------

def _accumulate_blend(
    module, vol_path: Path, native_resolution: Sequence[float], fine_nm: float,
    vmin: float, vmax: float, window: Sequence[int], blocks: Sequence[Block],
    padded_fine_shape: Sequence[int], device: torch.device, acc_path: Path,
) -> int:
    """Run the network once per (heavily overlapping) window in ``blocks``,
    weight its raw sem+aff logits by a 3-D Gaussian centered on the window,
    and accumulate both the weighted logits and the weight itself into an
    on-disk HDF5 file at ``acc_path`` (datasets ``"acc"`` and ``"weight"``,
    covering ``padded_fine_shape``). Returns the number of field channels
    accumulated (``N_AFF + 1``, i.e. affinities + semantic; the raw-recon
    channel is dropped)."""
    from nanocosmos.inference.sliding_window import create_gaussian_weight

    window = tuple(int(w) for w in window)
    gw = create_gaussian_weight(window, device=device)  # [D, H, W], peak 1.0
    gw_np = gw.detach().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        dummy = torch.zeros((1, 1) + window, device=device)
        n_fields = int(module(dummy).shape[1]) - 1  # drop the trailing raw-recon channel

    rdcc = dict(rdcc_nbytes=2 * 1024 ** 3, rdcc_nslots=1_000_003)
    t0 = time.time()
    with h5py.File(str(acc_path), "w", **rdcc) as f:
        acc_ds = f.create_dataset(
            "acc", shape=(n_fields,) + tuple(padded_fine_shape), dtype=np.float32,
            chunks=(n_fields,) + tuple(min(s, 128) for s in padded_fine_shape),
        )
        w_ds = f.create_dataset(
            "weight", shape=tuple(padded_fine_shape), dtype=np.float32,
            chunks=tuple(min(s, 128) for s in padded_fine_shape),
        )
        for bi, block in enumerate(blocks):
            elapsed = time.time() - t0
            clipped_lo, clipped_hi, src_lo, src_hi = _clip_window(block.padded_lo, block.padded_hi, padded_fine_shape)
            if any(clipped_hi[d] <= clipped_lo[d] for d in range(3)):
                print(f"  [blend] window {bi + 1}/{len(blocks)}  {block}  (outside padded grid, skipped)  [{elapsed:.0f}s]")
                continue

            native_lo = _fine_to_native(block.padded_lo, native_resolution, fine_nm)
            native_hi = _fine_to_native(block.padded_hi, native_resolution, fine_nm)
            native_size = tuple(max(1, native_hi[d] - native_lo[d]) for d in range(3))
            region = _read_native_region(vol_path, native_lo, native_size)
            if not region.any():
                print(f"  [blend] window {bi + 1}/{len(blocks)}  {block}  (empty, skipped)  [{elapsed:.0f}s]")
                continue

            image01 = np.clip((region - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
            t = torch.from_numpy(image01)[None, None].to(device)
            fine_image = F.interpolate(t, size=window, mode="trilinear", align_corners=False)
            with torch.no_grad():
                head = module(fine_image)[0, :n_fields]  # [n_fields, D, H, W]
            weighted = (head * gw).cpu().numpy().astype(np.float32)

            dst_sl = (slice(None),) + tuple(slice(clipped_lo[d], clipped_hi[d]) for d in range(3))
            src_sl = (slice(None),) + tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
            acc_ds[dst_sl] += weighted[src_sl]
            w_dst_sl = tuple(slice(clipped_lo[d], clipped_hi[d]) for d in range(3))
            w_src_sl = tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
            w_ds[w_dst_sl] += gw_np[w_src_sl]

            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"  [blend] window {bi + 1}/{len(blocks)}  {block}  [{elapsed:.0f}s]")
    return n_fields


# ----------------------------------------------------------------------
# Phase B -- Mutex Watershed over the blended field
# ----------------------------------------------------------------------

def _mws_from_blend(
    module, blend_path: Path, blocks: Sequence[Block], padded_fine_shape: Sequence[int],
    native_resolution: Sequence[float], fine_nm: float, native_shape: Sequence[int],
    device: torch.device, sem_threshold: float, full_ds, next_id_start: int,
) -> int:
    """Read the ALREADY Gaussian-blended (but not yet normalised) sem+aff
    logits for each MWS chunk in ``blocks``, normalise by the accumulated
    weight, sigmoid, run Mutex Watershed once, crop to the chunk's core,
    relabel, resample to native resolution and write into ``full_ds``.
    Returns the next free instance id."""
    next_id = next_id_start
    rdcc = dict(rdcc_nbytes=2 * 1024 ** 3, rdcc_nslots=1_000_003)
    t0 = time.time()
    with h5py.File(str(blend_path), "r", **rdcc) as f:
        acc_ds = f["acc"]
        w_ds = f["weight"]
        n_fields = acc_ds.shape[0]
        for bi, block in enumerate(blocks):
            elapsed = time.time() - t0
            window = tuple(block.padded_hi[d] - block.padded_lo[d] for d in range(3))
            clipped_lo, clipped_hi, src_lo, src_hi = _clip_window(block.padded_lo, block.padded_hi, padded_fine_shape)
            if any(clipped_hi[d] <= clipped_lo[d] for d in range(3)):
                print(f"  [mws] chunk {bi + 1}/{len(blocks)}  {block}  (outside padded grid, skipped)  [{elapsed:.0f}s]")
                continue

            buf = np.zeros((n_fields,) + window, dtype=np.float32)
            wbuf = np.zeros(window, dtype=np.float32)
            dst_sl = (slice(None),) + tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
            src_sl = (slice(None),) + tuple(slice(clipped_lo[d], clipped_hi[d]) for d in range(3))
            buf[dst_sl] = acc_ds[src_sl]
            w_dst_sl = tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
            w_src_sl = tuple(slice(clipped_lo[d], clipped_hi[d]) for d in range(3))
            wbuf[w_dst_sl] = w_ds[w_src_sl]

            if not wbuf.any():
                print(f"  [mws] chunk {bi + 1}/{len(blocks)}  {block}  (empty, skipped)  [{elapsed:.0f}s]")
                continue

            blended = buf / (wbuf[None] + 1e-8)  # weighted-average of participating windows
            t = torch.from_numpy(blended).to(device)
            aff = t[:-1].sigmoid().float()[None]  # [1, N_AFF, D, H, W]
            sem = t[-1:].sigmoid().float()[None]  # [1, 1, D, H, W]
            sem_fg = (sem[:, 0] > sem_threshold) if getattr(module.agglomerator, "gate_with_sem", True) else None
            seg_padded = module.agglomerator(aff, sem_fg)[0]

            off = tuple(block.core_lo[d] - block.padded_lo[d] for d in range(3))
            sz = tuple(block.core_hi[d] - block.core_lo[d] for d in range(3))
            seg_core_fine = seg_padded[off[0]:off[0] + sz[0], off[1]:off[1] + sz[1], off[2]:off[2] + sz[2]]
            seg_core_fine = seg_core_fine.cpu().numpy().astype(np.int64)
            if not seg_core_fine.any():
                print(f"  [mws] chunk {bi + 1}/{len(blocks)}  {block}  (no instances, skipped)  [{elapsed:.0f}s]")
                continue
            n_local = int(seg_core_fine.max())
            seg_core_fine = np.where(seg_core_fine > 0, seg_core_fine + next_id - 1, 0)
            next_id += n_local

            native_lo = _fine_to_native(block.core_lo, native_resolution, fine_nm)
            native_hi = _fine_to_native(block.core_hi, native_resolution, fine_nm)
            native_size_full = tuple(max(1, native_hi[d] - native_lo[d]) for d in range(3))
            seg_core_native_full = F.interpolate(
                torch.from_numpy(seg_core_fine)[None, None].float(), size=native_size_full, mode="nearest",
            )[0, 0].numpy().astype(np.int64)

            # Clip to the REAL (unpadded) native volume -- boundary chunks may
            # extend past it since the fine grid was rounded up for tiling.
            clip_lo = tuple(max(0, min(native_lo[d], native_shape[d])) for d in range(3))
            clip_hi = tuple(max(0, min(native_hi[d], native_shape[d])) for d in range(3))
            if any(clip_hi[d] <= clip_lo[d] for d in range(3)):
                print(f"  [mws] chunk {bi + 1}/{len(blocks)}  {block}  (fully outside real volume, skipped)  [{elapsed:.0f}s]")
                continue
            rel_lo = tuple(clip_lo[d] - native_lo[d] for d in range(3))
            rel_hi = tuple(clip_hi[d] - native_lo[d] for d in range(3))
            seg_core_native = seg_core_native_full[rel_lo[0]:rel_hi[0], rel_lo[1]:rel_hi[1], rel_lo[2]:rel_hi[2]]
            sl = tuple(slice(clip_lo[d], clip_hi[d]) for d in range(3))
            full_ds[sl] = seg_core_native
            print(f"  [mws] chunk {bi + 1}/{len(blocks)}  {block}  {n_local} local ids  [{elapsed:.0f}s]")
    return next_id


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
    window_size: Optional[Sequence[int]] = None,
    stride_frac: float = 0.25,
    keep_blend_cache: bool = False,
) -> Optional[Path]:
    """Run two-phase blockwise inference over the full volume and write a
    challenge submission file (see module docstring for Phase A/B details).

    Args:
        fine_core_size / fine_context: Phase B (Mutex Watershed) chunk CORE
            size / context margin, in fine-grid voxels -- applied to the
            ALREADY Gaussian-blended field (no network inference in this
            phase, so these can be larger than a Phase A window).
        window_size: Phase A network window (defaults to
            ``cfg.data.patch_size``, e.g. 400x256x256 @ 4nm -- the model's
            native field of view). Every Phase A window is resampled/padded
            to exactly this size, even at the volume's edges.
        stride_frac: Fraction of ``window_size`` used as the stride between
            adjacent Phase A windows (default 0.25 -> 75% overlap).
        keep_blend_cache: Keep the (potentially very large -- see module
            docstring) Phase A accumulator file instead of deleting it once
            Phase B finishes.
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

    window = tuple(int(w) for w in (window_size if window_size is not None else patch_size))
    blend_core = tuple(max(1, round(window[d] * stride_frac)) for d in range(3))
    extra = tuple(window[d] - blend_core[d] for d in range(3))
    blend_ctx_lo = tuple(e // 2 for e in extra)
    blend_ctx_hi = tuple(extra[d] - blend_ctx_lo[d] for d in range(3))

    # Round the fine grid UP to an exact multiple of the blend stride so
    # EVERY Phase A window (incl. ones touching the far edge) is the full,
    # constant `window` size -- the excess is zero-padded (_read_native_region).
    padded_fine_shape = tuple(-(-fine_shape[d] // blend_core[d]) * blend_core[d] for d in range(3))

    blend_blocks = _plan_blocks(padded_fine_shape, blend_core, blend_ctx_lo, blend_ctx_hi)
    mws_core = tuple(int(s) for s in fine_core_size)
    mws_ctx = tuple(int(s) for s in fine_context)
    mws_blocks = _plan_blocks(padded_fine_shape, mws_core, mws_ctx, mws_ctx)

    voxels = int(np.prod(padded_fine_shape))
    acc_gb = voxels * 31 * 4 / 1e9  # rough estimate assuming up to N_AFF=30 + sem=1; refined after the dummy forward
    weight_gb = voxels * 4 / 1e9

    print(f"Volume: {vol_path.name}  native shape {native_shape} @ {tuple(native_resolution)} nm")
    print(
        f"Fine grid: {fine_shape} @ {fine_nm} nm"
        + (f"  (padded to {padded_fine_shape} for exact window tiling)" if padded_fine_shape != fine_shape else "")
    )
    print(
        f"Phase A (Gaussian blend): window {window}, stride {blend_core} "
        f"({stride_frac:g}x window, context {blend_ctx_lo}+{blend_ctx_hi}) -> {len(blend_blocks)} overlapping windows"
    )
    print(f"Phase B (Mutex Watershed): core {mws_core}, context {mws_ctx} -> {len(mws_blocks)} chunks on the blended field")
    print(f"Blend accumulator (Phase A): ~{acc_gb:.0f} GB logits + ~{weight_gb:.1f} GB weight on disk "
          f"(shrink via --window-size / raise --stride-frac if impractical)")
    if dry_run:
        print("Sample Phase A windows:")
        for b in blend_blocks[:5]:
            print(f"  {b}")
        print("Sample Phase B chunks:")
        for b in mws_blocks[:5]:
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
    blend_path = out_dir / f"{vol_path.stem}_blend_acc.h5"
    print(f"Phase A: accumulating Gaussian-blended sem+aff logits -> {blend_path}")
    n_fields = _accumulate_blend(
        module, vol_path, native_resolution, fine_nm, vmin, vmax,
        window, blend_blocks, padded_fine_shape, device_t, blend_path,
    )
    print(f"Phase A done: {n_fields} field channels (aff x{n_fields - 1} + sem) blended over {padded_fine_shape} fine-grid voxels.")

    full_path = out_dir / f"{vol_path.stem}_pred_label_native_full.h5"
    with h5py.File(str(full_path), "w") as f:
        full_ds = f.create_dataset(
            "main", shape=native_shape, dtype=np.int64,
            chunks=tuple(min(s, 256) for s in native_shape), compression="gzip", compression_opts=4,
        )
        print(f"Phase B: Mutex Watershed over the blended field -> {full_path}")
        next_id = _mws_from_blend(
            module, blend_path, mws_blocks, padded_fine_shape,
            native_resolution, fine_nm, native_shape, device_t, sem_threshold,
            full_ds, next_id_start=1,
        )
        full_ds.attrs["resolution_zyx_nm"] = np.asarray(native_resolution, dtype=np.float64)
        full_ds.attrs["source"] = (
            f"nanocosmos blockwise inference from {ckpt_path}; Phase A Gaussian-blended sem+aff over "
            f"overlapping {window} windows (stride {blend_core}), Phase B Mutex Watershed per "
            f"{mws_core}+{mws_ctx} chunk of the blended field; see infer_submission.py docstring "
            "for the cross-chunk-merge limitation."
        )
        print(f"Full native-resolution segmentation ({next_id - 1} total ids) -> {full_path}")

    if keep_blend_cache:
        print(f"Blend cache kept at {blend_path} (--keep-blend-cache)")
    else:
        blend_path.unlink(missing_ok=True)

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
        help="Phase B (Mutex Watershed) chunk CORE size in fine-grid voxels, "
             "applied to the already Gaussian-blended field (no network "
             "inference in this phase).",
    )
    p.add_argument(
        "--fine-context", type=int, nargs=3, default=[200, 128, 128], metavar=("D", "H", "W"),
        help="Phase B (Mutex Watershed) chunk context margin per side, in "
             "fine-grid voxels.",
    )
    p.add_argument(
        "--window-size", type=int, nargs=3, default=None, metavar=("D", "H", "W"),
        help="Phase A network window (full field of view fed to the model "
             "per forward pass), in fine-grid voxels; defaults to the "
             "model's native cfg.data.patch_size (e.g. 400 256 256 @ 4nm). "
             "Every window is resampled/zero-padded to exactly this size, "
             "even at the volume's edges.",
    )
    p.add_argument(
        "--stride-frac", type=float, default=0.25,
        help="Fraction of --window-size used as the stride between "
             "adjacent Phase A windows (default 0.25 -> windows advance by "
             "1/4 of the window, i.e. 75%% overlap).",
    )
    p.add_argument(
        "--keep-blend-cache", action="store_true",
        help="Keep the (potentially very large, see docstring) Phase A "
             "blended-logits HDF5 cache instead of deleting it after Phase B.",
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
        help="Print the block plan (window/chunk counts, sizes, disk estimate) and exit -- no inference.",
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
        window_size=args.window_size,
        stride_frac=args.stride_frac,
        keep_blend_cache=args.keep_blend_cache,
        device=args.device,
        sem_threshold=args.sem_threshold,
        dry_run=args.dry_run,
        submission_format=args.submission_format,
    )


if __name__ == "__main__":
    main()
