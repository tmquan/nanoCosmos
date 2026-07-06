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
just applied at the whole-volume scale). Windows are processed in batches of
``--blend-batch-size`` (a single forward pass per batch, since each window
equals the native patch) to keep the GPU busy instead of one window at a
time; reading + resampling each window (I/O-bound) is parallelised across
``--blend-io-workers`` background threads feeding a shared work queue. For
every window: take the raw (pre-activation) unified head -- affinity +
semantic + raw-reconstruction channels, all of it, since Phase B also emits
fine-grid raw/sem diagnostics, not just the segmentation -- multiply by a
3-D Gaussian weight (peak at the window center, tapering to the edges), and
accumulate both the weighted logits and the weight itself into an on-disk
HDF5 accumulator covering the whole (padded) fine grid. If the real volume
doesn't divide evenly into an integer number of window strides, the fine
grid is rounded UP to the next multiple and the extra region is zero-padded
(never a smaller/ragged window) -- see ``_read_native_region``.

Phase A scales two ways, both via that shared queue (I/O threads produce
ready batches; consumer threads pull and forward them):
  * ``--workers-per-gpu N`` (N > 1) runs multiple batches *concurrently on
    one GPU*: PyTorch gives each Python thread its own default CUDA stream
    per device, so independent forward passes from separate threads on the
    same GPU genuinely overlap on-device instead of serialising -- useful
    when a single batch doesn't saturate the GPU. Each worker gets its own
    full model replica (see below), so GPU memory use scales with N too.
  * ``--gpu-ids 0 1 2 3`` scales *across GPUs*: one or more model replicas
    per id (``workers-per-gpu`` each), all accumulating into the SAME
    on-disk file (writes are lock-protected, so this is safe regardless of
    which worker finishes a given window first). Total Phase A concurrency
    is ``len(gpu_ids) * workers_per_gpu`` forward passes in flight.
Omitting ``--gpu-ids`` keeps the original single-``--device`` behaviour.

IMPORTANT: every worker gets its OWN model replica, never a shared
instance -- the cosmos_2_5_common DiT backbone keeps its intermediate-
feature hook buffer as INSTANCE state during ``forward()``
(``self._hook_buffer``), so two threads calling ``forward()`` concurrently
on the same instance corrupt each other's buffers (manifests as bogus
tensor-shape-mismatch crashes). This means ``--workers-per-gpu N`` costs
N x the model's GPU memory per GPU it's applied to, on top of N x the
activation memory from running N batches at once -- budget accordingly.

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
on model activations). Per chunk: normalise + split into affinity / semantic
/ raw; when ``--save-fine-grid`` (default true), crop the CORE and write the
raw reconstruction (linear) and semantic probability (sigmoid) straight into
two fine-grid ``.h5`` outputs -- this happens regardless of segmentation,
since they're valid predictions even where Mutex Watershed finds nothing.
Then sigmoid affinity + semantic, run MWS once, crop away the context margin
(keep only the CORE), relabel with a running id offset so ids never collide
between chunks, optionally write the (still fine-grid) instance ids into a
third fine-grid output, resample down to **native** resolution (nearest-
neighbour) and write into the pre-allocated full-native output ``.h5``
(chunked on disk -- never held fully in memory).

Chunks are processed concurrently on a ``--mws-workers``-thread pool. The
CPU MWS backend (``mws_np``) is numba-JIT'd with ``nogil=True`` (see
``nanocosmos/inference/mutex_watershed.py``), so concurrent chunks actually
run on separate CPU cores instead of serialising on the GIL -- the id-offset
bookkeeping and the reads/writes against the shared blend file / output
dataset are lock-protected, so this is safe regardless of completion order
(ids stay globally unique, just not assigned in chunk order). The GPU MWS
backend still executes on one device/stream, so extra workers there mostly
overlap I/O with compute rather than parallelising MWS itself.

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

Outputs
-------
Always written, in ``--out-dir``:
  * ``<vol>_pred_label_native_full.h5`` -- the full native-resolution
    instance segmentation (dataset ``main``); this is what gets cropped /
    packaged into the actual challenge submission file above.
  * the submission file itself (``*_submission.hdf`` for CREMI,
    ``*_submission.zip`` for SNEMI3D).
When ``--save-fine-grid`` (default true), also written at the network's
**fine grid** resolution (e.g. 4 nm -- diagnostic / super-resolution
outputs, matching ``infer_volume.py``'s ``--save-fine-grid``, NOT required
for scoring):
  * ``<vol>_pred_raw_fine.h5`` -- raw reconstruction (linear values).
  * ``<vol>_pred_sem_fine.h5`` -- semantic foreground probability (sigmoid).
  * ``<vol>_pred_label_fine.h5`` -- instance segmentation before the final
    native-resolution downsample (same ids as the native output, just not
    yet resampled).
All three are written straight from Phase B's per-chunk blended field (no
extra network pass), cropped to each chunk's CORE the same way the
segmentation is -- the raw/sem ones are written regardless of whether MWS
finds any instances in that chunk, since they're valid predictions either
way. Pass ``--no-save-fine-grid`` to skip them and save the (comparatively
modest, single-channel) extra disk + write time.

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
The Phase A accumulator holds ``(N_AFF + 2)`` float32 channels (every
unified-head channel: affinity + semantic + raw) over the **whole padded
fine grid**, e.g. for SNEMI3D AC3 (fine grid ~800x1536x1536) with the
default 30-offset model that's ~240 GB on disk (plus ~7.5 GB for the weight
map); for CREMI's much larger padded volumes this is on the order of
2+ TB PER SAMPLE. IMPORTANT: this size is essentially FIXED by the volume's
real fine-grid shape x channel count -- ``--window-size`` / ``--stride-frac``
barely move it (they only change the padding-rounding remainder, a tiny
fraction); they instead control how many overlapping windows are processed
and thus the read-modify-write I/O *volume* against that fixed-size
accumulator, not its size on disk. There is currently no way to shrink the
accumulator itself short of a code change (e.g. blending in float16, or
only blending aff+sem when ``--no-save-fine-grid`` is passed -- NOT
currently implemented; Phase A always blends all channels regardless of
``--save-fine-grid``). The ``--save-fine-grid`` outputs are comparatively
modest -- three single-channel volumes over the REAL (unpadded) fine grid,
e.g. ~28 GB total for AC3 but ~300 GB for CREMI. The printed block-plan
(also shown by ``--dry-run``) reports the exact numbers before you commit
to a run -- check available scratch space against it first.

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
import contextlib
import itertools
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

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

def _prepare_window(
    vol_path: Path, block: Block, native_resolution: Sequence[float], fine_nm: float,
    vmin: float, vmax: float, window: Tuple[int, int, int],
) -> Optional[torch.Tensor]:
    """Read + normalise + resample one window's native region onto the fine
    grid (on CPU, so this can run in a background thread while the GPU is
    busy with a previous batch). Returns a ``[1, D, H, W]`` CPU tensor ready
    to be stacked into a batch, or ``None`` if the region has no content
    (skip -- saves a wasted forward pass)."""
    native_lo = _fine_to_native(block.padded_lo, native_resolution, fine_nm)
    native_hi = _fine_to_native(block.padded_hi, native_resolution, fine_nm)
    native_size = tuple(max(1, native_hi[d] - native_lo[d]) for d in range(3))
    region = _read_native_region(vol_path, native_lo, native_size)
    if not region.any():
        return None
    image01 = np.clip((region - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    t = torch.from_numpy(image01)[None, None]  # [1, 1, d, h, w]
    return F.interpolate(t, size=window, mode="trilinear", align_corners=False)[0]  # [1, D, H, W]


def _accumulate_blend(
    replicas: "List[Tuple[torch.device, Any]]",
    vol_path: Path, native_resolution: Sequence[float], fine_nm: float,
    vmin: float, vmax: float, window: Sequence[int], blocks: Sequence[Block],
    padded_fine_shape: Sequence[int], acc_path: Path,
    batch_size: int = 4, io_workers: int = 4,
) -> int:
    """Run the network on batches of (heavily overlapping) windows from
    ``blocks`` -- ``batch_size`` windows per forward pass, keeping the
    GPU(s) busy instead of one window at a time -- weight each window's raw
    sem+aff logits by a 3-D Gaussian centered on it, and accumulate both the
    weighted logits and the weight itself into an on-disk HDF5 file at
    ``acc_path`` (datasets ``"acc"`` and ``"weight"``, covering
    ``padded_fine_shape``). Returns the number of field channels
    accumulated (``N_AFF + 2``, i.e. every unified-head channel: affinities
    + semantic + raw reconstruction -- all three are blended so Phase B can
    also emit fine-grid raw/sem diagnostics, not just the segmentation).

    Scales via a shared producer/consumer pipeline: a dedicated thread pool
    reads + resamples windows (I/O-bound) and pushes ready batches onto a
    bounded queue; one consumer thread per ``(device, module)`` pair in
    ``replicas`` pulls batches and runs the forward pass on its own model
    instance.

    - Multiple entries with the *same* device run concurrent batches on
      *one GPU* (PyTorch gives each thread its own default CUDA stream per
      device, so independent forward passes from separate threads on one
      GPU genuinely overlap on-device -- useful when a single batch doesn't
      saturate it).
    - Entries with *different* devices scale *across GPUs*.

    IMPORTANT: every worker needs its OWN model instance -- the
    cosmos_2_5_common DiT wrapper keeps its intermediate-feature hook
    buffer as INSTANCE state during ``forward()`` (``self._hook_buffer``,
    see ``nanocosmos/models/cosmos_2_5_common/wrapper_base.py``), so two
    threads calling ``forward()`` concurrently on the *same* instance
    corrupt each other's buffers (silent shape-mismatch crashes). The
    caller is responsible for building ``len(replicas)`` independent
    replicas; never pass the same module twice.

    All consumers share one accumulator file and a running progress
    counter, both lock-protected -- safe regardless of which worker/device
    finishes a given window first."""
    import queue
    import threading

    from nanocosmos.inference.sliding_window import create_gaussian_weight

    window = tuple(int(w) for w in window)
    primary_device, primary_module = replicas[0]
    gw_by_id = {id(mod): create_gaussian_weight(window, device=dev) for dev, mod in replicas}
    gw_np = gw_by_id[id(primary_module)].detach().cpu().numpy().astype(np.float32)

    with torch.no_grad():
        dummy = torch.zeros((1, 1) + window, device=primary_device)
        n_fields = int(primary_module(dummy).shape[1])  # keep every channel: aff + sem + raw

    rdcc = dict(rdcc_nbytes=2 * 1024 ** 3, rdcc_nslots=1_000_003)
    acc_file = h5py.File(str(acc_path), "w", **rdcc)
    acc_ds = acc_file.create_dataset(
        "acc", shape=(n_fields,) + tuple(padded_fine_shape), dtype=np.float32,
        chunks=(n_fields,) + tuple(min(s, 128) for s in padded_fine_shape),
    )
    w_ds = acc_file.create_dataset(
        "weight", shape=tuple(padded_fine_shape), dtype=np.float32,
        chunks=tuple(min(s, 128) for s in padded_fine_shape),
    )

    t0 = time.time()
    n_done = [0]
    progress_lock = threading.Lock()
    write_lock = threading.Lock()

    batches = [blocks[i:i + batch_size] for i in range(0, len(blocks), batch_size)]
    n_consumers = len(replicas)
    work_q: "queue.Queue" = queue.Queue(maxsize=max(2, n_consumers * 2))

    def _producer() -> None:
        with ThreadPoolExecutor(max_workers=io_workers) as io_pool:
            for batch_blocks in batches:
                prepared = list(io_pool.map(
                    lambda b: _prepare_window(vol_path, b, native_resolution, fine_nm, vmin, vmax, window),
                    batch_blocks,
                ))
                work_q.put((batch_blocks, prepared))
        for _ in range(n_consumers):
            work_q.put(None)  # one sentinel per consumer

    def _consumer(dev: torch.device, mod) -> None:
        gw = gw_by_id[id(mod)]
        while True:
            item = work_q.get()
            if item is None:
                work_q.task_done()
                return
            batch_blocks, prepared = item
            valid = [(b, img) for b, img in zip(batch_blocks, prepared) if img is not None]
            for b, img in zip(batch_blocks, prepared):
                if img is None:
                    with progress_lock:
                        n_done[0] += 1
                        bi, elapsed = n_done[0], time.time() - t0
                    print(f"  [blend] window {bi}/{len(blocks)}  {b}  (empty, skipped)  [{elapsed:.0f}s]")
            if valid:
                batch_t = torch.stack([img for _, img in valid], dim=0).to(dev, non_blocking=True)
                with torch.no_grad():
                    heads = mod(batch_t)  # [B, n_fields, D, H, W] -- every unified-head channel
                weighted = (heads * gw).cpu().numpy().astype(np.float32)  # gw broadcasts over B and channel dims

                for (block, _), w_arr in zip(valid, weighted):
                    with progress_lock:
                        n_done[0] += 1
                        bi, elapsed = n_done[0], time.time() - t0
                    clipped_lo, clipped_hi, src_lo, src_hi = _clip_window(block.padded_lo, block.padded_hi, padded_fine_shape)
                    if any(clipped_hi[d] <= clipped_lo[d] for d in range(3)):
                        print(f"  [blend] window {bi}/{len(blocks)}  {block}  (outside padded grid, skipped)  [{elapsed:.0f}s]")
                        continue
                    dst_sl = (slice(None),) + tuple(slice(clipped_lo[d], clipped_hi[d]) for d in range(3))
                    src_sl = (slice(None),) + tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
                    w_dst_sl = tuple(slice(clipped_lo[d], clipped_hi[d]) for d in range(3))
                    w_src_sl = tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
                    with write_lock:
                        acc_ds[dst_sl] += w_arr[src_sl]
                        w_ds[w_dst_sl] += gw_np[w_src_sl]
                    print(f"  [blend] window {bi}/{len(blocks)}  {block}  (batch of {len(valid)})  [{dev}]  [{elapsed:.0f}s]")

                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            work_q.task_done()

    try:
        if not batches:
            return n_fields
        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()
        consumers = [
            threading.Thread(target=_consumer, args=(dev, mod), daemon=True)
            for dev, mod in replicas
        ]
        for c in consumers:
            c.start()
        producer.join()
        for c in consumers:
            c.join()
    finally:
        acc_file.close()
    return n_fields


# ----------------------------------------------------------------------
# Phase B -- Mutex Watershed over the blended field
# ----------------------------------------------------------------------

def _mws_from_blend(
    module, blend_path: Path, blocks: Sequence[Block], padded_fine_shape: Sequence[int],
    native_resolution: Sequence[float], fine_nm: float, native_shape: Sequence[int],
    fine_shape: Sequence[int], device: torch.device, sem_threshold: float, full_ds, next_id_start: int,
    num_workers: int = 1, raw_ds=None, sem_ds=None, label_fine_ds=None,
) -> int:
    """Read the ALREADY Gaussian-blended (but not yet normalised) aff+sem+raw
    logits for each MWS chunk in ``blocks``, normalise by the accumulated
    weight. Optionally (whenever the corresponding ``*_ds`` is given) crop
    the chunk's CORE and write the fine-grid raw reconstruction / semantic
    probability into ``raw_ds`` / ``sem_ds`` -- this always happens,
    independent of segmentation, since they're valid predictions even where
    Mutex Watershed finds no instances. Then sigmoid the aff+sem channels,
    run Mutex Watershed once, crop to the chunk's core, relabel, optionally
    write the (still fine-grid) instance ids into ``label_fine_ds``, resample
    to native resolution and write into ``full_ds``.

    When ``num_workers > 1``, chunks are processed concurrently on a thread
    pool: MWS's CPU backend (``mws_np``) is JIT-compiled with numba's
    ``nogil=True`` (see ``nanocosmos/inference/mutex_watershed.py``), so
    separate chunks' agglomeration genuinely runs on separate CPU cores
    instead of serialising on the GIL. (The GPU backends still execute on
    one device/stream, so extra workers there mainly overlap I/O with
    compute rather than parallelising MWS itself.) Reads/writes against the
    shared blend file and the output datasets, and the running id offset,
    are lock-protected so concurrent chunks can't race; completion order
    (and hence which chunk gets which id range) is nondeterministic but ids
    are still guaranteed unique. Returns the next free instance id."""
    import threading

    rdcc = dict(rdcc_nbytes=2 * 1024 ** 3, rdcc_nslots=1_000_003)
    t0 = time.time()
    io_lock = threading.Lock()
    id_lock = threading.Lock()
    write_lock = threading.Lock()
    next_id = [next_id_start]
    n_done = [0]

    blend_file = h5py.File(str(blend_path), "r", **rdcc)
    acc_ds = blend_file["acc"]
    w_ds = blend_file["weight"]
    n_fields = acc_ds.shape[0]
    n_aff = n_fields - 2  # unified head layout: [aff x N_AFF, sem, raw]

    def _fine_clip(core_lo, core_hi):
        """Clip a chunk's CORE bounds (may exceed ``fine_shape`` near the far
        edge, since the fine grid was rounded up for tiling) to the REAL
        fine grid; returns ``None`` if fully outside, else
        ``(dst_slices, rel_lo, rel_hi)`` where ``rel_lo/hi`` index a
        core-local array of shape ``core_hi - core_lo``."""
        clip_lo = tuple(min(core_lo[d], fine_shape[d]) for d in range(3))
        clip_hi = tuple(min(core_hi[d], fine_shape[d]) for d in range(3))
        if any(clip_hi[d] <= clip_lo[d] for d in range(3)):
            return None
        rel_lo = tuple(clip_lo[d] - core_lo[d] for d in range(3))
        rel_hi = tuple(rel_lo[d] + (clip_hi[d] - clip_lo[d]) for d in range(3))
        dst = tuple(slice(clip_lo[d], clip_hi[d]) for d in range(3))
        return dst, rel_lo, rel_hi

    def _process(block: Block) -> None:
        window = tuple(block.padded_hi[d] - block.padded_lo[d] for d in range(3))
        clipped_lo, clipped_hi, src_lo, src_hi = _clip_window(block.padded_lo, block.padded_hi, padded_fine_shape)
        with io_lock:
            n_done[0] += 1
            bi = n_done[0]
        elapsed = time.time() - t0
        if any(clipped_hi[d] <= clipped_lo[d] for d in range(3)):
            print(f"  [mws] chunk {bi}/{len(blocks)}  {block}  (outside padded grid, skipped)  [{elapsed:.0f}s]")
            return

        buf = np.zeros((n_fields,) + window, dtype=np.float32)
        wbuf = np.zeros(window, dtype=np.float32)
        dst_sl = (slice(None),) + tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
        src_sl = (slice(None),) + tuple(slice(clipped_lo[d], clipped_hi[d]) for d in range(3))
        w_dst_sl = tuple(slice(src_lo[d], src_hi[d]) for d in range(3))
        w_src_sl = tuple(slice(clipped_lo[d], clipped_hi[d]) for d in range(3))
        with io_lock:
            buf[dst_sl] = acc_ds[src_sl]
            wbuf[w_dst_sl] = w_ds[w_src_sl]

        if not wbuf.any():
            print(f"  [mws] chunk {bi}/{len(blocks)}  {block}  (empty, skipped)  [{elapsed:.0f}s]")
            return

        blended = buf / (wbuf[None] + 1e-8)  # weighted-average of participating windows
        t = torch.from_numpy(blended).to(device)
        aff = t[:n_aff].sigmoid().float()[None]           # [1, N_AFF, D, H, W]
        sem = t[n_aff:n_aff + 1].sigmoid().float()[None]   # [1, 1, D, H, W]
        raw = t[n_aff + 1:n_aff + 2].float()[None]         # [1, 1, D, H, W], linear (not sigmoided)

        off = tuple(block.core_lo[d] - block.padded_lo[d] for d in range(3))
        sz = tuple(block.core_hi[d] - block.core_lo[d] for d in range(3))
        core_sl = (slice(off[0], off[0] + sz[0]), slice(off[1], off[1] + sz[1]), slice(off[2], off[2] + sz[2]))

        # ---- fine-grid raw/sem diagnostics -- independent of segmentation,
        # these are valid predictions even where MWS finds no instances.
        if raw_ds is not None or sem_ds is not None:
            clip = _fine_clip(block.core_lo, block.core_hi)
            if clip is not None:
                dst, rel_lo, rel_hi = clip
                rel_sl = (slice(rel_lo[0], rel_hi[0]), slice(rel_lo[1], rel_hi[1]), slice(rel_lo[2], rel_hi[2]))
                with write_lock:
                    if raw_ds is not None:
                        raw_ds[dst] = raw[0, 0][core_sl][rel_sl].cpu().numpy()
                    if sem_ds is not None:
                        sem_ds[dst] = sem[0, 0][core_sl][rel_sl].cpu().numpy()

        sem_fg = (sem[:, 0] > sem_threshold) if getattr(module.agglomerator, "gate_with_sem", True) else None
        seg_padded = module.agglomerator(aff, sem_fg)[0]

        seg_core_fine = seg_padded[core_sl].cpu().numpy().astype(np.int64)
        if not seg_core_fine.any():
            print(f"  [mws] chunk {bi}/{len(blocks)}  {block}  (no instances, skipped)  [{elapsed:.0f}s]")
            return
        n_local = int(seg_core_fine.max())
        with id_lock:
            offset = next_id[0]
            next_id[0] += n_local
        seg_core_fine = np.where(seg_core_fine > 0, seg_core_fine + offset - 1, 0)

        if label_fine_ds is not None:
            clip = _fine_clip(block.core_lo, block.core_hi)
            if clip is not None:
                dst, rel_lo, rel_hi = clip
                with write_lock:
                    label_fine_ds[dst] = seg_core_fine[rel_lo[0]:rel_hi[0], rel_lo[1]:rel_hi[1], rel_lo[2]:rel_hi[2]]

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
            print(f"  [mws] chunk {bi}/{len(blocks)}  {block}  (fully outside real volume, skipped)  [{elapsed:.0f}s]")
            return
        rel_lo = tuple(clip_lo[d] - native_lo[d] for d in range(3))
        rel_hi = tuple(clip_hi[d] - native_lo[d] for d in range(3))
        seg_core_native = seg_core_native_full[rel_lo[0]:rel_hi[0], rel_lo[1]:rel_hi[1], rel_lo[2]:rel_hi[2]]
        sl = tuple(slice(clip_lo[d], clip_hi[d]) for d in range(3))
        with write_lock:
            full_ds[sl] = seg_core_native
        print(f"  [mws] chunk {bi}/{len(blocks)}  {block}  {n_local} local ids  [{elapsed:.0f}s]")

    try:
        if num_workers <= 1:
            for block in blocks:
                _process(block)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                list(pool.map(_process, blocks))
    finally:
        blend_file.close()
    return next_id[0]


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
    blend_batch_size: int = 4,
    blend_io_workers: int = 4,
    mws_workers: int = 4,
    gpu_ids: Optional[Sequence[int]] = None,
    workers_per_gpu: int = 1,
    save_fine_grid: bool = True,
    reuse_blend_cache: bool = False,
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
        blend_batch_size: Number of Phase A windows forwarded through the
            network in a single batch (keeps the GPU busy instead of
            running one window at a time). Increase while GPU memory
            allows; each window is already the model's full field of view,
            so activation memory scales roughly linearly with this.
        blend_io_workers: Number of background threads used to read +
            resample windows (I/O-bound) concurrently, and to prefetch the
            next batch while the current one runs on the GPU.
        mws_workers: Number of MWS chunks processed concurrently in Phase B
            (thread pool). Real multi-core speedup requires numba (the CPU
            MWS backend releases the GIL); with the GPU backend, extra
            workers mainly overlap I/O with compute since MWS itself still
            runs on one device.
        gpu_ids: Phase A only -- CUDA device indices to scale across, e.g.
            ``[0, 1, 2, 3]``. ``None`` (default) uses the single ``device``
            below, unchanged.
        workers_per_gpu: Phase A only -- concurrent worker threads per GPU,
            each with its OWN model replica (each thread gets its own
            default CUDA stream, so >1 runs multiple batches at once on the
            same GPU -- helpful when one batch doesn't saturate it, at the
            cost of N x that GPU's model memory). Total Phase A concurrency
            (and replica count) is ``len(gpu_ids or [1]) * workers_per_gpu``.
        save_fine_grid: When ``True`` (default), additionally save the raw
            fine-grid (network-native, e.g. 4 nm) outputs -- ``pred_raw``,
            ``pred_sem``, ``pred_label_fine`` -- on top of the
            always-produced native-resolution instance segmentation (the
            actual submission artifact). Diagnostic / super-resolution
            outputs, matching ``infer_volume.py``'s ``--save-fine-grid``;
            set ``False`` to skip them (saves the extra disk + write time).
        reuse_blend_cache: If a ``<vol>_blend_acc.h5`` already exists at the
            expected path in ``out_dir`` (e.g. left over from a crashed run
            that got through Phase A) AND its ``acc``/``weight`` dataset
            shapes match this run's plan exactly, SKIP Phase A entirely and
            go straight to Phase B on the existing file -- avoids redoing
            the most expensive part of a recovery. Falls back to recomputing
            from scratch (with a warning) if the cache is missing/
            incompatible. Safe to always pass; it's a no-op when there's
            nothing to reuse.
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
    acc_gb = voxels * 32 * 4 / 1e9  # rough estimate assuming up to N_AFF=30 + sem=1 + raw=1; exact count known after the dummy forward
    weight_gb = voxels * 4 / 1e9
    fine_voxels = int(np.prod(fine_shape))
    fine_diag_gb = fine_voxels * (4 + 4 + 8) / 1e9  # raw (f32) + sem (f32) + label_fine (i64), unpadded fine grid

    print(f"Volume: {vol_path.name}  native shape {native_shape} @ {tuple(native_resolution)} nm")
    print(
        f"Fine grid: {fine_shape} @ {fine_nm} nm"
        + (f"  (padded to {padded_fine_shape} for exact window tiling)" if padded_fine_shape != fine_shape else "")
    )
    n_devices = len(gpu_ids) if gpu_ids else 1
    n_concurrent = n_devices * max(1, workers_per_gpu)
    print(
        f"Phase A (Gaussian blend): window {window}, stride {blend_core} "
        f"({stride_frac:g}x window, context {blend_ctx_lo}+{blend_ctx_hi}) -> {len(blend_blocks)} overlapping windows "
        f"(batch {blend_batch_size}, {blend_io_workers} I/O workers -> "
        f"{-(-len(blend_blocks) // blend_batch_size)} forward passes; "
        f"{n_devices} GPU{'s' if n_devices != 1 else ''}"
        f"{' (' + ','.join(str(g) for g in gpu_ids) + ')' if gpu_ids else ''}"
        f" x {workers_per_gpu} worker{'s' if workers_per_gpu != 1 else ''}/GPU "
        f"= {n_concurrent}-way concurrent)"
    )
    print(f"Phase B (Mutex Watershed): core {mws_core}, context {mws_ctx} -> {len(mws_blocks)} chunks on the blended field "
          f"({mws_workers} worker{'s' if mws_workers != 1 else ''})")
    print(f"Blend accumulator (Phase A): ~{acc_gb:.0f} GB logits + ~{weight_gb:.1f} GB weight on disk "
          f"(size is ~fixed by the volume's fine-grid shape x channel count -- --window-size/--stride-frac "
          f"barely move it, only the window/I/O count; see module docstring)")
    if save_fine_grid:
        print(f"Fine-grid diagnostics (Phase B, --save-fine-grid): ~{fine_diag_gb:.1f} GB "
              f"(pred_raw + pred_sem + pred_label_fine @ {fine_shape})")
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

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    blend_devices = (
        [torch.device(f"cuda:{gid}") for gid in gpu_ids] if gpu_ids
        else [torch.device(device if torch.cuda.is_available() else "cpu")]
    )
    # One full model replica per CONCURRENT WORKER (device x workers_per_gpu),
    # never shared -- the cosmos_2_5_common DiT wrapper keeps its
    # intermediate-feature hook buffer as instance state during forward()
    # (self._hook_buffer), so two threads calling forward() on the SAME
    # instance concurrently corrupt each other's buffers. Built + loaded
    # fresh per replica (rather than deep-copied) so peak CPU RAM stays at
    # one model's worth at a time.
    replicas: List[Tuple[torch.device, Any]] = []
    for dev in blend_devices:
        for _ in range(max(1, workers_per_gpu)):
            rep = build_module(cfg)
            rep.load_state_dict(state_dict, strict=False)
            rep.to(dev).eval()
            replicas.append((dev, rep))
    device_t, module = replicas[0]  # used by Phase B below (unaffected by Phase A scaling)

    out_dir.mkdir(parents=True, exist_ok=True)
    blend_path = out_dir / f"{vol_path.stem}_blend_acc.h5"

    n_fields = None
    if reuse_blend_cache and blend_path.exists():
        with torch.no_grad():
            n_fields_expected = int(module(torch.zeros((1, 1) + window, device=device_t)).shape[1])
        try:
            with h5py.File(str(blend_path), "r") as f:
                ok = (
                    "acc" in f and "weight" in f
                    and tuple(f["acc"].shape) == (n_fields_expected,) + tuple(padded_fine_shape)
                    and tuple(f["weight"].shape) == tuple(padded_fine_shape)
                )
        except OSError:
            ok = False
        if ok:
            n_fields = n_fields_expected
            print(f"Phase A: SKIPPED -- reusing existing compatible blend cache -> {blend_path}")
        else:
            print(f"Phase A: --reuse-blend-cache given but {blend_path} is missing/incompatible "
                  "(shape mismatch or unreadable) -- recomputing from scratch.")

    if n_fields is None:
        print(f"Phase A: accumulating Gaussian-blended sem+aff logits -> {blend_path}")
        n_fields = _accumulate_blend(
            replicas, vol_path, native_resolution, fine_nm, vmin, vmax,
            window, blend_blocks, padded_fine_shape, blend_path,
            batch_size=blend_batch_size, io_workers=blend_io_workers,
        )
    print(f"Phase A done: {n_fields} field channels (aff x{n_fields - 2} + sem + raw) blended over {padded_fine_shape} fine-grid voxels.")

    full_path = out_dir / f"{vol_path.stem}_pred_label_native_full.h5"
    raw_path = out_dir / f"{vol_path.stem}_pred_raw_fine.h5"
    sem_path = out_dir / f"{vol_path.stem}_pred_sem_fine.h5"
    label_fine_path = out_dir / f"{vol_path.stem}_pred_label_fine.h5"
    fine_source_note = (
        f"nanocosmos blockwise inference from {ckpt_path}; Phase A Gaussian-blended over overlapping "
        f"{window} windows (stride {blend_core}) @ {fine_nm} nm fine grid; Phase B per {mws_core}+{mws_ctx} "
        "chunk; diagnostic only -- see infer_submission.py docstring for the cross-chunk-merge limitation."
    )

    with contextlib.ExitStack() as stack:
        f_full = stack.enter_context(h5py.File(str(full_path), "w"))
        full_ds = f_full.create_dataset(
            "main", shape=native_shape, dtype=np.int64,
            chunks=tuple(min(s, 256) for s in native_shape), compression="gzip", compression_opts=4,
        )

        raw_ds = sem_ds = label_fine_ds = None
        if save_fine_grid:
            f_raw = stack.enter_context(h5py.File(str(raw_path), "w"))
            raw_ds = f_raw.create_dataset(
                "main", shape=fine_shape, dtype=np.float32,
                chunks=tuple(min(s, 256) for s in fine_shape), compression="gzip", compression_opts=4,
            )
            raw_ds.attrs["source"] = fine_source_note
            f_sem = stack.enter_context(h5py.File(str(sem_path), "w"))
            sem_ds = f_sem.create_dataset(
                "main", shape=fine_shape, dtype=np.float32,
                chunks=tuple(min(s, 256) for s in fine_shape), compression="gzip", compression_opts=4,
            )
            sem_ds.attrs["source"] = fine_source_note
            f_label_fine = stack.enter_context(h5py.File(str(label_fine_path), "w"))
            label_fine_ds = f_label_fine.create_dataset(
                "main", shape=fine_shape, dtype=np.int64,
                chunks=tuple(min(s, 256) for s in fine_shape), compression="gzip", compression_opts=4,
            )
            label_fine_ds.attrs["source"] = fine_source_note

        print(f"Phase B: Mutex Watershed over the blended field -> {full_path}")
        next_id = _mws_from_blend(
            module, blend_path, mws_blocks, padded_fine_shape,
            native_resolution, fine_nm, native_shape, fine_shape, device_t, sem_threshold,
            full_ds, next_id_start=1, num_workers=mws_workers,
            raw_ds=raw_ds, sem_ds=sem_ds, label_fine_ds=label_fine_ds,
        )
        full_ds.attrs["resolution_zyx_nm"] = np.asarray(native_resolution, dtype=np.float64)
        full_ds.attrs["source"] = (
            f"nanocosmos blockwise inference from {ckpt_path}; Phase A Gaussian-blended sem+aff over "
            f"overlapping {window} windows (stride {blend_core}), Phase B Mutex Watershed per "
            f"{mws_core}+{mws_ctx} chunk of the blended field; see infer_submission.py docstring "
            "for the cross-chunk-merge limitation."
        )
        print(f"Full native-resolution segmentation ({next_id - 1} total ids) -> {full_path}")
        if save_fine_grid:
            print(f"Fine-grid diagnostics -> {raw_path}, {sem_path}, {label_fine_path}")

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
    p.add_argument(
        "--reuse-blend-cache", action="store_true",
        help="If a compatible <vol>_blend_acc.h5 already exists in --out-dir "
             "(e.g. left over from a crashed run that got through Phase A), "
             "skip Phase A and reuse it instead of recomputing from scratch. "
             "Falls back to recomputing (with a warning) if missing/"
             "incompatible -- safe to always pass.",
    )
    p.add_argument(
        "--blend-batch-size", type=int, default=4,
        help="Number of Phase A windows forwarded through the network in a "
             "single batch, to keep the GPU busy instead of running one "
             "window at a time. Raise while GPU memory allows.",
    )
    p.add_argument(
        "--blend-io-workers", type=int, default=4,
        help="Background threads used to read/resample Phase A windows "
             "concurrently and to prefetch the next batch while the "
             "current one runs on the GPU.",
    )
    p.add_argument(
        "--mws-workers", type=int, default=4,
        help="Number of Phase B (Mutex Watershed) chunks processed "
             "concurrently on a thread pool. Real multi-core speedup "
             "requires numba (the CPU MWS backend releases the GIL); with "
             "the GPU backend this mainly overlaps I/O with compute.",
    )
    p.add_argument(
        "--gpu-ids", type=int, nargs="+", default=None, metavar="GPU_ID",
        help="Phase A only: CUDA device indices to scale across, e.g. "
             "'--gpu-ids 0 1 2 3'. Default (unset): use the single --device "
             "below.",
    )
    p.add_argument(
        "--workers-per-gpu", type=int, default=1,
        help="Phase A only: concurrent worker threads per GPU, each with "
             "its own model replica (own default CUDA stream, so >1 runs "
             "multiple batches at once on the same GPU -- helpful when one "
             "batch doesn't saturate it, at the cost of N x that GPU's "
             "model memory).",
    )
    p.add_argument(
        "--save-fine-grid", dest="save_fine_grid", action=argparse.BooleanOptionalAction,
        default=True,
        help="Also save the fine-grid (network-native) diagnostics -- "
             "pred_raw / pred_sem / pred_label_fine -- on top of the "
             "always-produced native-resolution segmentation (default: "
             "true; use --no-save-fine-grid to skip them).",
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
        blend_batch_size=args.blend_batch_size,
        blend_io_workers=args.blend_io_workers,
        mws_workers=args.mws_workers,
        gpu_ids=args.gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        save_fine_grid=args.save_fine_grid,
        reuse_blend_cache=args.reuse_blend_cache,
        device=args.device,
        sem_threshold=args.sem_threshold,
        dry_run=args.dry_run,
        submission_format=args.submission_format,
    )


if __name__ == "__main__":
    main()
