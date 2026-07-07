#!/usr/bin/env python
"""Affinity-weighted global fragment stitch for chunked Mutex-Watershed output.

Chunked Phase B inference (see ``infer_submission.py``) assigns a fresh id
range per chunk with no cross-chunk merging, so a single true neuron that
spans multiple chunks (or that Mutex Watershed over-split even within one
chunk) ends up as multiple fragment ids. Shrinking Phase B's chunk size to
get good CPU-worker parallelism (many small chunks -> many workers fit in
RAM) makes this worse, trading segmentation accuracy for speed.

This script recovers accuracy cheaply, as a POST-PROCESS, without touching
Phase A/B or re-running any network inference or Mutex Watershed:

  1. Build a region-adjacency graph (RAG) directly from the FINE-grid label
     volume (``<vol>_pred_label_fine.h5``, requires ``--save-fine-grid`` to
     have been on): every pair of touching, differently-labelled fragments
     is an edge.
  2. Weight each edge by the MEAN of the network's own short-range ("pull")
     affinity prediction across the shared boundary -- read straight from
     the kept Phase A blend accumulator (``<vol>_blend_acc.h5``, requires
     ``--keep-blend-cache``), i.e. exactly the same signal Mutex Watershed
     itself would have used had the two fragments been agglomerated in one
     chunk. This is what distinguishes "these two fragments are the same
     neuron, artificially split by a chunk boundary or an over-eager small
     chunk" from "these are two different neurons that happen to touch".
  3. Union-merge every edge whose mean affinity clears ``--threshold`` (and
     has at least ``--min-support`` boundary voxels, to avoid one noisy
     corner-touch voxel driving a merge).
  4. Apply the resulting id-merge map to the NATIVE-resolution segmentation,
     relabel consecutively, and repackage the challenge submission file
     (uint16 if it fits after merging, else falls back to the smallest
     integer dtype that does).

Affinity channel convention (see
``nanocosmos.losses._common.affinity_target_from_offsets``):
    aff[c, v] = 1{labels[v] == labels[v + offsets[c]]}
i.e. the value is stored at voxel ``v``, referring to the pair
``(v, v + offset)``. Only the first 3 offsets (index 0,1,2 = the direct
z/y/x nearest-neighbour "pull" edges, ``n_pull=5`` >= 3 in
``configs/nanocosmos-2B.yaml``) are used here -- they are the unambiguous
"P(same instance across this single-voxel step)" signal; longer-range
offsets are left alone since Mutex Watershed uses them as push (mutex/
exclusion) constraints internally, not simple attractive merge evidence.

Usage:
    python scripts/stitch_affinity_rag.py \\
        --out-dir outputs/submission/snemi3d_AC3 --vol AC3_inputs \\
        --submission-format snemi3d
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np


def _log(t0: float, msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _accumulate_axis_pairs(
    lab: np.ndarray, aff: np.ndarray, w: np.ndarray, axis: int,
    pair_sum: dict, pair_cnt: dict, min_label: int = 1,
) -> None:
    """Accumulate (sum_affinity, count) per unordered fragment-id pair for
    every boundary face along ``axis`` where the two neighbouring voxels
    carry different nonzero labels. ``aff``/``w`` are the RAW accumulator
    channel + weight (not yet normalised); normalise lazily only at the
    (relatively few) boundary voxels to avoid a full-volume division."""
    a = np.take(lab, range(0, lab.shape[axis] - 1), axis=axis)
    b = np.take(lab, range(1, lab.shape[axis]), axis=axis)
    diff = (a != b) & (a >= min_label) & (b >= min_label)
    if not diff.any():
        return
    lo = np.minimum(a[diff], b[diff]).astype(np.int64)
    hi = np.maximum(a[diff], b[diff]).astype(np.int64)
    # Affinity value for offset (..,-1,..) at position p is stored at the
    # LARGER-index side of the pair -- i.e. the "b" (index 1:) slice here.
    aff_b = np.take(aff, range(1, lab.shape[axis]), axis=axis)
    w_b = np.take(w, range(1, lab.shape[axis]), axis=axis)
    a_val = (aff_b[diff] / (w_b[diff] + 1e-8))
    a_val = 1.0 / (1.0 + np.exp(-a_val))  # sigmoid -> probability
    # Pair as python (lo, hi) int tuples -- avoids any risk of an integer-
    # encoding collision regardless of how large fragment ids get (with
    # many small chunks, ids can run well into the hundreds of thousands).
    pairs = np.stack([lo, hi], axis=1)
    uk, inv = np.unique(pairs, axis=0, return_inverse=True)
    sums = np.bincount(inv, weights=a_val.astype(np.float64), minlength=len(uk))
    cnts = np.bincount(inv, minlength=len(uk))
    for (l, h), s, c in zip(uk.tolist(), sums.tolist(), cnts.tolist()):
        k = (l, h)
        if k in pair_sum:
            pair_sum[k] += s
            pair_cnt[k] += c
        else:
            pair_sum[k] = s
            pair_cnt[k] = c


def build_merge_map(
    label_fine_path: Path, blend_acc_path: Path, threshold: float, min_support: int,
    z_slab: int = 128,
) -> dict:
    """Stream the fine-grid label volume + accumulator in z-slabs (with a
    1-voxel overlap so cross-slab boundaries aren't missed), accumulate
    per-fragment-pair mean affinity, and return ``{old_id: new_root_id}``."""
    t0 = time.time()
    pair_sum: dict = {}
    pair_cnt: dict = {}

    with h5py.File(str(label_fine_path), "r") as flab, h5py.File(str(blend_acc_path), "r") as facc:
        lab_ds = flab["main"]
        acc_ds = facc["acc"]
        w_ds = facc["weight"]
        fine_shape = lab_ds.shape
        n_z = fine_shape[0]
        _log(t0, f"fine label shape {fine_shape}, accumulator acc {acc_ds.shape} weight {w_ds.shape}")

        for z0 in range(0, n_z, z_slab):
            z1 = min(z0 + z_slab, n_z)
            z1_ov = min(z1 + 1, n_z)  # +1 row overlap so the z-boundary at the slab edge is counted
            lab_blk = lab_ds[z0:z1_ov].astype(np.int64)
            aff_blk = acc_ds[0:3, z0:z1_ov]  # channels 0,1,2 = z,y,x direct-neighbour pull affinities
            w_blk = w_ds[z0:z1_ov]

            _accumulate_axis_pairs(lab_blk, aff_blk[0], w_blk, axis=0, pair_sum=pair_sum, pair_cnt=pair_cnt)
            _accumulate_axis_pairs(lab_blk, aff_blk[1], w_blk, axis=1, pair_sum=pair_sum, pair_cnt=pair_cnt)
            _accumulate_axis_pairs(lab_blk, aff_blk[2], w_blk, axis=2, pair_sum=pair_sum, pair_cnt=pair_cnt)
            _log(t0, f"z-slab [{z0}:{z1}) done ({len(pair_sum):,} pairs so far)")

    _log(t0, f"total boundary fragment pairs: {len(pair_sum):,}")

    # ---- union-find over fragment ids ----
    keys = list(pair_sum.keys())
    max_id = max((hi for _, hi in keys), default=0)
    parent = np.arange(max_id + 1, dtype=np.int64)

    def find(x: int) -> int:
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n_merged = 0
    for lo, hi in keys:
        cnt = pair_cnt[(lo, hi)]
        if cnt < min_support:
            continue
        mean_aff = pair_sum[(lo, hi)] / cnt
        if mean_aff >= threshold:
            union(int(lo), int(hi))
            n_merged += 1
    _log(t0, f"merged {n_merged:,} / {len(keys):,} candidate pairs (threshold={threshold}, min_support={min_support})")

    roots = {}
    for i in range(max_id + 1):
        roots[i] = int(find(i))
    return roots


def apply_merge_and_package(
    native_label_path: Path, merge_map: dict, out_dir: Path, vol_stem: str,
    element_size_um, submission_format: str,
) -> Path:
    t0 = time.time()
    with h5py.File(str(native_label_path), "r") as f:
        arr = f["main"][:]
    max_id = int(arr.max())
    _log(t0, f"native label loaded {arr.shape}, max_id={max_id}, fg_ids(before)={len(np.unique(arr)) - 1:,}")

    root_lut = np.arange(max_id + 1, dtype=np.int64)
    for old_id, root in merge_map.items():
        if old_id <= max_id:
            root_lut[old_id] = root
    roots = root_lut[arr]

    uniq = np.unique(roots)
    relabel = np.zeros(int(uniq.max()) + 1, dtype=np.int64)
    nid = 1
    for r in uniq.tolist():
        if r == 0:
            continue
        relabel[r] = nid
        nid += 1
    final = relabel[roots]
    n_fg = nid - 1
    _log(t0, f"after merge: fg_ids={n_fg:,}")

    dtype = np.uint16 if n_fg <= 65535 else (np.uint32 if n_fg <= 2**32 - 1 else np.int64)
    final = final.astype(dtype)
    _log(t0, f"output dtype={dtype.__name__}")

    stitched_path = out_dir / f"{vol_stem}_pred_label_native_stitched.h5"
    with h5py.File(str(stitched_path), "w") as f:
        d = f.create_dataset("main", data=final, compression="gzip", compression_opts=4)
        d.attrs["element_size_um"] = np.asarray(element_size_um, dtype=np.float32)
        d.attrs["source"] = (
            "affinity-weighted RAG stitch of chunked Mutex-Watershed fragments "
            "(scripts/stitch_affinity_rag.py) over " + str(native_label_path)
        )
    _log(t0, f"wrote {stitched_path}")

    submission_path = out_dir / f"{vol_stem}_submission.zip"
    if submission_format == "snemi3d":
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            h5_tmp = Path(td) / "test-input.h5"
            with h5py.File(str(h5_tmp), "w") as f:
                d = f.create_dataset("main", data=final.astype(np.uint16 if dtype != np.int64 else np.uint32))
                d.attrs["element_size_um"] = np.asarray(element_size_um, dtype=np.float32)
            with zipfile.ZipFile(str(submission_path), "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(str(h5_tmp), arcname="test-input.h5")
        _log(t0, f"wrote submission zip {submission_path}")
    else:
        raise NotImplementedError(f"submission_format={submission_format!r} not implemented in this script yet")

    return submission_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--vol", required=True, help="volume stem, e.g. AC3_inputs")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-support", type=int, default=4, help="min boundary voxels between a pair before merging")
    p.add_argument("--z-slab", type=int, default=128)
    p.add_argument("--element-size-um", type=float, nargs=3, default=[0.030, 0.006, 0.006])
    p.add_argument("--submission-format", default="snemi3d", choices=["snemi3d"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    label_fine_path = out_dir / f"{args.vol}_pred_label_fine.h5"
    blend_acc_path = out_dir / f"{args.vol}_blend_acc.h5"
    native_label_path = out_dir / f"{args.vol}_pred_label_native_full.h5"
    for req in (label_fine_path, blend_acc_path, native_label_path):
        if not req.exists():
            raise FileNotFoundError(f"required input missing: {req}")

    merge_map = build_merge_map(
        label_fine_path, blend_acc_path, threshold=args.threshold,
        min_support=args.min_support, z_slab=args.z_slab,
    )
    apply_merge_and_package(
        native_label_path, merge_map, out_dir, args.vol,
        element_size_um=args.element_size_um, submission_format=args.submission_format,
    )


if __name__ == "__main__":
    main()
