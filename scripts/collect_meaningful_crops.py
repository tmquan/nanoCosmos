#!/usr/bin/env python
"""Probe a CloudVolume source and fetch only *meaningful* (structured) crops.

Companion to ``download_cosem3d.py`` / ``download_flyem3d.py``: those scripts
fetch one crop at a fixed ``--origin``.  This script instead **searches** a
grid of candidate origins, scores each with a *cheap* small probe patch (no
full download), and only pays for the full crop fetch on candidates that pass
the same structure/content gates as ``LazyVolDataset`` (see
``nanocosmos/datasets/lazy.py`` and
``doc/DATASETS.md#ssl-crop-cleaning-content--structure-gates``):

* ``content_frac`` -- fraction of small blocks whose local std clears
  ``--content-std`` (rejects FLAT crops: resin / off-tissue / embedding
  medium).
* ``autocorr`` -- lag-1 in-plane spatial autocorrelation (rejects
  NOISE-dominated crops: detector / resin grain, or corrupt data).  Real EM
  ultrastructure is spatially coherent (~0.6-0.9); white noise is ~0.  A
  content-only gate CANNOT catch this (noise has high local variance).
* ``nz_frac`` -- non-zero voxel fraction (rejects off-tissue / missing-data
  fill, common outside the imaged core of Hemibrain / MaleCNS).

Motivating case: COSEM3D ``jrc_hela-3`` crops centred on the cell edge are
mostly embedding **resin** -- high local variance (passes a plain content
gate) but ~white noise (autocorr ~0.25) -- while ``x4096_y0_z4096`` is genuine
ultrastructure (autocorr ~0.8).  This script finds the latter kind
automatically instead of hand-picking origins and re-downloading blind.

Sources supported (mirrors the existing single-crop scripts' buckets):

    cosem3d    janelia-cosem-datasets S3 bucket (see download_cosem3d.py)
    flyem3d    FlyEM GCS buckets: fib25 / hemibrain / malecns
               (see download_flyem3d.py)

MitoEM2 has **no CloudVolume source** (it ships as local nnU-Net
``.nii.gz`` -- see ``convert_mitoem2.py``), so it is not supported here; audit
already-converted MitoEM2 ``.h5`` files with the same metrics directly (no
network) instead.

Example
-------
    # Search jrc_hela-3 for 2 replacement train crops (dry run first):
    python scripts/collect_meaningful_crops.py --source cosem3d --dataset jrc_hela-3 \\
        --num-crops 2 --dry-run

    # Same, actually fetch:
    python scripts/collect_meaningful_crops.py --source cosem3d --dataset jrc_hela-3 \\
        --num-crops 2

    # FLYEM3D malecns replacement crop, restricted to the known-imaged core
    # (full volume bounds are ~100k^3 -- do NOT grid-search the whole thing):
    python scripts/collect_meaningful_crops.py --source flyem3d --dataset malecns \\
        --role ssl --size 1024 1024 1024 --num-crops 1 \\
        --search-bounds 14000 30000 18000 32000 18000 38000
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

# Single source of truth for the COSEM S3 source lives in download_cosem3d.
from download_cosem3d import _BUCKET as _COSEM_BUCKET, _EM_SUFFIX as _COSEM_EM_SUFFIX

_FLYEM_SOURCES = {
    "fib25": {
        "image": "precomputed://gs://neuroglancer-public-data/flyem_fib-25/image",
        "seg": "precomputed://gs://neuroglancer-public-data/flyem_fib-25/ground_truth",
    },
    "hemibrain": {
        "image": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg",
        "seg": "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
    },
    "malecns": {
        "image": "precomputed://gs://flyem-male-cns/em/em-clahe-jpeg",
        "seg": "precomputed://gs://flyem-male-cns/v1.0/segmentation",
    },
}


# ----------------------------------------------------------------------
# Structure / content metrics -- exact counterparts of
# ``nanocosmos.datasets.lazy.LazyVolDataset._content_fraction`` /
# ``._lag1_autocorr`` (duplicated here, numpy-only, so this script stays a
# standalone downloader like its siblings -- no torch / nanocosmos import).
# ----------------------------------------------------------------------

def _content_fraction(img01: np.ndarray, std_thr: float, block=(4, 16, 16)) -> float:
    if img01.ndim > 3:
        img01 = img01.reshape(img01.shape[-3:])
    z, y, x = img01.shape[-3:]
    bz, by, bx = min(block[0], z), min(block[1], y), min(block[2], x)
    z2, y2, x2 = z - (z % bz), y - (y % by), x - (x % bx)
    if z2 == 0 or y2 == 0 or x2 == 0:
        return float(img01.std() >= std_thr)
    bricks = (
        img01[:z2, :y2, :x2]
        .reshape(z2 // bz, bz, y2 // by, by, x2 // bx, bx)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(-1, bz * by * bx)
    )
    return float(np.mean(bricks.std(axis=1) >= std_thr))


def _lag1_autocorr(img01: np.ndarray) -> float:
    x = img01.astype(np.float32)
    x = x - x.mean()
    den = float((x * x).mean()) + 1e-8
    ay = float((x[..., :-1, :] * x[..., 1:, :]).mean())
    ax = float((x[..., :, :-1] * x[..., :, 1:]).mean())
    return 0.5 * (ay + ax) / den


def _score_probe(probe: np.ndarray, vmin: float, vmax: float, content_std: float):
    """Return ``(content_frac, autocorr, nz_frac)`` for a raw uint8/float probe."""
    scale = (vmax - vmin) or 1.0
    img01 = np.clip((probe.astype(np.float32) - vmin) / scale, 0.0, 1.0)
    nz_frac = float(np.count_nonzero(probe)) / probe.size
    return _content_fraction(img01, content_std), _lag1_autocorr(img01), nz_frac


# ----------------------------------------------------------------------
# CloudVolume helpers
# ----------------------------------------------------------------------

def _open(src: str, mip: int):
    from cloudvolume import CloudVolume

    return CloudVolume(src, mip=mip, use_https=True, progress=False, fill_missing=True)


def _fetch_zyx(vol, x0, y0, z0, x1, y1, z1) -> np.ndarray:
    """CloudVolume returns [X,Y,Z,C]; squeeze channel, move to [Z,Y,X]."""
    img = vol[x0:x1, y0:y1, z0:z1][..., 0]
    return np.ascontiguousarray(np.transpose(img, (2, 1, 0)))


def _estimate_norm_range(vol, probe: int = 512, n_probes: int = 5):
    """Estimate a representative (vmin, vmax) for ``content_frac`` scoring.

    Mirrors ``LazyVolDataset._compute_norm_params``: a handful of small
    centred probes spread across z.  ``autocorr`` is scale-invariant (a ratio
    of covariance to variance, so any constant positive rescale cancels), but
    ``content_frac`` is an ABSOLUTE threshold on the normalised std -- a
    hardcoded ``0-255`` range silently fails every candidate on a volume whose
    true dynamic range is narrower (e.g. a high-key ~140-250 source), because
    dividing by a too-large span shrinks the normalised std well below the
    threshold regardless of how structured the content actually is.
    """
    lo = np.array(vol.bounds.minpt, dtype=np.int64)
    hi = np.array(vol.bounds.maxpt, dtype=np.int64)
    D, H, W = int(hi[2] - lo[2]), int(hi[1] - lo[1]), int(hi[0] - lo[0])
    cx, cy = lo[0] + W // 2, lo[1] + H // 2
    half_x, half_y = min(probe, W) // 2, min(probe, H) // 2
    z_indices = sorted(set(
        max(lo[2], min(lo[2] + int(i), hi[2] - 1))
        for i in np.linspace(0, D - 1, n_probes)
    ))
    vmin, vmax = float("inf"), float("-inf")
    for z in z_indices:
        try:
            patch = _fetch_zyx(
                vol, int(cx - half_x), int(cy - half_y), int(z),
                int(cx + half_x), int(cy + half_y), int(z) + 1,
            )
        except Exception:  # noqa: BLE001 -- fall back to the dtype-implied range
            continue
        vmin = min(vmin, float(patch.min()))
        vmax = max(vmax, float(patch.max()))
    if not np.isfinite(vmin) or vmax <= vmin:
        return 0.0, 255.0
    return vmin, vmax


def _save_h5(arr_zyx: np.ndarray, path: Path, resolution_xyz, source_note: str) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "main", data=arr_zyx, compression="gzip", compression_opts=4, chunks=True,
        )
        ds.attrs["resolution_zyx_nm"] = np.asarray(
            [resolution_xyz[2], resolution_xyz[1], resolution_xyz[0]], dtype=np.float64,
        )
        ds.attrs["source"] = source_note


def _existing_origins(out_dir: Path, name_prefix: str) -> list:
    """Parse ``x{X}_y{Y}_z{Z}`` origins already on disk for this name prefix,
    so the grid search skips cells that overlap an already-downloaded crop."""
    import re

    origins = []
    for p in out_dir.glob(f"{name_prefix}*_volume.h5"):
        m = re.search(r"x(\d+)_y(\d+)_z(\d+)", p.name)
        if m:
            origins.append(tuple(int(g) for g in m.groups()))
    return origins


def _overlaps(o1, size1, o2, size2) -> bool:
    return all(
        o1[i] < o2[i] + size2[i] and o2[i] < o1[i] + size1[i] for i in range(3)
    )


# ----------------------------------------------------------------------
# Candidate grid + probing
# ----------------------------------------------------------------------

def _grid_origins(lo, hi, size, stride=None):
    """Non-overlapping (or ``stride``-spaced) candidate origins within [lo, hi)."""
    stride = stride or size
    axes = []
    for a in range(3):
        span = max(1, hi[a] - lo[a] - size[a])
        pts = sorted(set(range(lo[a], lo[a] + span + 1, max(1, stride[a]))) | {lo[a]})
        pts = [p for p in pts if p + size[a] <= hi[a]] or [lo[a]]
        axes.append(pts)
    return list(itertools.product(*axes))


def _probe_and_rank(
    vol, candidates, size, probe_size, content_std, min_content, min_autocorr, min_nz,
    num_probes: int = 4, seed: int = 0, norm_range=(0.0, 255.0),
):
    """Score each candidate cell by averaging several RANDOM sub-patches inside
    it (not one fixed centred probe).

    A single small probe fixed at the cell's geometric centre is not
    representative of a large (e.g. 2048^3) candidate: it can land squarely
    inside a locally homogeneous sub-region (a nucleus, a patch of resin) that
    does not reflect the cell's overall character, giving spuriously
    high/low scores.  Averaging ``num_probes`` RANDOM sub-patches -- sized like
    an actual training crop -- mirrors how ``LazyVolDataset`` itself samples,
    and matches the calibration that established the ``content_frac`` /
    ``autocorr`` thresholds in the first place.
    """
    lo = np.array(vol.bounds.minpt, dtype=np.int64)
    hi = np.array(vol.bounds.maxpt, dtype=np.int64)
    rng = np.random.default_rng(seed)
    results = []
    for origin in candidates:
        ox, oy, oz = origin
        cell_hi = (min(ox + size[0], hi[0]), min(oy + size[1], hi[1]), min(oz + size[2], hi[2]))
        cell_lo = (max(ox, lo[0]), max(oy, lo[1]), max(oz, lo[2]))
        span = [max(1, cell_hi[a] - cell_lo[a] - probe_size[a]) for a in range(3)]
        if any(cell_hi[a] - cell_lo[a] < probe_size[a] for a in range(3)):
            continue
        metrics = []
        for _ in range(num_probes):
            cx = cell_lo[0] + int(rng.integers(0, span[0] + 1))
            cy = cell_lo[1] + int(rng.integers(0, span[1] + 1))
            cz = cell_lo[2] + int(rng.integers(0, span[2] + 1))
            try:
                probe = _fetch_zyx(
                    vol, cx, cy, cz,
                    cx + probe_size[0], cy + probe_size[1], cz + probe_size[2],
                )
            except Exception as exc:  # noqa: BLE001 -- skip unreachable probes
                print(f"  probe failed at x{cx}_y{cy}_z{cz}: {exc}")
                continue
            metrics.append(_score_probe(probe, norm_range[0], norm_range[1], content_std))
        if not metrics:
            continue
        cf, ac, nz = np.mean(metrics, axis=0)
        ok = cf >= min_content and ac >= min_autocorr and nz >= min_nz
        score = min(cf / max(min_content, 1e-6), ac / max(min_autocorr, 1e-6), nz / max(min_nz, 1e-6))
        results.append((score if ok else -1.0, origin, float(cf), float(ac), float(nz)))
    results.sort(key=lambda r: r[0], reverse=True)
    return results


def _select_non_overlapping(ranked, size, existing, num_crops):
    chosen = []
    for score, origin, cf, ac, nz in ranked:
        if score < 0 or len(chosen) >= num_crops:
            continue
        if any(_overlaps(origin, size, e, size) for e in existing):
            continue
        if any(_overlaps(origin, size, c[1], size) for c in chosen):
            continue
        chosen.append((score, origin, cf, ac, nz))
    return chosen


# ----------------------------------------------------------------------
# Per-source drivers
# ----------------------------------------------------------------------

def _run_cosem3d(args) -> None:
    name = args.name or args.dataset
    src = f"precomputed://{_COSEM_BUCKET}/{args.dataset}/{args.em_path or _COSEM_EM_SUFFIX}"
    vol = _open(src, args.mip)
    res = [float(r) for r in vol.resolution]
    lo = np.array(vol.bounds.minpt, dtype=np.int64)
    hi = np.array(vol.bounds.maxpt, dtype=np.int64)
    size = tuple(args.size)
    print(f"{args.dataset}: bounds {tuple(lo)} .. {tuple(hi)}  resolution(x,y,z)={res} nm")

    if args.search_bounds:
        sb = args.search_bounds
        lo = np.maximum(lo, [sb[0], sb[2], sb[4]])
        hi = np.minimum(hi, [sb[1], sb[3], sb[5]])

    norm_range = _estimate_norm_range(vol)
    print(f"Estimated normalisation range for content_frac scoring: "
          f"{norm_range[0]:.0f}-{norm_range[1]:.0f} (autocorr is scale-invariant, unaffected)")
    candidates = _grid_origins(tuple(lo), tuple(hi), size, args.grid_stride)
    print(f"Probing {len(candidates)} candidate origins "
          f"({args.num_probes} x probe size {tuple(args.probe_size)}) ...")
    ranked = _probe_and_rank(
        vol, candidates, size, tuple(args.probe_size), args.content_std,
        args.min_content_frac, args.min_autocorr, args.min_nz_frac,
        num_probes=args.num_probes, norm_range=norm_range,
    )
    res_tag = "x".join(f"{r:g}" for r in res) + "nm"
    existing = _existing_origins(Path(args.out_dir), f"{name}_{res_tag}_")
    chosen = _select_non_overlapping(ranked, size, existing, args.num_crops)

    print(f"\n{'origin':>22} {'content_frac':>13} {'autocorr':>9} {'nz_frac':>8}  status")
    for score, origin, cf, ac, nz in ranked[:20]:
        tag = "SELECTED" if any(origin == c[1] for c in chosen) else ("ok" if score >= 0 else "reject")
        print(f"  x{origin[0]}_y{origin[1]}_z{origin[2]:<10} {cf:>13.2f} {ac:>9.2f} {nz:>8.2f}  {tag}")
    if not chosen:
        print("\nNo candidate passed the gates -- widen --search-bounds or lower thresholds.")
        return

    print(f"\n{len(chosen)} crop(s) selected. Existing on-disk origins skipped: {existing}")
    if args.dry_run:
        print("--dry-run: not fetching. Re-run without --dry-run to download these crops.")
        return

    out_dir = Path(args.out_dir)
    print("\nConfig volume entries (paste under data.branches.ssl.volumes):")
    for _score, origin, cf, ac, nz in chosen:
        ox, oy, oz = origin
        # Clamp the full-fetch box to the volume bounds (e.g. COSEM's thin y
        # axis is far smaller than the nominal --size).
        x1 = min(ox + size[0], int(hi[0]))
        y1 = min(oy + size[1], int(hi[1]))
        z1 = min(oz + size[2], int(hi[2]))
        stem = f"{name}_{res_tag}_x{ox}_y{oy}_z{oz}"
        out_path = out_dir / f"{stem}_volume.h5"
        print(f"Fetching {stem} (content_frac={cf:.2f} autocorr={ac:.2f} nz_frac={nz:.2f}) ...")
        img_zyx = _fetch_zyx(vol, ox, oy, oz, x1, y1, z1)
        _save_h5(
            img_zyx, out_path, res,
            f"OpenOrganelle / COSEM {args.dataset} "
            f"(s3://janelia-cosem-datasets/{args.dataset}); image-only, SSL branch. "
            f"Auto-selected via collect_meaningful_crops.py "
            f"(content_frac={cf:.2f}, autocorr={ac:.2f}).",
        )
        print(f"Saved: {out_path.name}  shape(z,y,x)={img_zyx.shape}")
        print(f"  - vol: {stem}_volume")
        print(f"    root: {args.out_dir}")
        print("    # image-only; SSL (no seg)")


def _run_flyem3d(args) -> None:
    if args.dataset not in _FLYEM_SOURCES:
        raise SystemExit(f"Unknown flyem3d dataset {args.dataset!r}; choose from {sorted(_FLYEM_SOURCES)}.")
    name = args.name or ("flyem3d" if args.dataset == "fib25" else f"flyem3d_{args.dataset}")
    image_src = _FLYEM_SOURCES[args.dataset]["image"]
    vol = _open(image_src, args.mip)
    res = [int(r) for r in vol.resolution]
    lo = np.array(vol.bounds.minpt, dtype=np.int64)
    hi = np.array(vol.bounds.maxpt, dtype=np.int64)
    size = tuple(args.size)
    print(f"{args.dataset}: full bounds {tuple(lo)} .. {tuple(hi)}  resolution(x,y,z)={res} nm")

    if args.search_bounds:
        sb = args.search_bounds
        lo = np.maximum(lo, [sb[0], sb[2], sb[4]])
        hi = np.minimum(hi, [sb[1], sb[3], sb[5]])
    elif args.dataset in ("hemibrain", "malecns"):
        raise SystemExit(
            f"{args.dataset} bounds are ~{tuple(hi)} voxels -- grid-searching the FULL "
            "volume is impractical (mostly off-tissue). Pass --search-bounds x0 x1 y0 y1 "
            "z0 z1 restricted to the known-imaged core (see doc/DATASETS.md)."
        )
    print(f"Search region: {tuple(lo)} .. {tuple(hi)}")

    norm_range = _estimate_norm_range(vol)
    print(f"Estimated normalisation range for content_frac scoring: "
          f"{norm_range[0]:.0f}-{norm_range[1]:.0f} (autocorr is scale-invariant, unaffected)")
    candidates = _grid_origins(tuple(lo), tuple(hi), size, args.grid_stride)
    print(f"Probing {len(candidates)} candidate origins "
          f"({args.num_probes} x probe size {tuple(args.probe_size)}) ...")
    ranked = _probe_and_rank(
        vol, candidates, size, tuple(args.probe_size), args.content_std,
        args.min_content_frac, args.min_autocorr, args.min_nz_frac,
        num_probes=args.num_probes, norm_range=norm_range,
    )
    role_tag = "_ssl" if args.role == "ssl" else ""
    existing = _existing_origins(Path(args.out_dir), f"{name}_{res[0]}nm{role_tag}_")
    chosen = _select_non_overlapping(ranked, size, existing, args.num_crops)

    print(f"\n{'origin':>22} {'content_frac':>13} {'autocorr':>9} {'nz_frac':>8}  status")
    for score, origin, cf, ac, nz in ranked[:20]:
        tag = "SELECTED" if any(origin == c[1] for c in chosen) else ("ok" if score >= 0 else "reject")
        print(f"  x{origin[0]}_y{origin[1]}_z{origin[2]:<10} {cf:>13.2f} {ac:>9.2f} {nz:>8.2f}  {tag}")
    if not chosen:
        print("\nNo candidate passed the gates -- widen --search-bounds or lower thresholds.")
        return

    print(f"\n{len(chosen)} crop(s) selected. Existing on-disk origins skipped: {existing}")
    if args.dry_run:
        print("--dry-run: not fetching. Re-run without --dry-run to download these crops.")
        return

    no_seg = args.role == "ssl"
    seg_vol = None if no_seg else _open(_FLYEM_SOURCES[args.dataset]["seg"], args.mip)
    out_dir = Path(args.out_dir)
    where = "data.branches.ssl.volumes" if args.role == "ssl" else "data.branches.sft.volumes"
    print(f"\nConfig volume entries (paste under {where}):")
    for _score, origin, cf, ac, nz in chosen:
        ox, oy, oz = origin
        x1 = min(ox + size[0], int(hi[0]))
        y1 = min(oy + size[1], int(hi[1]))
        z1 = min(oz + size[2], int(hi[2]))
        stem = f"{name}_{res[0]}nm{role_tag}_x{ox}_y{oy}_z{oz}"
        out_path = out_dir / f"{stem}_volume.h5"
        print(f"Fetching {stem} (content_frac={cf:.2f} autocorr={ac:.2f} nz_frac={nz:.2f}) ...")
        img_zyx = _fetch_zyx(vol, ox, oy, oz, x1, y1, z1)
        _save_h5(
            img_zyx, out_path, res,
            f"FlyEM {args.dataset} (auto-selected via collect_meaningful_crops.py; "
            f"content_frac={cf:.2f}, autocorr={ac:.2f}).",
        )
        print(f"Saved image: {out_path.name}  shape(z,y,x)={img_zyx.shape}")
        print(f"  - vol: {stem}_volume")
        if seg_vol is not None:
            seg_zyx = _fetch_zyx(seg_vol, ox, oy, oz, x1, y1, z1)
            seg_path = out_dir / f"{stem}_segmentation.h5"
            _save_h5(seg_zyx, seg_path, res, f"FlyEM {args.dataset} ground truth.")
            fg = float((seg_zyx > 0).mean()) * 100.0
            print(f"Saved segmentation: {seg_path.name}  fg={fg:.1f}%")
            print(f"    seg: {stem}_segmentation")
        print(f"    root: {args.out_dir}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=("cosem3d", "flyem3d"))
    p.add_argument("--dataset", required=True, help="cosem3d: jrc_hela-3 etc. flyem3d: fib25/hemibrain/malecns.")
    p.add_argument("--role", default="ssl", choices=("ssl", "sft"), help="flyem3d only: image-only vs image+seg.")
    p.add_argument("--out-dir", default=None, help="Default: data/COSEM3D or data/FLYEM3D.")
    p.add_argument("--name", default=None, help="Crop basename prefix override.")
    p.add_argument("--size", type=int, nargs=3, metavar=("X", "Y", "Z"), default=[2048, 2048, 2048])
    p.add_argument("--mip", type=int, default=0)
    p.add_argument("--em-path", default=None, dest="em_path", help="cosem3d only: override the EM sub-path.")
    p.add_argument("--num-crops", type=int, default=1, dest="num_crops")
    p.add_argument(
        "--search-bounds", type=int, nargs=6, default=None,
        metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
        help="Restrict the candidate grid to this box (required for flyem3d "
             "hemibrain/malecns -- their full bounds are ~10^5 vox/axis).",
    )
    p.add_argument("--grid-stride", type=int, nargs=3, default=None, help="Default: --size (non-overlapping grid).")
    p.add_argument(
        "--probe-size", type=int, nargs=3, default=[48, 128, 128], dest="probe_size",
        help="Sub-patch size for scoring (default matches the SSL training patch "
             "scale -- see doc/RESOLUTION_LADDER.md).",
    )
    p.add_argument(
        "--num-probes", type=int, default=4, dest="num_probes",
        help="Random sub-patches averaged per candidate cell (not one fixed centred probe).",
    )
    p.add_argument("--content-std", type=float, default=0.05, dest="content_std")
    p.add_argument("--min-content-frac", type=float, default=0.8, dest="min_content_frac")
    p.add_argument("--min-autocorr", type=float, default=0.5, dest="min_autocorr")
    p.add_argument("--min-nz-frac", type=float, default=0.5, dest="min_nz_frac")
    p.add_argument("--dry-run", action="store_true", dest="dry_run", help="Probe + rank only; no downloads.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.out_dir is None:
        args.out_dir = "data/COSEM3D" if args.source == "cosem3d" else "data/FLYEM3D"
    if args.source == "cosem3d":
        _run_cosem3d(args)
    else:
        _run_flyem3d(args)


if __name__ == "__main__":
    main()
