#!/usr/bin/env python
"""Download FlyWire FAFB sub-volumes as MICrONS-style HDF5 crops.

``FLYWIRE`` is the umbrella source for the Princeton/Seung FlyWire female
adult fly brain (FAFB) connectome.  Crops are fetched from public
CloudVolume buckets and written as ``.h5`` files that
``nanocosmos.datasets.FLYWIREDataset`` (== MICRONSDataset layout: key
``main``, axis order ``[Z, Y, X]``) loads directly.

Sources (materialized release v783, no CAVE token required):
  EM:  precomputed://gs://microns-seunglab/drosophila_v0/alignment/image_rechunked
  Seg: precomputed://gs://flywire_v141_m783

Resolution modes (``--resolution``):
  8nm  (default) -- EM mip 1 (8 x 8 x 40 nm) + seg mip 0 (16 x 16 x 40 nm)
                    nearest-neighbour upsampled 2x in XY to match EM.
  16nm           -- EM mip 2 + seg mip 0, both native 16 x 16 x 40 nm.

Predefined ``--split`` provides 10 train + 2 test crops (MICrONS-style),
all inside the proofread segmentation bbox with verified foreground.

Examples
--------
    python scripts/download_flywire.py --split
    python scripts/download_flywire.py --split --resolution 16nm
    python scripts/download_flywire.py --start 44000 10000 3600 --size 512 512 256 --resolution 16nm
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Cloud paths
# ---------------------------------------------------------------------------
EM_PATH = (
    "precomputed://gs://microns-seunglab/drosophila_v0/alignment/image_rechunked"
)
SEG_PATH = "precomputed://gs://flywire_v141_m783"
DEFAULT_SEG_VERSION = 783

# Physical crop extent at 16 nm in-plane (seg voxel units); EM at 8 nm is 2x.
_SEG_SIZE_16 = (512, 512, 256)

# 10 train + 2 test.  ``start`` is the 16 nm seg origin (x, y, z); EM uses
# ``(2x, 2y, z)`` at 8 nm.  Origins verified for dense proofread neuropil.
SPLITS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "train01": {"start": (14000, 16000, 3000), "size": _SEG_SIZE_16},
    "train02": {"start": (16000, 20000, 3500), "size": _SEG_SIZE_16},
    "train03": {"start": (32000, 14000, 1500), "size": _SEG_SIZE_16},
    "train04": {"start": (36000, 18000, 2800), "size": _SEG_SIZE_16},
    "train05": {"start": (44000, 10000, 3600), "size": _SEG_SIZE_16},
    "train06": {"start": (48000, 22000, 4000), "size": _SEG_SIZE_16},
    "train07": {"start": (26000, 16000, 2800), "size": _SEG_SIZE_16},
    "train08": {"start": (30000, 22000, 3200), "size": _SEG_SIZE_16},
    "train09": {"start": (34000,  8000, 2800), "size": _SEG_SIZE_16},
    "train10": {"start": (22000, 12000, 2400), "size": _SEG_SIZE_16},
    "test01":  {"start": (40000,  8000, 4500), "size": _SEG_SIZE_16},
    "test02":  {"start": (52000, 12000, 5000), "size": _SEG_SIZE_16},
}

RESOLUTION_PRESETS = {
    "8nm": {
        "em_mip": 1,
        "seg_mip": 0,
        "seg_upsample_xy": 2,
        "resolution_xyz": (8, 8, 40),
    },
    "16nm": {
        "em_mip": 2,
        "seg_mip": 0,
        "seg_upsample_xy": 1,
        "resolution_xyz": (16, 16, 40),
    },
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="data/FLYWIRE", help="Output directory (default: data/FLYWIRE).")
    p.add_argument(
        "--resolution", default="8nm", choices=sorted(RESOLUTION_PRESETS),
        help="Voxel size for the download (default: 8nm).",
    )
    p.add_argument(
        "--size", type=int, nargs=3, default=list(_SEG_SIZE_16), metavar=("X", "Y", "Z"),
        help="Crop size in 16 nm seg voxels (default: 512 512 256).",
    )
    p.add_argument(
        "--start", type=int, nargs=3, default=[44000, 10000, 3600], metavar=("X", "Y", "Z"),
        help="Crop origin in 16 nm seg voxels when not using --split.",
    )
    p.add_argument(
        "--split", action="store_true",
        help="Download all 12 predefined splits (10 train + 2 test).",
    )
    p.add_argument(
        "--role", default="sft", choices=("sft", "ssl"),
        help="'sft' = image + segmentation; 'ssl' = image only (adds _ssl to stem).",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip crops whose output .h5 already exists and is valid.",
    )
    return p.parse_args()


def _is_valid_h5(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import h5py
        with h5py.File(str(path), "r", locking=False) as f:
            return "main" in f and f["main"].size > 0
    except Exception:  # noqa: BLE001
        return False


def _open_cloud(path: str, mip: int):
    from cloudvolume import CloudVolume

    return CloudVolume(path, mip=mip, use_https=True, fill_missing=True, progress=True)


def _download_cutout(
    cloud_path: str, mip: int, start: Tuple[int, int, int], size: Tuple[int, int, int],
) -> np.ndarray:
    vol = _open_cloud(cloud_path, mip)
    x0, y0, z0 = start
    sx, sy, sz = size
    data = vol[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz]
    arr = np.asarray(data)[..., 0]
    return np.ascontiguousarray(np.transpose(arr, (2, 1, 0)))


def _upsample_seg_xy(seg_zyx: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour upsample segmentation in Y and X (Z unchanged)."""
    if factor == 1:
        return seg_zyx
    return np.ascontiguousarray(np.repeat(np.repeat(seg_zyx, factor, axis=1), factor, axis=2))


def _seg_to_em_coords(
    seg_start: Tuple[int, int, int],
    seg_size: Tuple[int, int, int],
    preset: dict,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """Map a 16 nm seg box to matching EM / output voxel coordinates."""
    sx16, sy16, sz = seg_size
    x16, y16, z0 = seg_start
    up = preset["seg_upsample_xy"]
    if up == 2:
        em_start = (x16 * 2, y16 * 2, z0)
        em_size = (sx16 * 2, sy16 * 2, sz)
        out_size = em_size
    else:
        em_start = seg_start
        em_size = seg_size
        out_size = seg_size
    return em_start, em_size, (x16, y16, z0), (sx16, sy16, sz)


def _save_h5(arr_zyx: np.ndarray, path: Path, resolution_xyz: Tuple[int, int, int]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("main", data=arr_zyx, compression="gzip", compression_opts=4, chunks=True)
        ds.attrs["resolution_zyx_nm"] = np.asarray(
            [resolution_xyz[2], resolution_xyz[1], resolution_xyz[0]], dtype=np.float64,
        )
        ds.attrs["source"] = (
            "FlyWire FAFB v783 (Dorkenwald et al. 2024); "
            "EM: gs://microns-seunglab/drosophila_v0/alignment/image_rechunked; "
            f"Seg: gs://flywire_v141_m{DEFAULT_SEG_VERSION}"
        )


def _make_stem(
    em_mip: int,
    out_size: Tuple[int, int, int],
    em_start: Tuple[int, int, int],
    role_tag: str = "",
    seg_version: int = 0,
) -> str:
    x, y, z = em_start
    sx, sy, sz = out_size
    base = f"flywire_mip{em_mip}_{sx}x{sy}x{sz}_x{x}_y{y}_z{z}{role_tag}"
    if seg_version:
        return f"{base}_m{seg_version}"
    return base


def _quality_report(seg_zyx: np.ndarray) -> Tuple[int, float, float, float]:
    per_z = [(seg_zyx[z] > 0).mean() * 100.0 for z in range(seg_zyx.shape[0])]
    n_ids = int(np.unique(seg_zyx).size) - int((seg_zyx == 0).all())
    fg = float((seg_zyx > 0).mean()) * 100.0
    return n_ids, fg, min(per_z), float(np.std(per_z))


def _download_one(
    out_dir: Path,
    label: str,
    seg_start: Tuple[int, int, int],
    seg_size: Tuple[int, int, int],
    preset: dict,
    *,
    role_tag: str,
    no_seg: bool,
    skip_existing: bool,
) -> None:
    em_start, em_size, seg_origin, seg_fetch_size = _seg_to_em_coords(seg_start, seg_size, preset)
    em_mip = preset["em_mip"]
    seg_mip = preset["seg_mip"]
    res = preset["resolution_xyz"]
    up = preset["seg_upsample_xy"]

    stem = _make_stem(em_mip, em_size, em_start, role_tag=role_tag)
    vol_path = out_dir / f"{stem}_volume.h5"
    seg_path = out_dir / f"{stem}_m{DEFAULT_SEG_VERSION}_segmentation.h5"

    print(f"--- {label} ---")
    print(f"  EM start/size : {em_start} / {em_size}  (mip {em_mip})")
    print(f"  Seg start/size: {seg_origin} / {seg_fetch_size}  (mip {seg_mip}, upsample {up}x)")
    print(f"  Resolution    : {res} nm (x,y,z)")

    if skip_existing and _is_valid_h5(vol_path) and (no_seg or _is_valid_h5(seg_path)):
        print(f"  Skip (valid HDF5 exists): {stem}_*.h5\n")
        return

    if not (skip_existing and _is_valid_h5(vol_path)):
        print("  Downloading EM ...")
        em = _download_cutout(EM_PATH, em_mip, em_start, em_size)
        _save_h5(em, vol_path, res)
        print(f"  Saved image: {vol_path.name}  shape(z,y,x)={em.shape}  dtype={em.dtype}")
    else:
        print(f"  EM: reuse {vol_path.name}")

    if not no_seg:
        if skip_existing and _is_valid_h5(seg_path):
            print(f"  Seg: reuse {seg_path.name}\n")
        else:
            print("  Downloading segmentation ...")
            seg = _download_cutout(SEG_PATH, seg_mip, seg_origin, seg_fetch_size)
            seg = _upsample_seg_xy(seg, up)
            if seg.shape != (em_size[2], em_size[1], em_size[0]):
                raise RuntimeError(f"seg shape {seg.shape} != EM zyx {em_size[2::-1]}")
            _save_h5(seg, seg_path, res)
            n_ids, fg, z_min, z_std = _quality_report(seg)
            print(
                f"  Saved segmentation: {seg_path.name}  shape(z,y,x)={seg.shape}  "
                f"{n_ids} ids, fg {fg:.1f}%, z-min {z_min:.1f}%, z-std {z_std:.2f}"
            )
            if fg < 90.0 or z_min < 80.0:
                print("  WARNING: low foreground or empty z-slices -- review this crop.")

    print()
    print("  Config snippet:")
    print(f"    - vol: {stem}_volume")
    if not no_seg:
        print(f"      seg: {stem}_m{DEFAULT_SEG_VERSION}_segmentation")
    print(f"      root: {out_dir}")
    print(f"      native_resolution: [{res[2]}, {res[0]}, {res[1]}]")
    print()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = RESOLUTION_PRESETS[args.resolution]

    no_seg = args.role == "ssl"
    role_tag = "_ssl" if args.role == "ssl" else ""

    if args.split:
        crops: List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]] = [
            (name, sp["start"], sp["size"]) for name, sp in SPLITS.items()
        ]
    else:
        crops = [("custom", tuple(args.start), tuple(args.size))]

    res = preset["resolution_xyz"]
    print("=" * 60)
    print("FlyWire FAFB Download")
    print("=" * 60)
    print(f"  Output      : {out_dir}")
    print(f"  Resolution  : {args.resolution} ({res[0]} x {res[1]} x {res[2]} nm)")
    print(f"  EM mip      : {preset['em_mip']}")
    print(f"  Seg mip     : {preset['seg_mip']}  (upsample {preset['seg_upsample_xy']}x)")
    print(f"  Role        : {args.role}")
    print(f"  Crops       : {len(crops)}")
    for label, seg_start, seg_size in crops:
        _, em_size, _, _ = _seg_to_em_coords(seg_start, seg_size, preset)
        sx, sy, sz = em_size
        est_gb = (sx * sy * sz) / 1e9 + (0 if no_seg else sx * sy * sz * 8 / 1e9)
        print(f"    {label:8s}: seg_start={seg_start}  em_size={em_size}  ~{est_gb:.2f} GB")
    print()

    for label, seg_start, seg_size in crops:
        _download_one(
            out_dir, label, seg_start, seg_size, preset,
            role_tag=role_tag, no_seg=no_seg, skip_existing=args.skip_existing,
        )

    print("=" * 60)
    print("Download complete!")
    print(f"  Output directory: {out_dir}")
    for f in sorted(out_dir.glob("flywire_mip*.h5")):
        print(f"    {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
