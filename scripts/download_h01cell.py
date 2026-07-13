#!/usr/bin/env python
"""Download H01 (Shapson-Coe et al.) EM + c3 segmentation crops as HDF5.

Sources (public, ``use_https=True`` -- no GCS credentials required)::

  EM:  gs://h01-release/data/20210601/4nm_raw   (4 x 4 x 33 nm, mip 0)
  Seg: gs://h01-release/data/20210601/c3        (8 x 8 x 33 nm, mip 0)

c3 is 2x coarser in xy than the EM, so segmentation is nearest-neighbour
upsampled 2x in Y/X to land on the 4 nm grid (same pattern as FlyWire).

The full release is ``1031784 x 712800 x 5293`` at 4 nm -- only crops are
practical.  A z-extent of 5000 fits for any ``z0 <= 293``.  ``--extend`` uses
``z0=0`` so crops still cover the original ``z=2000..2256`` windows while
avoiding corrupt EM JPEG shards near the volume end (~z>=5221).

Crops are written **z-slab-chunked** so a ``1024 x 1024 x 5000`` uint64
segmentation (~42 GB uncompressed) does not need to fit in RAM.

Examples
--------
    # Re-download the 12 wired mip0 crops at 1024 x 1024 x 5000
    python scripts/download_h01cell.py --extend --skip-existing

    # One custom crop
    python scripts/download_h01cell.py --start 320000 320000 293 --size 1024 1024 5000
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

EM_PATH = "gs://h01-release/data/20210601/4nm_raw"
SEG_PATH = "gs://h01-release/data/20210601/c3"
SOURCE_ATTR = (
    "H01 release (Shapson-Coe et al.); "
    "EM: gs://h01-release/data/20210601/4nm_raw; "
    "Seg: gs://h01-release/data/20210601/c3/ (CC-BY-4.0)"
)
# Physical resolution of the on-disk 4 nm EM grid (z, y, x) nm.
RESOLUTION_ZYX_NM = (33.0, 4.0, 4.0)
# Full EM mip0 depth; z0 + sz must be <= this.
EM_Z_MAX = 5293
# Default extended crop: 5000 of 5293 z-slices.  Start at z=0 (not z=293) so the
# window still covers the original z=2000..2256 crops but avoids corrupt EM JPEG
# shards near the volume end (CloudVolume reshape failures around z>=5221).
DEFAULT_EXTEND_SIZE = (1024, 1024, 5000)
DEFAULT_EXTEND_Z0 = 0

# Original 12 mip0 xy origins (EM 4 nm coords) wired into nanocosmos-2B.yaml.
# (label, x, y) -- z replaced by --extend.
EXISTING_XY: List[Tuple[str, int, int]] = [
    ("train01", 320000, 320000),
    ("train02", 360000, 440000),
    ("train03", 400000, 320000),
    ("train04", 480000, 200000),
    ("train05", 480000, 320000),
    ("train06", 480000, 440000),
    ("train07", 520000, 160000),
    ("train08", 560000, 200000),
    ("train09", 560000, 320000),
    ("train10", 600000, 160000),
    ("test01", 680000, 160000),
    ("test02", 760000, 160000),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", default="data/H01Cell", help="Output directory.")
    p.add_argument(
        "--start", type=int, nargs=3, metavar=("X", "Y", "Z"),
        help="Crop origin in EM mip0 (4 nm) voxels.",
    )
    p.add_argument(
        "--size", type=int, nargs=3, metavar=("X", "Y", "Z"),
        default=list(DEFAULT_EXTEND_SIZE),
        help=f"Crop size in EM mip0 voxels (default: {' '.join(map(str, DEFAULT_EXTEND_SIZE))}).",
    )
    p.add_argument(
        "--extend", action="store_true",
        help=(
            f"Download all 12 existing xy crops at "
            f"{DEFAULT_EXTEND_SIZE[0]}x{DEFAULT_EXTEND_SIZE[1]}x{DEFAULT_EXTEND_SIZE[2]} "
            f"with z0={DEFAULT_EXTEND_Z0} (covers prior z=2000 windows)."
        ),
    )
    p.add_argument(
        "--slab-z", type=int, default=64,
        help="Z-slab height for chunked download/write (default: 64).",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip crops whose volume+seg HDF5 already exist and look valid.",
    )
    return p.parse_args()


def _is_valid_h5(path: Path, expect_shape: Tuple[int, int, int] | None = None) -> bool:
    if not path.exists():
        return False
    try:
        import h5py
        with h5py.File(str(path), "r", locking=False) as f:
            if "main" not in f or f["main"].size == 0:
                return False
            if expect_shape is not None and tuple(f["main"].shape) != tuple(expect_shape):
                return False
            return True
    except Exception:  # noqa: BLE001
        return False


def _open_cloud(path: str, mip: int = 0):
    from cloudvolume import CloudVolume

    return CloudVolume(
        path, mip=mip, use_https=True, fill_missing=True, progress=True,
        bounded=False,
    )


def _fetch_slab_zyx(
    cv,
    start_xyz: Tuple[int, int, int],
    size_xyz: Tuple[int, int, int],
    *,
    label: str,
) -> np.ndarray:
    """Fetch one XYZ cutout as ZYX, with per-slice fallback on corrupt shards.

    H01's sharded JPEG EM has a few truncated chunks near the volume end;
    CloudVolume then raises ``ValueError: cannot reshape array of size ...``.
    On failure, retry slice-by-slice and zero-fill any slice that still fails.
    """
    x0, y0, z0 = start_xyz
    sx, sy, sz = size_xyz
    try:
        data = np.asarray(cv[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz])[..., 0]
        return np.ascontiguousarray(np.transpose(data, (2, 1, 0)))
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN {label}: slab fetch failed ({type(exc).__name__}: {exc})")
        print(f"  WARN {label}: retrying slice-by-slice ...")

    out = np.zeros((sz, sy, sx), dtype=np.dtype(cv.dtype))
    for i in range(sz):
        z = z0 + i
        try:
            sl = np.asarray(cv[x0:x0 + sx, y0:y0 + sy, z:z + 1])[..., 0]
            out[i] = np.ascontiguousarray(np.transpose(sl, (2, 1, 0)))[0]
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN {label}: zero-fill z={z} ({type(exc).__name__})")
    return out


def _stem(size_xyz: Sequence[int], start_xyz: Sequence[int]) -> str:
    sx, sy, sz = size_xyz
    x, y, z = start_xyz
    return f"h01cell_mip0_{sx}x{sy}x{sz}_x{x}_y{y}_z{z}"


def _create_h5(
    path: Path,
    shape_zyx: Tuple[int, int, int],
    dtype: np.dtype,
    chunks_zyx: Tuple[int, int, int],
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "main",
            shape=shape_zyx,
            dtype=dtype,
            chunks=chunks_zyx,
            compression="gzip",
            compression_opts=1,
        )
        ds.attrs["resolution_zyx_nm"] = np.asarray(RESOLUTION_ZYX_NM, dtype=np.float64)
        ds.attrs["source"] = SOURCE_ATTR


def _download_crop(
    out_dir: Path,
    label: str,
    start_xyz: Tuple[int, int, int],
    size_xyz: Tuple[int, int, int],
    *,
    slab_z: int,
    skip_existing: bool,
) -> None:
    import h5py

    x0, y0, z0 = start_xyz
    sx, sy, sz = size_xyz
    if z0 < 0 or z0 + sz > EM_Z_MAX:
        raise ValueError(
            f"z window [{z0}, {z0 + sz}) outside EM depth [0, {EM_Z_MAX})"
        )
    if sx % 2 or sy % 2:
        raise ValueError(f"EM xy size must be even for 2x c3 upsample; got {sx}x{sy}")

    shape_zyx = (sz, sy, sx)
    stem = _stem(size_xyz, start_xyz)
    vol_path = out_dir / f"{stem}_volume.h5"
    seg_path = out_dir / f"{stem}_c3_segmentation.h5"

    print(f"--- {label}: {stem} ---")
    print(f"  EM  start/size : {(x0, y0, z0)} / {(sx, sy, sz)}  (mip 0, 4 nm)")
    print(
        f"  Seg start/size : {(x0 // 2, y0 // 2, z0)} / {(sx // 2, sy // 2, sz)}  "
        f"(mip 0, 8 nm -> 2x upsample)"
    )

    if (
        skip_existing
        and _is_valid_h5(vol_path, shape_zyx)
        and _is_valid_h5(seg_path, shape_zyx)
    ):
        print("  Skip (valid HDF5 exists)\n")
        return

    # Chunks: keep xy full-width slabs for efficient z-slab writes.
    em_chunks = (min(slab_z, sz), min(64, sy), min(64, sx))
    seg_chunks = em_chunks

    need_em = not (skip_existing and _is_valid_h5(vol_path, shape_zyx))
    need_seg = not (skip_existing and _is_valid_h5(seg_path, shape_zyx))

    em_cv = _open_cloud(EM_PATH, 0) if need_em else None
    seg_cv = _open_cloud(SEG_PATH, 0) if need_seg else None

    if need_em:
        print(f"  Writing EM -> {vol_path.name}")
        _create_h5(vol_path, shape_zyx, np.dtype("uint8"), em_chunks)
    if need_seg:
        print(f"  Writing Seg -> {seg_path.name}")
        _create_h5(seg_path, shape_zyx, np.dtype("uint64"), seg_chunks)

    fg_vox = 0
    total_vox = sx * sy * sz
    id_sample: set = set()

    with h5py.File(str(vol_path), "a" if need_em else "r") as f_em, \
         h5py.File(str(seg_path), "a" if need_seg else "r") as f_seg:
        ds_em = f_em["main"] if need_em else None
        ds_seg = f_seg["main"] if need_seg else None

        for z_off in range(0, sz, slab_z):
            cur = min(slab_z, sz - z_off)
            z1, z2 = z0 + z_off, z0 + z_off + cur
            print(f"  slab z=[{z1}, {z2})  ({z_off}/{sz})", flush=True)

            if need_em:
                em_zyx = _fetch_slab_zyx(
                    em_cv, (x0, y0, z1), (sx, sy, cur), label=f"EM z=[{z1},{z2})",
                )
                ds_em[z_off:z_off + cur] = em_zyx
                del em_zyx

            if need_seg:
                seg_native = _fetch_slab_zyx(
                    seg_cv,
                    (x0 // 2, y0 // 2, z1),
                    (sx // 2, sy // 2, cur),
                    label=f"Seg z=[{z1},{z2})",
                )
                # 2x nearest upsample in Y, X
                seg_zyx = np.repeat(np.repeat(seg_native, 2, axis=1), 2, axis=2)
                del seg_native
                if seg_zyx.shape != (cur, sy, sx):
                    raise RuntimeError(
                        f"seg slab shape {seg_zyx.shape} != {(cur, sy, sx)}"
                    )
                ds_seg[z_off:z_off + cur] = seg_zyx
                fg_vox += int((seg_zyx > 0).sum())
                # cheap id sample: unique on this slab only (union approx)
                if z_off == 0 or z_off + cur >= sz or (z_off // slab_z) % 8 == 0:
                    id_sample.update(np.unique(seg_zyx).tolist())
                del seg_zyx

            gc.collect()

    if need_seg:
        id_sample.discard(0)
        fg = 100.0 * fg_vox / max(total_vox, 1)
        print(
            f"  Done. fg≈{fg:.1f}%  unique-ids(sampled)≈{len(id_sample)}  "
            f"shape(z,y,x)={shape_zyx}"
        )
    else:
        print("  Done (reused existing files).")

    print("  Config snippet:")
    print(f"    - vol: {stem}_volume")
    print(f"      seg: {stem}_c3_segmentation")
    print(f"      root: {out_dir}")
    print("      native_resolution: [33, 4, 4]")
    print()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    size_xyz = tuple(args.size)

    if args.extend:
        z0 = DEFAULT_EXTEND_Z0
        size_xyz = DEFAULT_EXTEND_SIZE
        crops = [
            (label, (x, y, z0), size_xyz) for label, x, y in EXISTING_XY
        ]
    else:
        if args.start is None:
            raise SystemExit("Provide --start X Y Z, or pass --extend.")
        crops = [("custom", tuple(args.start), size_xyz)]

    print("=" * 60)
    print("H01 Cell Download")
    print("=" * 60)
    print(f"  Output   : {out_dir}")
    print(f"  Size xyz : {size_xyz}")
    print(f"  Slab z   : {args.slab_z}")
    print(f"  Crops    : {len(crops)}")
    for label, start, size in crops:
        print(f"    {label}: start={start} size={size}")
    print()

    for label, start, size in crops:
        _download_crop(
            out_dir, label, start, size,
            slab_z=args.slab_z, skip_existing=args.skip_existing,
        )

    print("All requested crops finished.")


if __name__ == "__main__":
    main()
