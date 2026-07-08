"""Bake existing local COSEM3D crops down to an exact 4 nm cubic voxel.

One-off data-prep utility, complementary to ``download_cosem3d.py
--resample-isotropic`` (which does the same resample but re-fetches from
CloudVolume). This script instead resamples ALREADY-DOWNLOADED local ``.h5``
crops in place-to-a-NEW-file (never overwrites the source), so it can be run
safely even while a training job is live-reading the *original* files.

Why this exists (see the "for all <4nm, we treat them all as near ~4nm?"
discussion in doc/RESOLUTION_LADDER.md #2.1): the joint datamodule already
*clamps* the reconstruction-target resolution to ``max(native_res, 4nm)`` at
train time (since the model's output grid is fixed at 4nm and can't produce
anything sharper), but it does NOT change the native voxel size used to
determine how many voxels to read from disk for a given physical FOV. If you
just relabel a volume's ``native_resolution`` as ``[4,4,4]`` in the config
WITHOUT actually resampling the data, the loader silently reads a physically
SMALLER field of view and skips the image resample entirely (shapes already
match), silently miscalibrating that volume's physical scale relative to
every correctly-labeled dataset. This script does the resample for real, so
declaring ``native_resolution: [4,4,4]`` in the config is actually true.

Only the ``z`` axis is resampled (COSEM's xy is already 4 nm for every
tier used here); the resample is linear/trilinear in fp32, matching
``download_cosem3d.py``'s ``_resample_isotropic``. Processed in Y-chunks
to keep peak memory bounded regardless of volume size (a naive whole-volume
fp32 resample of a 2048^3 crop would need ~34 GB just for one buffer).

Usage::

    python scripts/resample_cosem_isotropic.py \\
        --root data/COSEM3D \\
        --files jrc_hela-3_4x4x3.24nm_x4096_y0_z2048_volume ... \\
        --out-root data/COSEM3D

Prints the new file stems + a ready-to-paste config snippet at the end.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/COSEM3D", help="Directory containing the source .h5 files.")
    p.add_argument("--out-root", default=None, help="Output directory (default: same as --root).")
    p.add_argument("--files", nargs="+", required=True, help="Volume stems (without .h5) to resample.")
    p.add_argument("--target-nm", type=float, default=4.0, help="Target cubic voxel size (nm).")
    p.add_argument("--chunk-y", type=int, default=200, help="Y-axis chunk size (bounds peak memory).")
    return p.parse_args()


def _new_stem(stem: str, target_nm: float) -> str:
    """``{name}_{oldres}nm_{coords}_volume`` -> ``{name}_{target}nm_{coords}_volume``.

    Mirrors ``download_cosem3d.py``'s naming convention exactly, so the
    resampled files slot into the same directory/config style.
    """
    m = re.match(r"^(?P<name>.+?)_(?P<res>[\d.x]+nm)_(?P<coords>x-?\d+_y-?\d+_z-?\d+)_volume$", stem)
    if not m:
        raise ValueError(f"Stem {stem!r} doesn't match the expected naming convention.")
    tag = f"{target_nm:g}nm"
    return f"{m.group('name')}_{tag}_{m.group('coords')}_volume"


def _resample_z_chunk(chunk_zyx: np.ndarray, z_ratio: float, new_z: int) -> np.ndarray:
    """Resample a ``[Z, y_chunk, X]`` uint8 block along Z only (trilinear,
    fp32, align_corners=False) -- y_chunk/X pass through unchanged since
    their ratio is 1.0 (matches ``download_cosem3d.py::_resample_isotropic``,
    which the whole-volume-at-once case reduces to when Y/X are no-ops).
    """
    del z_ratio  # size (not ratio) drives F.interpolate; kept for clarity at call site
    z, y, x = chunk_zyx.shape
    t = torch.as_tensor(chunk_zyx, dtype=torch.float32)[None, None]  # [1,1,Z,y,X]
    t = F.interpolate(t, size=(new_z, y, x), mode="trilinear", align_corners=False)
    out = t[0, 0].round().clamp(0, 255).numpy().astype(np.uint8)
    return np.ascontiguousarray(out)


def _resample_one(src_path: Path, dst_path: Path, target_nm: float, chunk_y: int) -> Tuple[int, int]:
    with h5py.File(str(src_path), "r", locking=False) as fsrc:
        ds = fsrc["main"]
        z, y, x = ds.shape
        res_zyx = ds.attrs["resolution_zyx_nm"]
        src_z_nm = float(res_zyx[0])
        new_z = max(1, int(round(z * src_z_nm / target_nm)))

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(dst_path), "w") as fdst:
            out_ds = fdst.create_dataset(
                "main", shape=(new_z, y, x), dtype=np.uint8,
                compression="gzip", compression_opts=4, chunks=True,
            )
            for y0 in range(0, y, chunk_y):
                y1 = min(y0 + chunk_y, y)
                block = ds[:, y0:y1, :]  # [Z, y1-y0, X] uint8
                resampled = _resample_z_chunk(block, src_z_nm / target_nm, new_z)
                out_ds[:, y0:y1, :] = resampled
            out_ds.attrs["resolution_zyx_nm"] = np.asarray([target_nm, target_nm, target_nm], dtype=np.float64)
            src_source = ds.attrs.get("source", "")
            out_ds.attrs["source"] = (
                f"{src_source} Resampled {src_z_nm:g}->{ target_nm:g} nm cubic (z-only, trilinear) "
                f"via resample_cosem_isotropic.py from {src_path.name}."
            )
    return z, new_z


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    out_root = Path(args.out_root) if args.out_root else root

    print(f"{'stem':<55} {'old_z':>6} {'new_z':>6}  new_stem")
    snippet_lines: List[str] = []
    for stem in args.files:
        src_path = root / f"{stem}.h5"
        new_stem = _new_stem(stem, args.target_nm)
        dst_path = out_root / f"{new_stem}.h5"
        old_z, new_z = _resample_one(src_path, dst_path, args.target_nm, args.chunk_y)
        print(f"{stem:<55} {old_z:>6} {new_z:>6}  {new_stem}")
        snippet_lines.append(
            f"      - vol: {new_stem}\n"
            f"        root: {out_root}\n"
            f"        native_resolution: [{args.target_nm:g}, {args.target_nm:g}, {args.target_nm:g}]"
        )

    print("\n--- paste-ready config snippet (native_resolution now truthful) ---")
    print("\n".join(snippet_lines))


if __name__ == "__main__":
    main()
