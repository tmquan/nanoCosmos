#!/usr/bin/env python
"""Download a COSEM3D (COSEM / OpenOrganelle) 4 nm small-voxel FIB-SEM crop as brainbow HDF5.

Ladder source name: **COSEM3D** (the smallest-voxel rung; doc/RESOLUTION_LADDER.md).
Data lives under ``data/COSEM3D``; the upstream bucket is the COSEM /
OpenOrganelle ``s3://janelia-cosem-datasets`` (unchanged).

The OpenOrganelle (COSEM) collection is the only public source of genuinely
**~4 nm, near-cubic (small-voxel)** FIB-SEM, which anchors the
*smallest-voxel* rung of the nanoCosmos resolution ladder (the DAPT
super-resolution prior; see
doc/RESOLUTION_LADDER.md).  Volumes are served in Neuroglancer ``precomputed``
from the AWS Open Data bucket ``s3://janelia-cosem-datasets`` (no account
needed), so we crop them the same way the MICrONS / FIB-25 volumes are fetched
-- via CloudVolume random access -- and write the image as an ``.h5`` file that
``nanocosmos.datasets.MICRONSDataset`` loads directly (dataset key ``main``,
axis order ``[Z, Y, X]``).

These are **cell-biology** FIB-SEM (HeLa, Jurkat, macrophage, ...), not
neuropil: we use them **image-only**, for the label-free DAPT branch (their
labels are organelle classes, not neuron instances).  They transfer the fine
ultrastructure / z-continuity prior; pair with unlabeled FIB-25 for a
domain-matched DAPT mix.

Verified mip-0 voxel sizes (x, y, z nm), read from each volume's ``info``:

    jrc_hela-3        4 x 4 x 3.24      jrc_jurkat-1      4 x 4 x 3.44
    jrc_macrophage-2  4 x 4 x 3.36      jrc_hela-2        4 x 4 x 5.24

(Resolution doubles per mip; mip 0 is the ~4 nm native scale.)

DAPT budget (doc/RESOLUTION_LADDER.md): **5x 2048^3 per COSEM volume**.  The
volumes are thin in y (1000-3000 vox), so a 2048 y-request is clamped to the
volume; the 5 crops tile x/z.  One crop per invocation -- loop over origins:

Example
-------
    # 5x 2048^3 crops of jrc_hela-3 (y clamps to ~1000), tiling x/z:
    for ox in 2048 6144; do for oz in 2048 6144; do \
      python scripts/download_cosem3d.py --dataset jrc_hela-3 \
        --out-dir data/COSEM3D --origin $ox 0 $oz --size 2048 2048 2048; done; done
    # (run 5 origins total per volume; repeat for jrc_macrophage-2, jrc_jurkat-1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Per-dataset Neuroglancer ``precomputed`` EM source on the public S3 bucket.
# Path template is stable across COSEM datasets; add more keys as needed.
_BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com"
_EM_SUFFIX = "neuroglancer/em/fibsem-uint8.precomputed"
KNOWN = (
    "jrc_hela-3",
    "jrc_jurkat-1",
    "jrc_macrophage-2",
    "jrc_hela-2",
)


def _em_src(dataset: str, em_path: str | None) -> str:
    suffix = em_path or _EM_SUFFIX
    return f"precomputed://{_BUCKET}/{dataset}/{suffix}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="jrc_hela-3",
        help=f"COSEM dataset id (known: {', '.join(KNOWN)}; others work too "
             f"if they expose {_EM_SUFFIX}).",
    )
    p.add_argument("--out-dir", default="data/COSEM3D", help="Output directory for the .h5 crop.")
    p.add_argument("--name", default=None, help="Crop basename prefix (default: the dataset id).")
    p.add_argument(
        "--origin", type=int, nargs=3, metavar=("X", "Y", "Z"), default=None,
        help="Crop origin in voxels (x y z) at the chosen mip. Default: centred.",
    )
    p.add_argument(
        "--size", type=int, nargs=3, metavar=("X", "Y", "Z"), default=[2048, 2048, 2048],
        help="Crop size in voxels (x y z) at the chosen mip (DAPT budget = "
             "2048^3; clamped to the volume, so the thin y axis caps it).",
    )
    p.add_argument("--mip", type=int, default=0, help="Mip level (0 = ~4 nm native; doubles per level).")
    p.add_argument(
        "--em-path", default=None, dest="em_path",
        help=f"Override the precomputed EM sub-path (default {_EM_SUFFIX}).",
    )
    p.add_argument(
        "--resample-isotropic", action="store_true", dest="resample_isotropic",
        help="Resample the crop to an exact 4 nm CUBIC voxel before saving "
             "(COSEM z is 3.24-5.24 nm).  OFF by default: per "
             "doc/RESOLUTION_LADDER.md §2.1 the <=1.24x z difference is "
             "negligible and the train-time resample onto the 4 nm grid already "
             "handles it -- so we keep the native voxel and only bake the cubic "
             "4 nm here when you explicitly want isotropic files on disk.",
    )
    p.add_argument(
        "--iso-target", type=float, default=4.0, dest="iso_target",
        help="Target cubic voxel size (nm) for --resample-isotropic (default 4).",
    )
    p.add_argument(
        "--skip-existing", action="store_true", dest="skip_existing",
        help="If the output .h5 already exists and is a valid HDF5 (key 'main', "
             "non-empty), skip the CloudVolume fetch (integrity check on a prior "
             "download).  Lets a re-run resume without re-fetching.",
    )
    return p.parse_args()


def _is_valid_h5(path: Path) -> bool:
    """True iff ``path`` is a readable HDF5 with a non-empty ``main`` dataset."""
    if not path.exists():
        return False
    try:
        import h5py
        with h5py.File(str(path), "r", locking=False) as f:
            return "main" in f and f["main"].size > 0
    except Exception:  # noqa: BLE001 -- corrupt / partial download
        return False


def _resample_isotropic(img_zyx: np.ndarray, res_xyz, target_nm: float):
    """Resample a ``[Z, Y, X]`` crop to an exact ``target_nm`` cubic voxel.

    ``res_xyz`` is the native voxel size ``(x, y, z)`` nm.  Each axis is
    rescaled by ``native_spacing / target_nm`` (e.g. COSEM z 3.24 -> 4 nm is a
    0.81x downsample; xy 4 -> 4 nm is a no-op).  Trilinear in fp32, then cast
    back to the input dtype.  Returns ``(resampled_zyx, [target, target, target])``.
    """
    import torch
    import torch.nn.functional as F

    res_zyx = [float(res_xyz[2]), float(res_xyz[1]), float(res_xyz[0])]  # (z, y, x)
    Z, Y, X = img_zyx.shape
    new = [max(1, int(round(n * s / target_nm))) for n, s in zip((Z, Y, X), res_zyx)]
    if new == [Z, Y, X]:
        return img_zyx, [target_nm, target_nm, target_nm]
    t = torch.as_tensor(img_zyx, dtype=torch.float32)[None, None]
    t = F.interpolate(t, size=tuple(new), mode="trilinear", align_corners=False)
    out = t[0, 0]
    if np.issubdtype(img_zyx.dtype, np.integer):
        info = np.iinfo(img_zyx.dtype)
        out = out.round().clamp(info.min, info.max)
    out = out.numpy().astype(img_zyx.dtype)
    return np.ascontiguousarray(out), [target_nm, target_nm, target_nm]


def _open(src: str, mip: int):
    from cloudvolume import CloudVolume

    return CloudVolume(src, mip=mip, use_https=True, progress=True, fill_missing=True)


def _clamp_box(origin, size, vol):
    """Clamp an (origin, size) request to the volume bounds (x, y, z)."""
    lo = np.array(vol.bounds.minpt, dtype=np.int64)
    hi = np.array(vol.bounds.maxpt, dtype=np.int64)
    size = np.array(size, dtype=np.int64)
    if origin is None:
        centre = (lo + hi) // 2
        origin = centre - size // 2
    # Keep the WHOLE box inside [lo, hi]: clamp the origin to
    # [lo, max(lo, hi - size)] so an out-of-bounds origin slides in instead of
    # producing end < start (negative span -> OutOfBoundsError).
    origin = np.clip(np.array(origin, dtype=np.int64), lo, np.maximum(lo, hi - size))
    end = np.minimum(origin + size, hi)
    return origin, end


def _save_h5(arr_zyx: np.ndarray, path: Path, resolution_xyz, dataset: str) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "main", data=arr_zyx, compression="gzip", compression_opts=4, chunks=True,
        )
        # Provenance: physical voxel size in z, y, x (nm) next to the data.
        ds.attrs["resolution_zyx_nm"] = np.asarray(
            [resolution_xyz[2], resolution_xyz[1], resolution_xyz[0]], dtype=np.float64,
        )
        ds.attrs["source"] = (
            f"OpenOrganelle / COSEM {dataset} "
            f"(s3://janelia-cosem-datasets/{dataset}); image-only, DAPT branch."
        )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    name = args.name or args.dataset

    src = _em_src(args.dataset, args.em_path)
    vol = _open(src, args.mip)
    res = [float(r) for r in vol.resolution]  # x, y, z nm at this mip
    origin, end = _clamp_box(args.origin, args.size, vol)
    x0, y0, z0 = (int(v) for v in origin)
    x1, y1, z1 = (int(v) for v in end)
    coords = f"x{x0}_y{y0}_z{z0}"
    # Final res tag / stem are known before the fetch (resample only retags to
    # the cubic target), so the skip check can avoid the heavy download.
    res_tag = (f"{args.iso_target:g}nm" if args.resample_isotropic
               else "x".join(f"{r:g}" for r in res) + "nm")
    stem = f"{name}_{res_tag}_{coords}"
    out_path = out_dir / f"{stem}_volume.h5"

    print(f"Dataset: {args.dataset}  mip {args.mip}: resolution (x,y,z) = {res} nm")
    print(f"Crop box (x,y,z): [{x0}:{x1}, {y0}:{y1}, {z0}:{z1}]  "
          f"-> size {(x1 - x0, y1 - y0, z1 - z0)} voxels")
    if not (res[0] == res[1] and abs(res[0] - res[2]) <= 0.25 * res[0]):
        print(f"  NOTE: voxel is not near-cubic ({res}); the ladder "
              f"expects a ~4 nm small (near-cubic) voxel for the DAPT anchor.")

    if args.skip_existing and _is_valid_h5(out_path):
        print(f"Skip (already downloaded, valid HDF5): {out_path.name}")
        return

    # CloudVolume returns [X, Y, Z, C]; squeeze channel, move to [Z, Y, X].
    img = vol[x0:x1, y0:y1, z0:z1][..., 0]
    img_zyx = np.ascontiguousarray(np.transpose(img, (2, 1, 0)))

    # Optional: bake an exact cubic voxel (off by default -- see the flag help).
    if args.resample_isotropic:
        img_zyx, res = _resample_isotropic(img_zyx, res, args.iso_target)
        print(f"Resampled to exact {args.iso_target:g} nm cubic voxel -> "
              f"shape(z,y,x) = {img_zyx.shape}")

    _save_h5(img_zyx, out_path, res, args.dataset)
    print(f"Saved image: {out_path.name}  shape(z,y,x) = {img_zyx.shape}")

    print("\nConfig volume entry (paste under data.branches.dapt.volumes):")
    print(f"  - vol: {stem}_volume")
    print(f"    root: {out_dir}")
    print("    # image-only; DAPT (no seg)")


if __name__ == "__main__":
    main()
