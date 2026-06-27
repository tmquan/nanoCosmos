#!/usr/bin/env python
"""Download a FLYEM3D crop (FlyEM 8 nm FIB-SEM) as MICrONS-style HDF5.

``FLYEM3D`` is the umbrella source for the Janelia FlyEM 8 nm FIB-SEM
connectomes (``--dataset``):

    fib25      FlyEM 7-column Drosophila medulla (Takemura 2015)
               full 6446 x 6643 x 8090 vox; image + ground_truth (precomputed)
    hemibrain  Drosophila central brain (~25 k neurons)
               full 34427 x 39725 x 41394 vox; CLAHE-jpeg image + v1.2 seg
    malecns    full male Drosophila CNS (v1.0, 2026)
               full 94088 x 78317 x 134576 vox; CLAHE-jpeg image + v1.0 seg

All are **8 x 8 x 8 nm** at mip 0 (doubles per mip).  We crop via CloudVolume
random access and write image (+ segmentation) as ``.h5`` files that
``nanocosmos.datasets.FLYEM3DDataset`` (== MICRONSDataset layout: key ``main``,
axis order ``[Z, Y, X]``) loads directly.

``--role`` picks the ladder role (doc/RESOLUTION_LADDER.md): ``sft`` = image +
segmentation (labeled rung); ``dapt`` = image only (label-free reconstruction;
e.g. FIB-25's unsegmented surround, or any Hemibrain / MaleCNS crop).

Examples
--------
    # SFT crop of FIB-25 (image + proofread segmentation):
    python scripts/download_flyem3d.py --dataset fib25 --role sft \
        --origin 2304 2048 6144 --size 512 512 512

    # DAPT (image-only) crop of Hemibrain neuropil:
    python scripts/download_flyem3d.py --dataset hemibrain --role dapt \
        --size 512 512 512

    # FIB-25 only: fetch the native cube once, then make local variants
    # (orientations / z-stride) without re-downloading (see --from-local):
    python scripts/download_flyem3d.py --dataset fib25 --from-local \
        data/FLYEM3D/flyem3d_fib25_8nm_x2304_y2048_z6144_volume.h5 --z-stride 4

(``--origin`` / ``--size`` are voxel units, x y z, at the chosen mip.  The
``--orientations`` / ``--z-stride`` / ``--from-local`` machinery is FIB-25-only
legacy augmentation; hemibrain / malecns use a plain crop with the defaults.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Verified Neuroglancer ``precomputed`` sources per FlyEM dataset (8 nm iso).
SOURCES = {
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="fib25", choices=sorted(SOURCES),
                   help="Which FlyEM 8 nm volume to crop.")
    p.add_argument("--out-dir", default="data/FLYEM3D", help="Output directory for the .h5 crops.")
    p.add_argument("--name", default=None,
                   help="Crop basename prefix (default: flyem3d_<dataset>).")
    p.add_argument(
        "--origin", type=int, nargs=3, metavar=("X", "Y", "Z"), default=None,
        help="Crop origin in voxels (x y z) at the chosen mip. Default: centred.",
    )
    p.add_argument(
        "--size", type=int, nargs=3, metavar=("X", "Y", "Z"), default=[512, 512, 256],
        help="Crop size in voxels (x y z) at the chosen mip.",
    )
    p.add_argument("--mip", type=int, default=0, help="Mip level (0 = 8 nm; doubles per level).")
    p.add_argument(
        "--from-local", default=None, dest="from_local",
        help="Path to an already-downloaded native cube ``*_volume.h5`` (and its "
             "sibling ``*_segmentation.h5``).  When set, the script skips the "
             "CloudVolume download and generates all variants (orientations / "
             "z-stride / phases) from this local cube -- so one fetch covers "
             "every variant.  Because the native FIB cube is isotropic, "
             "transpose-then-stride is valid along any axis.",
    )
    p.add_argument(
        "--z-stride", type=int, default=1, dest="z_stride",
        help="Keep every N-th z-section after cropping, making the crop "
             "anisotropic (effective z res = mip_res * N).  Use 4 at mip 0 to "
             "turn the native 8 nm isotropic data into 32 x 8 x 8 nm (z,y,x), "
             "matching the flyem3d_z32 resolution_map key -- 32 nm sits just "
             "above the union z floor so the jitter mostly downsamples it "
             "(honest) rather than upsampling discarded sections.",
    )
    p.add_argument(
        "--orientations", nargs="+", default=["z"], choices=["z", "y", "x"],
        help="Emit isotropic transposed copies with the given axis in the thin "
             "(section/z) position: 'z' = native, 'y' = y<->z (suffix _yz), "
             "'x' = x<->z (suffix _xz).  Exploits FIB-25's isotropy for "
             "orientation augmentation -- the shape-changing transposes the "
             "fixed-shape batch pipeline cannot do at runtime.  Requires "
             "--z-stride 1 (isotropic).",
    )
    p.add_argument(
        "--z-phases", nargs="+", default=["all"],
        help="For anisotropic crops (--z-stride > 1), which z-section phase "
             "offsets to emit as separate variants.  'all' (default) emits all "
             "N = z-stride phases -- each keeps a different subset of sections "
             "(a distinct thick-section sampling of the same tissue), reusing "
             "the data a single-phase stride would discard (free anisotropic "
             "augmentation).  Or pass a list of offsets (e.g. 0 2).  Ignored "
             "when --z-stride 1.",
    )
    p.add_argument(
        "--role", default="sft", choices=("sft", "dapt"),
        help="Ladder role (doc/RESOLUTION_LADDER.md).  'sft' = image + "
             "segmentation, for crops inside the proofread core (labeled "
             "rung).  'dapt' = image only (implies --no-seg) + a `_dapt` name "
             "tag, for the large UNSEGMENTED surround used as the label-free "
             "self-supervised reconstruction source.",
    )
    p.add_argument(
        "--no-seg", action="store_true",
        help="Download the image only (skip the ground-truth segmentation; "
             "also implied by --role dapt).",
    )
    p.add_argument(
        "--skip-existing", action="store_true", dest="skip_existing",
        help="If the output .h5 already exists and is a valid HDF5 (key 'main', "
             "non-empty), skip it -- the integrity check on a prior download. "
             "For the simple (z-stride 1, default orientation) case this skips "
             "the CloudVolume fetch entirely; for variant runs it skips per file.",
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


def _open(src: str, mip: int):
    from cloudvolume import CloudVolume

    return CloudVolume(src, mip=mip, use_https=True, progress=True, fill_missing=True)


def _clamp_box(origin, size, vol):
    """Clamp an (origin, size) request to the volume bounds (x,y,z)."""
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


def _save_h5(arr_zyx: np.ndarray, path: Path, resolution_xyz) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset("main", data=arr_zyx, compression="gzip", compression_opts=4, chunks=True)
        # Provenance: store the physical voxel size in z,y,x (nm) next to the data.
        ds.attrs["resolution_zyx_nm"] = np.asarray(
            [resolution_xyz[2], resolution_xyz[1], resolution_xyz[0]], dtype=np.float64
        )
        ds.attrs["source"] = "FlyEM FIB-25 (Takemura et al. 2015), gs://neuroglancer-public-data/flyem_fib-25"


def _load_local(vol_path: str, no_seg: bool):
    """Load a native cube + sibling segmentation from disk for re-processing.

    Returns ``(img_zyx, seg_zyx | None, res_xyz, coords)`` -- the full (un-
    strided) ``[Z, Y, X]`` arrays, the native voxel size in ``(x, y, z)`` nm
    (read from the ``resolution_zyx_nm`` attr; default 8 nm isotropic), and the
    ``x..._y..._z...`` coordinate token parsed from the filename.
    """
    import re
    import h5py

    p = Path(vol_path)
    with h5py.File(str(p), "r", locking=False) as f:
        img = np.asarray(f["main"][:])
        attr = f["main"].attrs.get("resolution_zyx_nm")
    res_zyx = [int(round(float(v))) for v in attr] if attr is not None else [8, 8, 8]
    res_xyz = [res_zyx[2], res_zyx[1], res_zyx[0]]

    seg = None
    if not no_seg:
        seg_path = Path(str(p).replace("_volume.h5", "_segmentation.h5"))
        if seg_path.exists():
            with h5py.File(str(seg_path), "r", locking=False) as f:
                seg = np.asarray(f["main"][:])
        else:
            print(f"  note: no sibling segmentation at {seg_path.name} -- image only")

    m = re.search(r"x\d+_y\d+_z\d+", p.name)
    coords = m.group(0) if m else "local"
    return img, seg, res_xyz, coords


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    # FIB-25 is the canonical FLYEM3D volume -> plain ``flyem3d`` stem (keeps the
    # legacy resolution_map keys ``flyem3d`` / ``flyem3d_z32`` matching); the
    # others are sub-named so they don't collide.
    name = args.name or ("flyem3d" if args.dataset == "fib25" else f"flyem3d_{args.dataset}")
    image_src = SOURCES[args.dataset]["image"]
    seg_src = SOURCES[args.dataset]["seg"]

    # DAPT crops are image-only (e.g. FIB-25's unsegmented surround); SFT crops
    # carry the proofread segmentation.  ``--role dapt`` implies ``--no-seg``.
    no_seg = bool(args.no_seg) or args.role == "dapt"
    role_tag = "_dapt" if args.role == "dapt" else ""

    # --- acquire the FULL [Z, Y, X] crop (no stride) from one source ---
    if args.from_local:
        img_full, seg_full, res, coords = _load_local(args.from_local, no_seg)
        print(f"Local source: {args.from_local}  resolution (x,y,z) = {res} nm  "
              f"shape(z,y,x) = {img_full.shape}")
    else:
        img_vol = _open(image_src, args.mip)
        res = [int(r) for r in img_vol.resolution]  # x,y,z nm at this mip
        origin, end = _clamp_box(args.origin, args.size, img_vol)
        x0, y0, z0 = (int(v) for v in origin)
        x1, y1, z1 = (int(v) for v in end)
        coords = f"x{x0}_y{y0}_z{z0}"
        print(f"Mip {args.mip}: resolution (x,y,z) = {res} nm")
        print(f"Crop box (x,y,z): [{x0}:{x1}, {y0}:{y1}, {z0}:{z1}]  "
              f"-> size {(x1 - x0, y1 - y0, z1 - z0)} voxels")
        # Pre-fetch skip for the simple (single-output) case: a plain z-stride-1
        # crop with the default orientation emits exactly one file, whose stem
        # is known here -- if it's already a valid HDF5, skip the heavy fetch.
        _simple = max(1, int(args.z_stride)) == 1 and list(dict.fromkeys(args.orientations)) == ["z"]
        if args.skip_existing and _simple:
            stem0 = f"{name}_{int(res[0])}nm{role_tag}_{coords}"
            if _is_valid_h5(out_dir / f"{stem0}_volume.h5") and (
                no_seg or _is_valid_h5(out_dir / f"{stem0}_segmentation.h5")
            ):
                print(f"Skip (already downloaded, valid HDF5): {stem0}_*.h5")
                return
        # CloudVolume returns [X, Y, Z, C]; squeeze channel and move to [Z, Y, X].
        img = img_vol[x0:x1, y0:y1, z0:z1][..., 0]
        img_full = np.ascontiguousarray(np.transpose(img, (2, 1, 0)))
        seg_full = None
        if not no_seg:
            seg_vol = _open(seg_src, args.mip)
            seg = seg_vol[x0:x1, y0:y1, z0:z1][..., 0]
            seg_full = np.ascontiguousarray(np.transpose(seg, (2, 1, 0)))

    z_stride = max(1, int(args.z_stride))
    orientations = list(dict.fromkeys(args.orientations))  # dedup, keep order
    # Non-z orientations (xz/yz transpose) are only geometrically valid on an
    # isotropic source -- the emit loop transposes the cube THEN strides the new
    # z, so any section axis works as long as all native axes are equal.
    isotropic = res[0] == res[1] == res[2]
    if not isotropic and any(o != "z" for o in orientations):
        raise SystemExit(
            f"non-z orientations (xz/yz) require an isotropic source; got "
            f"resolution (x,y,z) = {res}.  Use an isotropic native cube."
        )

    # z-section phase offsets (only meaningful when striding): each phase keeps
    # a different subset of sections, so emitting all N reuses the data a
    # single-phase stride would discard -- free anisotropic augmentation.
    if z_stride == 1:
        phases = [0]
    elif "all" in args.z_phases:
        phases = list(range(z_stride))
    else:
        phases = sorted({int(p) for p in args.z_phases})
        for p in phases:
            if not (0 <= p < z_stride):
                raise SystemExit(f"--z-phases offset {p} out of [0,{z_stride}).")
    multi_phase = len(phases) > 1

    eff_res = [res[0], res[1], res[2] * z_stride]
    base_res = f"{res[0]}nm" if z_stride == 1 else f"z{eff_res[2]}xy{res[0]}nm"
    print(f"z-stride {z_stride} -> effective resolution (x,y,z) = {eff_res} nm")
    print(f"orientations: {orientations} | z-phases: {phases}")

    # Emit each orientation x phase by permuting the section axis into z, then
    # sub-sampling z at the phase offset (the cube was acquired full-z above).
    # Axis permutation (on [Z, Y, X]) + filename suffix per orientation.
    _ORIENT = {"z": ((0, 1, 2), ""), "y": ((1, 0, 2), "_yz"), "x": ((2, 1, 0), "_xz")}

    print("\nConfig volume entries (paste under data.train_volumes / val_volumes):")
    for o in orientations:
        perm, tag = _ORIENT[o]
        img_o = np.transpose(img_full, perm)
        seg_o = np.transpose(seg_full, perm) if seg_full is not None else None
        for p in phases:
            suffix = f"{tag}_p{p}" if multi_phase else tag
            stem = f"{name}_{base_res}{role_tag}{suffix}_{coords}"
            # Per-file skip (covers the multi-variant / from-local case).
            if args.skip_existing and _is_valid_h5(out_dir / f"{stem}_volume.h5") and (
                seg_o is None or _is_valid_h5(out_dir / f"{stem}_segmentation.h5")
            ):
                print(f"Skip (valid HDF5 exists): {stem}_*.h5")
                continue
            img_p = np.ascontiguousarray(img_o[p::z_stride])
            _save_h5(img_p, out_dir / f"{stem}_volume.h5", eff_res)
            print(f"Saved image:        {stem}_volume.h5  shape(z,y,x)={img_p.shape}")
            if seg_o is not None:
                seg_p = np.ascontiguousarray(seg_o[p::z_stride])
                _save_h5(seg_p, out_dir / f"{stem}_segmentation.h5", eff_res)
                n_ids = int(np.unique(seg_p).size)
                fg = float((seg_p > 0).mean()) * 100.0
                print(f"Saved segmentation: {stem}_segmentation.h5  {n_ids} ids, fg {fg:.1f}%")
                if fg < 5.0:
                    print("  WARNING: very little foreground -- region may be outside "
                          "the proofread cube; try a different --origin.")
            print(f"  - vol: {stem}_volume")
            if seg_o is not None:
                print(f"    seg: {stem}_segmentation")
            print(f"    root: {out_dir}")


if __name__ == "__main__":
    main()
