#!/usr/bin/env python
"""Convert MitoEM2 (nnU-Net ``.nii.gz``) volumes to nanoCosmos ``.h5``.

MitoEM2 ships as nnU-Net datasets under ``data/MitoEM2``::

    data/MitoEM2/
      Dataset00N_ME2-<subset>.zip          # raw download (optional)
      Dataset00N_ME2-<subset>/
        imagesTr/me2-<subset>_train{NN}_0000.nii.gz
        imagesTs/me2-<subset>_test{NN}_0000.nii.gz
        labelsTr/...                        # mito/boundary -- UNUSED here

The joint recipe consumes flat ``*_volume.h5`` files (dataset key ``main``,
axis order ``[Z, Y, X]``), so this script converts each ``imagesTr`` /
``imagesTs`` image to::

    data/MitoEM2/mitoem2_<subset>_<crop>_volume.h5

It is **image-only** (the ``ssl`` reconstruction branch): labels are ignored.
The folder split maps to the joint recipe as ``imagesTr`` -> ssl **train** and
``imagesTs`` -> ssl **validation** holdout (``task: ssl``).

``native_resolution`` is read from each volume's voxel spacing and reversed to
``[z, y, x]`` (e.g. nnU-Net zooms ``(8, 8, 30)`` -> ``[30, 8, 8]``).  nibabel's
array axes ``(X, Y, Z)`` are transposed to ``(Z, Y, X)`` to match the loader.

Robustness: an existing output is **validated** (openable, has ``main``, 3-D)
and only skipped if valid -- a truncated file from an interrupted run is
overwritten rather than silently kept (the bug that left a corrupt
``mossy_test02`` once).

Example
-------
    # convert everything (extract any *.zip first), print config YAML:
    python scripts/convert_mitoem2.py --data-dir data/MitoEM2 --extract-zips --print-yaml
    # only the train split, force re-convert:
    python scripts/convert_mitoem2.py --splits tr --overwrite
"""

from __future__ import annotations

import argparse
import glob
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

_SPLIT_DIRS = {"tr": "imagesTr", "ts": "imagesTs"}


def _is_valid_h5(path: Path) -> bool:
    """True iff ``path`` is an openable HDF5 with a 3-D ``main`` dataset."""
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return "main" in f and f["main"].ndim == 3
    except Exception:
        return False


def _out_name(nii_path: Path) -> str:
    """``me2-<subset>_<crop>_0000.nii.gz`` -> ``mitoem2_<subset>_<crop>_volume``."""
    stem = nii_path.name[: -len(".nii.gz")]
    if stem.endswith("_0000"):
        stem = stem[: -len("_0000")]
    return f"mitoem2_{stem.replace('me2-', '', 1)}_volume"


def _convert_one(nii_path: Path, out: Path, *, overwrite: bool) -> tuple[list[int], str]:
    """Convert one ``.nii.gz`` to ``.h5``; return ``(native_res, status)``."""
    img = nib.load(str(nii_path))
    sx, sy, sz = (round(float(z)) for z in img.header.get_zooms()[:3])
    res = [sz, sy, sx]  # (z, y, x)

    if not overwrite and _is_valid_h5(out):
        return res, "skip(valid)"
    if out.exists():
        out.unlink()  # truncated/corrupt or --overwrite: drop and rewrite

    arr = np.ascontiguousarray(
        np.asarray(img.dataobj, dtype=np.uint8).transpose(2, 1, 0)  # (X,Y,Z)->(Z,Y,X)
    )
    Z, Y, X = arr.shape
    tmp = out.with_suffix(".h5.tmp")  # write to tmp then rename -> never leave a partial
    with h5py.File(tmp, "w") as hf:
        hf.create_dataset(
            "main", data=arr,
            chunks=(min(64, Z), min(256, Y), min(256, X)),
            compression="gzip", compression_opts=4,
        )
    tmp.rename(out)
    return res, "wrote"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/MitoEM2",
                    help="MitoEM2 root containing Dataset0NN_ME2-* folders.")
    ap.add_argument("--splits", default="both", choices=("tr", "ts", "both"),
                    help="Which nnU-Net split(s) to convert (tr=train, ts=test).")
    ap.add_argument("--extract-zips", action="store_true",
                    help="Extract any Dataset*.zip whose folder is missing first.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-convert even when a valid output already exists.")
    ap.add_argument("--print-yaml", action="store_true",
                    help="Print config volume entries (grouped by split + resolution).")
    args = ap.parse_args()

    root = Path(args.data_dir)
    if not root.is_dir():
        raise SystemExit(f"data dir not found: {root}")

    if args.extract_zips:
        for z in sorted(root.glob("Dataset*.zip")):
            if (root / z.stem).exists():
                continue
            print(f"extracting {z.name} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(root)

    splits = ("tr", "ts") if args.splits == "both" else (args.splits,)
    # split -> resolution -> [vol_name, ...]
    summary: dict = {s: defaultdict(list) for s in splits}
    n_wrote = n_skip = 0
    for d in sorted(root.glob("Dataset*/")):
        if not d.is_dir():
            continue
        for split in splits:
            for f in sorted(glob.glob(str(d / _SPLIT_DIRS[split] / "*.nii.gz"))):
                nii = Path(f)
                vol = _out_name(nii)
                res, status = _convert_one(nii, root / f"{vol}.h5", overwrite=args.overwrite)
                summary[split][tuple(res)].append(vol)
                if status == "wrote":
                    n_wrote += 1
                    print(f"  [{split}] wrote {vol}.h5  res={res}")
                else:
                    n_skip += 1
    print(f"\ndone: wrote={n_wrote} skipped(valid)={n_skip}")

    if args.print_yaml:
        for split in splits:
            where = "data.branches.ssl.volumes" if split == "tr" else "data.val_volumes (task: ssl)"
            print(f"\n# ==== MitoEM2 {_SPLIT_DIRS[split]} -> {where} ====")
            indent = "      " if split == "tr" else "  "
            for res in sorted(summary[split]):
                print(f"{indent}# --- MitoEM2 {list(res)} ---")
                for vol in sorted(summary[split][res]):
                    print(f"{indent}- vol: {vol}")
                    print(f"{indent}  root: {root.as_posix()}")
                    if split == "ts":
                        print(f"{indent}  task: ssl")
                    print(f"{indent}  native_resolution: {list(res)}")


if __name__ == "__main__":
    main()
