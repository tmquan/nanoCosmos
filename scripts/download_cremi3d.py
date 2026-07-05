#!/usr/bin/env python
"""Download the CREMI training + test samples and convert them to nanocosmos HDF5.

CREMI (https://cremi.org) provides three ssTEM volumes of adult *Drosophila*
brain at **4 x 4 x 40 nm** (x, y, z):

* **A, B, C** -- labelled TRAINING volumes (125 x 1250 x 1250, dense neuron
  ids), used for ``train_volumes``.
* **A+, B+, C+** -- TEST volumes; public EM only (the challenge withholds the
  test neuron ids), converted **image-only** (no seg -- there is nothing to
  submit against locally; this is purely SSL / qualitative-inference data).

Each test sample ships in **two** official downloads (see
https://cremi.org/data/):

* **cropped** (~151 MB) -- ``125 x 1250 x 1250``, the same footprint as the
  labelled A/B/C training volumes.
* **padded** (~1.46 GB) -- ``200 x 3072 x 3072``, the SAME tissue with ~4x
  more surrounding context (more neighbouring voxels per axis; useful as
  extra free SSL signal, or as receptive-field context for inference).

Verified by exact full-volume array match (not from any header/attribute --
CREMI does not publish an offset for label-less test volumes): the cropped
volume is a byte-for-byte sub-array of the padded one, at the **same** native
voxel offset for every sample --

    padded[37 : 37+125, 911 : 911+1250, 911 : 911+1250] == cropped

i.e. ``offset (z, y, x) = (37, 911, 911)``, ``cropped shape = (125, 1250, 1250)``.
Pass ``--padded`` to fetch the padded version instead of the default cropped
one; the offset/shape are recorded as ``.h5`` attributes on the converted
padded volume for later crop-back (e.g. before a challenge submission -- see
``scripts/infer_cremi_submission.py``).

The official files pack raw + labels into a single nested ``.hdf``
(``volumes/raw`` + ``volumes/labels/neuron_ids``), which ``LazyVolDataset``
cannot read directly.

This script downloads the originals and converts each sample into the
nanocosmos convention -- two ``.h5`` files (dataset key ``main``, axis order
``[Z, Y, X]``) consumed by ``MICRONSDataset`` / ``CREMI3DDataset`` and the
lazy 3-D patch loader::

    cremi3d_sample_A_volume.h5         (uint8  EM intensity)
    cremi3d_sample_A_segmentation.h5   (int64  neuron ids, 0 = background)

Example
-------
    # download A, B, C (train) + A+, B+, C+ cropped (test) and convert
    python scripts/download_cremi3d.py --out-dir data/CREMI3D

    # the PADDED test volumes instead (more SSL context per sample)
    python scripts/download_cremi3d.py --out-dir data/CREMI3D \\
        --samples A+ B+ C+ --padded

    # reuse already-downloaded .hdf files (skip the network)
    python scripts/download_cremi3d.py --out-dir data/CREMI3D --hdf-dir /scratch/CREMI3D
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

CREMI_BASE_URL = "https://cremi.org/static/data"
# Labelled TRAINING volumes -- dense neuron ids.
TRAIN_FILES = {
    "A": "sample_A_20160501.hdf",
    "B": "sample_B_20160501.hdf",
    "C": "sample_C_20160501.hdf",
}
# TEST volumes (A+/B+/C+), cropped download.  Public EM only -- the challenge
# withholds the test neuron ids -- so these convert to image-only crops.
TEST_FILES = {
    "A+": "sample_A+_20160601.hdf",
    "B+": "sample_B+_20160601.hdf",
    "C+": "sample_C+_20160601.hdf",
}
# TEST volumes, PADDED download -- same tissue, ~4x more context per axis.
TEST_FILES_PADDED = {
    "A+": "sample_A+_padded_20160601.hdf",
    "B+": "sample_B+_padded_20160601.hdf",
    "C+": "sample_C+_padded_20160601.hdf",
}
SAMPLE_FILES = {**TRAIN_FILES, **TEST_FILES}
# CREMI marks unlabeled voxels (padded volumes) with the max uint64 value.
_NO_DATA = int(np.iinfo(np.uint64).max)
# Verified (full-volume exact array match, all three test samples) offset of
# the cropped test volume within its padded counterpart -- see module docstring.
PADDED_OFFSET_ZYX = (37, 911, 911)
CROPPED_SHAPE_ZYX = (125, 1250, 1250)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default="data/CREMI3D", help="Output directory for converted .h5.")
    p.add_argument(
        "--samples", nargs="+",
        default=["A", "B", "C", "A+", "B+", "C+"],
        choices=["A", "B", "C", "A+", "B+", "C+"],
        help="Which CREMI samples to fetch/convert.  A/B/C are labelled "
             "training volumes; A+/B+/C+ are the TEST volumes "
             "(public EM only, labels withheld -> image-only).",
    )
    p.add_argument(
        "--padded", action="store_true",
        help="For A+/B+/C+, fetch the PADDED download (200x3072x3072, ~1.46 GB "
             "each) instead of the default cropped one (125x1250x1250, ~151 MB). "
             "Ignored for A/B/C (training volumes have no padded variant).",
    )
    p.add_argument(
        "--hdf-dir", default=None,
        help="Directory of already-downloaded sample_*.hdf (skip the download step).",
    )
    return p.parse_args()


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip download {dest.name} (exists, {dest.stat().st_size / 1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  ok {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def _save_h5(arr: np.ndarray, path: Path, resolution_zyx_nm, extra_attrs: dict | None = None) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        ds = f.create_dataset(
            "main", data=arr, compression="gzip", compression_opts=4, chunks=True,
        )
        ds.attrs["resolution_zyx_nm"] = np.asarray(resolution_zyx_nm, dtype=np.float64)
        ds.attrs["source"] = "CREMI Challenge (https://cremi.org)"
        for k, v in (extra_attrs or {}).items():
            ds.attrs[k] = v


def _convert(hdf_path: Path, out_dir: Path, sample: str, *, padded: bool = False) -> None:
    import h5py

    res = (40.0, 4.0, 4.0)  # z, y, x nm
    with h5py.File(str(hdf_path), "r", locking=False) as f:
        raw = f["volumes/raw"][:]  # [Z, Y, X] uint8
        seg = (
            f["volumes/labels/neuron_ids"][:]
            if "volumes/labels/neuron_ids" in f else None
        )

    raw = np.ascontiguousarray(raw.astype(np.uint8))
    stem = f"cremi3d_sample_{sample}_padded" if padded else f"cremi3d_sample_{sample}"
    vol_path = out_dir / f"{stem}_volume.h5"
    extra_attrs = None
    if padded:
        extra_attrs = {
            "cropped_region_offset_zyx": np.asarray(PADDED_OFFSET_ZYX, dtype=np.int64),
            "cropped_region_shape_zyx": np.asarray(CROPPED_SHAPE_ZYX, dtype=np.int64),
            "source": (
                "CREMI Challenge (https://cremi.org) -- PADDED test volume; "
                "the official cropped download is a byte-exact sub-array at "
                "cropped_region_offset_zyx (verified by full-volume array match)."
            ),
        }
    _save_h5(raw, vol_path, res, extra_attrs)

    # Map CREMI's "no data" marker to background; keep ids otherwise.  Test
    # volumes (A+/B+/C+) have no public labels, so only write a segmentation
    # when real foreground ids are present.
    if seg is not None:
        seg = np.where(np.asarray(seg) == _NO_DATA, 0, seg).astype(np.int64)
    if seg is not None and bool((seg > 0).any()):
        seg_path = out_dir / f"{stem}_segmentation.h5"
        _save_h5(seg, seg_path, res)
        n_ids = int(np.unique(seg).size)
        fg = float((seg > 0).mean()) * 100.0
        print(
            f"  converted sample {sample}{' (padded)' if padded else ''}: image {raw.shape} uint8, "
            f"seg {seg.shape} int64, {n_ids} ids, foreground {fg:.1f}%"
        )
    else:
        print(
            f"  converted sample {sample}{' (padded)' if padded else ''}: image {raw.shape} uint8, "
            f"NO public labels (test/withheld) -> image-only"
        )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    hdf_dir = Path(args.hdf_dir) if args.hdf_dir else out_dir

    print(f"CREMI3D -> {out_dir}  (samples: {', '.join(args.samples)}"
          f"{', padded test volumes' if args.padded else ''})")
    print("Resolution: 4 x 4 x 40 nm (x, y, z) -> resolution_map key 'cremi3d': [40, 4, 4]\n")

    for s in args.samples:
        use_padded = args.padded and s in TEST_FILES_PADDED
        fname = TEST_FILES_PADDED[s] if use_padded else SAMPLE_FILES[s]
        # cremi.org URL-encodes the literal '+' in the sample name.
        url_name = fname.replace("+", "%2B")
        hdf_path = hdf_dir / fname
        if args.hdf_dir is None:
            _download(f"{CREMI_BASE_URL}/{url_name}", hdf_path)
        if not hdf_path.exists():
            print(f"  MISSING {hdf_path} -- skipping sample {s}")
            continue
        _convert(hdf_path, out_dir, s, padded=use_padded)

    train = [s for s in args.samples if s in TRAIN_FILES]
    test = [s for s in args.samples if s in TEST_FILES]
    if train:
        print("\nConfig train_volumes (labelled A/B/C):")
        for s in train:
            print(f"  - vol: cremi3d_sample_{s}_volume")
            print(f"    seg: cremi3d_sample_{s}_segmentation")
            print(f"    root: {out_dir}")
    if test:
        suffix = "_padded" if args.padded else ""
        label = "padded" if args.padded else "cropped"
        print(f"\nConfig volumes (A+/B+/C+, {label}, image-only -- labels withheld):")
        for s in test:
            print(f"  - vol: cremi3d_sample_{s}{suffix}_volume")
            print(f"    root: {out_dir}")


if __name__ == "__main__":
    main()
