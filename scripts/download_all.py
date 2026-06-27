#!/usr/bin/env python
"""Download (and verify) every dataset for the joint nanoCosmos recipe.

Orchestrates the per-dataset downloaders into the resolution-ladder layout
(doc/RESOLUTION_LADDER.md) and then verifies each produced ``.h5`` (key
``main``, ``[Z,Y,X]`` shape, ``resolution_zyx_nm`` provenance, dtype, and
segmentation foreground %).

Groups (``--datasets``):
    cosem    COSEM3D 4 nm small-voxel (DAPT)         -> data/COSEM3D
    flyem    FLYEM3D 8 nm: fib25 (SFT core + DAPT
             surround), hemibrain / malecns (DAPT)   -> data/FLYEM3D
    cremi    CREMI3D ssTEM (SFT)                      -> data/CREMI3D
    snemi    SNEMI3D / Neurons (SFT)                  -> data/SNEMI3D
    microns  MICrONS (SFT)                            -> data/MICRONS

Examples
--------
    python scripts/download_all.py --verify-only           # just check disk
    python scripts/download_all.py --dry-run               # print the plan
    python scripts/download_all.py --datasets cosem flyem  # fetch a subset
    python scripts/download_all.py                         # fetch + verify all

The download jobs are a *representative* manifest (crop counts / origins match
the census USED_SIZE); edit ``_jobs`` or run the individual scripts for finer
control.  Verification is config-driven: it scans the roots referenced by
``configs/nanocosmos_joint.yaml``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_PY = sys.executable
_CONFIG = _HERE.parent / "configs" / "nanocosmos_joint.yaml"

# COSEM crop origins (x y z); y kept at 0 because the volumes are thin in y.
_COSEM_ORIGINS = [(0, 0, 0), (4096, 0, 4096), (8192, 0, 0), (0, 0, 8192), (4096, 0, 0)]
_COSEM_VOLS = ["jrc_hela-3", "jrc_macrophage-2", "jrc_jurkat-1"]
# Hemibrain / MaleCNS DAPT crop origins (x y z); 4 train + 1 test.  Kept within
# Hemibrain's bounds (~34427 x 39725 x 41394) for a 1024^3 crop; the per-volume
# clamp in _clamp_box slides any out-of-bounds origin inside regardless.
_FLYEM_ORIGINS = [(4000, 4000, 4000), (12000, 12000, 12000), (20000, 20000, 20000),
                  (28000, 24000, 30000), (16000, 30000, 36000)]


def _cos(ds: str, origin: Tuple[int, int, int]) -> List[str]:
    return [_PY, str(_HERE / "download_cosem3d.py"), "--dataset", ds,
            "--origin", *map(str, origin), "--size", "2048", "2048", "2048"]


def _flyem(ds: str, role: str, origin: Tuple[int, int, int], size: int) -> List[str]:
    return [_PY, str(_HERE / "download_flyem3d.py"), "--dataset", ds, "--role", role,
            "--origin", *map(str, origin), "--size", str(size), str(size), str(size)]


def _jobs(datasets: List[str]) -> List[Tuple[str, List[str]]]:
    jobs: List[Tuple[str, List[str]]] = []

    if "cosem" in datasets:
        for ds in _COSEM_VOLS:
            for o in _COSEM_ORIGINS:                       # 5x 2048^3 (y clamps)
                jobs.append((f"cosem {ds} {o}", _cos(ds, o)))

    if "flyem" in datasets:
        # FIB-25: SFT proofread core (image+seg) + DAPT unsegmented surround.
        jobs.append(("flyem fib25 sft core",
                     _flyem("fib25", "sft", (2304, 2048, 6144), 512)))
        for o in [(1024, 1024, 1024), (4096, 4096, 1024), (1024, 4096, 4096), (4096, 1024, 4096)]:
            jobs.append((f"flyem fib25 dapt {o}", _flyem("fib25", "dapt", o, 1024)))
        for ds in ("hemibrain", "malecns"):                # 5x 1024^3 (4 train + 1 test)
            for o in _FLYEM_ORIGINS:
                jobs.append((f"flyem {ds} dapt {o}", _flyem(ds, "dapt", o, 1024)))

    if "cremi" in datasets:
        jobs.append(("cremi A/B/C",
                     [_PY, str(_HERE / "download_cremi3d.py"), "--out-dir", "data/CREMI3D",
                      "--samples", "A", "B", "C"]))

    if "snemi" in datasets:
        jobs.append(("snemi3d",
                     [_PY, str(_HERE / "download_snemi3d.py"), "--output", "data/SNEMI3D"]))

    if "microns" in datasets:
        jobs.append(("microns",
                     [_PY, str(_HERE / "download_microns.py"), "--output", "data/MICRONS",
                      "--size", "4096", "4096", "800", "--seg-version", "1300"]))
    return jobs


def _run_jobs(jobs: List[Tuple[str, List[str]]], dry_run: bool) -> Dict[str, int]:
    results: Dict[str, int] = {}
    for label, argv in jobs:
        print(f"\n=== {label} ===\n  $ {' '.join(argv)}")
        if dry_run:
            results[label] = 0
            continue
        rc = subprocess.run(argv).returncode
        results[label] = rc
        if rc != 0:
            print(f"  [WARN] '{label}' exited {rc} (continuing)")
    return results


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------

def _roots_from_config() -> List[str]:
    """All data roots referenced by the joint config (branches + val)."""
    roots = set()
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(_CONFIG))
        for bcfg in (cfg.data.get("branches") or {}).values():
            for v in bcfg.get("volumes", []):
                roots.add(str(v.get("root", "data")))
        for v in (cfg.data.get("val_volumes") or []):
            roots.add(str(v.get("root", "data")))
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read {_CONFIG}: {exc}; using defaults)")
    roots |= {"data/COSEM3D", "data/FLYEM3D", "data/CREMI3D", "data/SNEMI3D", "data/MICRONS"}
    return sorted(roots)


def _verify_h5(path: Path) -> Tuple[bool, str]:
    import h5py
    import numpy as np
    try:
        with h5py.File(str(path), "r", locking=False) as f:
            if "main" not in f:
                return False, f"no 'main' key (keys={list(f.keys())})"
            ds = f["main"]
            res = ds.attrs.get("resolution_zyx_nm")
            res_s = (f"res(zyx)={[round(float(r), 2) for r in res]}nm"
                     if res is not None else "res=?")
            info = f"shape{tuple(ds.shape)} {ds.dtype} {res_s}"
            if "_segmentation" in path.name:
                # Subsample (strided) so verification stays cheap on huge
                # volumes -- reading a full 4096^2x800 seg would be ~100 GB.
                sl = tuple(slice(None, None, max(1, s // 96)) for s in ds.shape)
                arr = np.asarray(ds[sl])
                info += (f" ids~{int(np.unique(arr).size)} "
                         f"fg~{float((arr > 0).mean()) * 100:.1f}% (sampled)")
            return True, info
    except Exception as exc:  # noqa: BLE001
        return False, f"ERROR {exc}"


def _verify(roots: List[str]) -> bool:
    print("\n" + "=" * 70 + "\nVERIFY\n" + "=" * 70)
    all_ok = True
    total = 0
    for root in roots:
        rp = Path(root)
        vols = sorted(rp.glob("*_volume.h5")) if rp.is_dir() else []
        segs = sorted(rp.glob("*_segmentation.h5")) if rp.is_dir() else []
        print(f"\n[{root}]  {'(missing dir)' if not rp.is_dir() else f'{len(vols)} volume(s), {len(segs)} seg(s)'}")
        for p in vols + segs:
            ok, info = _verify_h5(p)
            total += 1
            all_ok = all_ok and ok
            print(f"  {'OK ' if ok else 'BAD'}  {p.name}: {info}")
    if total == 0:
        print("\n  No .h5 files found -- run without --verify-only to download.")
        return False
    print(f"\n{'ALL OK' if all_ok else 'SOME FILES BAD/MISSING'} ({total} file(s) checked).")
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    all_groups = ["cosem", "flyem", "cremi", "snemi", "microns"]
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=all_groups + ["all"], help="Which groups to download.")
    p.add_argument("--verify-only", action="store_true", help="Skip download; just verify disk.")
    p.add_argument("--dry-run", action="store_true", help="Print the download plan, do not run.")
    args = p.parse_args()

    datasets = all_groups if "all" in args.datasets else args.datasets

    if not args.verify_only:
        jobs = _jobs(datasets)
        print(f"Planned {len(jobs)} download job(s) for: {', '.join(datasets)}")
        results = _run_jobs(jobs, args.dry_run)
        if not args.dry_run:
            n_fail = sum(1 for rc in results.values() if rc != 0)
            print(f"\nDownload: {len(results) - n_fail}/{len(results)} jobs OK"
                  + (f", {n_fail} failed (see warnings above)." if n_fail else "."))

    if not args.dry_run:
        ok = _verify(_roots_from_config())
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
