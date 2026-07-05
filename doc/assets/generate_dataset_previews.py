#!/usr/bin/env python
"""Generate representative preview PNGs for every row in doc/data.csv.

SSL rows  -> central z-slice of the EM volume.
SFT rows  -> EM | instance-label RGB (side by side).

Outputs land in doc/assets/datasets/<slug>.png and the CSV gains a Preview
column (relative to the repo root).

Run from the nanoCosmos repo root:
    python doc/assets/generate_dataset_previews.py
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_CSV = _REPO / "doc" / "data.csv"
_OUT = _REPO / "doc" / "assets" / "datasets"


def _slug(*parts: str) -> str:
    raw = "_".join(parts).lower()
    raw = re.sub(r"[^a-z0-9._-]+", "_", raw)
    return re.sub(r"_+", "_", raw).strip("_")


def _label_rgb(slice2d: np.ndarray) -> np.ndarray:
    """Deterministic pastel RGB for instance ids (background = black)."""
    labels = slice2d.astype(np.int64)
    out = np.zeros((*labels.shape, 3), dtype=np.float32)
    mask = labels > 0
    if not mask.any():
        return out
    ids = np.unique(labels[mask])
    for lid in ids:
        h = int(hashlib.md5(str(int(lid)).encode()).hexdigest()[:8], 16) % 360
        rgb = matplotlib.colors.hsv_to_rgb((h / 360.0, 0.55, 0.95))
        out[labels == lid] = rgb
    return out


def _read_slice(vol_path: Path, z_frac: float = 0.5) -> np.ndarray:
    with h5py.File(vol_path, "r", locking=False) as f:
        ds = f["main"]
        z = int(round((ds.shape[0] - 1) * z_frac))
        return np.asarray(ds[z])


def _norm_em(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, (1, 99))
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _save_ssl(em: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(em, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_sft(em: np.ndarray, labels: np.ndarray, title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(em, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("EM")
    axes[0].axis("off")
    axes[1].imshow(_label_rgb(labels))
    axes[1].set_title("Segmentation")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# (Dataset, Subset, Task, Split) -> (root, vol_stem, seg_stem|None)
# Stems omit the _volume / _segmentation / _labels suffix.
REPRESENTATIVE: Dict[Tuple[str, str, str, str], Tuple[str, str, Optional[str]]] = {
    ("COSEM3D", "jrc_hela-3", "SSL", "Train"): ("data/COSEM3D", "jrc_hela-3_4x4x3.24nm_x4096_y0_z2048", None),
    ("COSEM3D", "jrc_hela-3", "SSL", "Test"): ("data/COSEM3D", "jrc_hela-3_4x4x3.24nm_x6144_y0_z0", None),
    ("COSEM3D", "jrc_macrophage-2", "SSL", "Train"): ("data/COSEM3D", "jrc_macrophage-2_4x4x3.36nm_x4096_y0_z4096", None),
    ("COSEM3D", "jrc_macrophage-2", "SSL", "Test"): ("data/COSEM3D", "jrc_macrophage-2_4x4x3.36nm_x2048_y0_z4096", None),
    ("COSEM3D", "jrc_jurkat-1", "SSL", "Train"): ("data/COSEM3D", "jrc_jurkat-1_4x4x3.44nm_x4096_y0_z4096", None),
    ("COSEM3D", "jrc_jurkat-1", "SSL", "Test"): ("data/COSEM3D", "jrc_jurkat-1_4x4x3.44nm_x0_y0_z2048", None),
    ("CREMI3D", "A / B / C", "SFT", "Train"): ("data/CREMI3D", "cremi3d_sample_A", "cremi3d_sample_A"),
    ("CREMI3D", "A+ / B+ / C+ (padded)", "SSL", "Train"): ("data/CREMI3D", "cremi3d_sample_A+_padded", None),
    ("FLYEM3D", "FIB-25 (labeled core)", "SFT", "Train"): ("data/FLYEM3D", "flyem3d_8nm_x2304_y2048_z6144", "flyem3d_8nm_x2304_y2048_z6144"),
    ("FLYEM3D", "FIB-25 (surround)", "SSL", "Train"): ("data/FLYEM3D", "flyem3d_8nm_ssl_x4096_y4096_z1024", None),
    ("FLYEM3D", "FIB-25 (surround)", "SSL", "Test"): ("data/FLYEM3D", "flyem3d_8nm_ssl_x2700_y2700_z3500", None),
    ("FLYEM3D", "Hemibrain", "SSL", "Train"): ("data/FLYEM3D", "flyem3d_hemibrain_8nm_ssl_x12000_y12000_z12000", None),
    ("FLYEM3D", "Hemibrain", "SSL", "Test"): ("data/FLYEM3D", "flyem3d_hemibrain_8nm_ssl_x16000_y16000_z16000", None),
    ("FLYEM3D", "MaleCNS", "SSL", "Train"): ("data/FLYEM3D", "flyem3d_malecns_8nm_ssl_x18000_y18000_z30000", None),
    ("FLYEM3D", "MaleCNS", "SSL", "Test"): ("data/FLYEM3D", "flyem3d_malecns_8nm_ssl_x16000_y20000_z24000", None),
    ("FLYWIRE", "FAFB v783", "SFT", "Train"): (
        "data/FLYWIRE",
        "flywire_mip1_1024x1024x256_x88000_y20000_z3600",
        "flywire_mip1_1024x1024x256_x88000_y20000_z3600_m783",
    ),
    ("FLYWIRE", "FAFB v783", "SFT", "Test"): (
        "data/FLYWIRE",
        "flywire_mip1_1024x1024x256_x80000_y16000_z4500",
        "flywire_mip1_1024x1024x256_x80000_y16000_z4500_m783",
    ),
    ("MICrONS", "minnie65", "SFT", "Train"): (
        "data/MICRONS",
        "minnie65_mip0_4096x4096x800_x50000_y60000_z16000",
        "minnie65_mip0_4096x4096x800_x50000_y60000_z16000_v1300",
    ),
    ("MICrONS", "minnie65", "SFT", "Test"): (
        "data/MICRONS",
        "minnie65_mip0_4096x4096x800_x70000_y90000_z17000",
        "minnie65_mip0_4096x4096x800_x70000_y90000_z17000_v1300",
    ),
    ("MitoEM2", "beta", "SSL", "Train"): ("data/MitoEM2", "mitoem2_beta_train01", None),
    ("MitoEM2", "beta", "SSL", "Test"): ("data/MitoEM2", "mitoem2_beta_test01", None),
    ("MitoEM2", "jurkat", "SSL", "Train"): ("data/MitoEM2", "mitoem2_jurkat_train01", None),
    ("MitoEM2", "jurkat", "SSL", "Test"): ("data/MitoEM2", "mitoem2_jurkat_test01", None),
    ("MitoEM2", "macro", "SSL", "Train"): ("data/MitoEM2", "mitoem2_macro_train01", None),
    ("MitoEM2", "macro", "SSL", "Test"): ("data/MitoEM2", "mitoem2_macro_test01", None),
    ("MitoEM2", "podo", "SSL", "Train"): ("data/MitoEM2", "mitoem2_podo_train01", None),
    ("MitoEM2", "podo", "SSL", "Test"): ("data/MitoEM2", "mitoem2_podo_test01", None),
    ("MitoEM2", "sperm", "SSL", "Train"): ("data/MitoEM2", "mitoem2_sperm_train01", None),
    ("MitoEM2", "sperm", "SSL", "Test"): ("data/MitoEM2", "mitoem2_sperm_test01", None),
    ("MitoEM2", "mossy", "SSL", "Train"): ("data/MitoEM2", "mitoem2_mossy_train01", None),
    ("MitoEM2", "mossy", "SSL", "Test"): ("data/MitoEM2", "mitoem2_mossy_test01", None),
    ("MitoEM2", "pyra", "SSL", "Train"): ("data/MitoEM2", "mitoem2_pyra_train01", None),
    ("MitoEM2", "pyra", "SSL", "Test"): ("data/MitoEM2", "mitoem2_pyra_test01", None),
    ("Neurons", "Kasthuri cylinder", "SFT", "Train"): (
        "data/SNEMI3D",
        "neurons_5000x2900x300_x3000_y7200_z950",
        "neurons_5000x2900x300_x3000_y7200_z950",
    ),
    ("SNEMI3D", "AC4", "SFT", "Train"): ("data/SNEMI3D", "AC4_inputs", "AC4_labels"),
    ("SNEMI3D", "AC3", "SSL", "Train"): ("data/SNEMI3D", "AC3_inputs", None),
}


def _vol_path(root: Path, stem: str) -> Path:
    for suffix in ("_volume.h5", ".h5"):
        p = root / f"{stem}{suffix}"
        if p.exists():
            return p
    raise FileNotFoundError(f"volume not found for stem {stem} under {root}")


def _seg_path(root: Path, stem: str) -> Path:
    candidates = [
        f"{stem}_segmentation.h5",
        f"{stem}_labels.h5",
        f"{stem}.h5",
    ]
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    raise FileNotFoundError(f"segmentation not found for stem {stem} under {root}")


_MITO_SUBSET = {
    "beta": "beta",
    "jurkat": "jurkat",
    "macro": "macro",
    "podo": "podo",
    "sperm": "sperm",
    "mossy": "mossy",
    "pyramidal": "pyra",
}


def _row_key(row: dict) -> Tuple[str, str, str, str]:
    subset = row["Subset"]
    if row["Dataset"] == "MitoEM2":
        token = subset.split()[0].lower()
        subset = _MITO_SUBSET.get(token, token)
    return row["Dataset"], subset, row["Task"], row["Split"]


def _preview_for_row(row: dict) -> str:
    key = _row_key(row)
    if key not in REPRESENTATIVE:
        raise KeyError(f"No representative mapping for {key}")
    root, vol_stem, seg_stem = REPRESENTATIVE[key]
    root_p = _REPO / root
    slug = _slug(row["Dataset"], key[1], row["Task"], row["Split"])
    # Disambiguate repeated census rows (e.g. multiple beta-train size entries).
    size_tag = re.sub(r"[^a-z0-9]+", "_", row.get("Size crop", "").lower()).strip("_")
    if size_tag:
        slug = f"{slug}_{size_tag}"
    out = _OUT / f"{slug}.png"
    title = f"{row['Dataset']} · {row['Subset']} · {row['Task']} · {row['Split']}"

    em_slice = _norm_em(_read_slice(_vol_path(root_p, vol_stem)))
    if row["Task"] == "SSL":
        _save_ssl(em_slice, title, out)
    else:
        seg_slice = _read_slice(_seg_path(root_p, seg_stem or vol_stem))
        _save_sft(em_slice, seg_slice, title, out)
    return str(out.relative_to(_REPO))


def main() -> None:
    with _CSV.open(newline="") as f:
        rows: List[dict] = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    if "Preview" not in fieldnames:
        fieldnames.append("Preview")

    for row in rows:
        rel = _preview_for_row(row)
        row["Preview"] = rel
        print(f"  {rel}")

    with _CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} previews -> {_OUT}/")
    print(f"Updated {_CSV}")


if __name__ == "__main__":
    main()
