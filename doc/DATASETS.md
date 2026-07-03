# Datasets

Every dataset nanocosmos trains on is **3-D electron-microscopy (EM)**, stored
on disk in one shared convention and pulled in by the datamodule. Most carry
dense **instance** (neuron-id) labels for the segmentation (`sft`) branch; the
self-supervised (`ssl`) sources — COSEM3D, MitoEM2, and the unlabeled FlyEM
surround — are **image-only**. This doc covers what each dataset is, its native
resolution, and the exact script that downloads (or converts) it.

> Resolutions are quoted as the source reports them (usually `x × y × z`
> nm). The config's `resolution_map` uses **`(z, y, x)`** order — both are
> given below so there's no ambiguity.

## At a glance

| Dataset | Tissue / EM modality | Native res (x,y,z nm) | `resolution_map` (z,y,x) | Publication | Download / convert | Data root | Config key (`data.dataset`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SNEMI3D | Mouse S1 cortex, **ssSEM** (AC3/AC4) | 6 × 6 × 30 | `AC: [30,6,6]` | Kasthuri et al. 2015, *Cell* 162:648 | `download_snemi3d.py` | `data/SNEMI3D` | `snemi3d` |
| Neurons | Mouse S1 cortex, **ssSEM** (Kasthuri cylinder) | 6 × 6 × 30 | `neurons: [30,6,6]` | Kasthuri et al. 2015, *Cell* 162:648 | `download_snemi3d.py` | `data/SNEMI3D` | `neurons` |
| MICrONS | Mouse V1 cortex, **ssTEM** (minnie65) | 8 × 8 × 40 | `minnie65: [40,8,8]` | MICrONS Consortium 2025, *Nature* (preprint bioRxiv 2021.07.28.454025) | `download_microns.py` | `data/MICRONS` | `microns` |
| CREMI3D | *Drosophila* brain, **ssTEM** (A/B/C) | 4 × 4 × 40 | `cremi3d: [40,4,4]` | CREMI challenge 2016 (cremi.org); FAFB Zheng et al. 2018, *Cell* | `download_cremi3d.py` | `data/CREMI3D` | `cremi3d` |
| FLYEM3D | FlyEM *Drosophila*, **FIB-SEM** (FIB-25 / Hemibrain / MaleCNS) | 8 × 8 × 8 (isotropic) | `flyem3d: [8,8,8]` | Takemura 2015 *PNAS* (FIB-25); Scheffer 2020 *eLife* (Hemibrain); Berg 2025 *bioRxiv* (MaleCNS) | `download_flyem3d.py` | `data/FLYEM3D` | `flyem3d` / `joint3d` |
| COSEM3D | OpenOrganelle / COSEM cell, **FIB-SEM** | 4 × 4 × ~3.2–5.2 (near-cubic) | *(joint3d SSL anchor)* | Xu et al. 2021 *Nature*; Heinrich et al. 2021 *Nature* | `download_cosem3d.py` | `data/COSEM3D` | `joint3d` |
| MitoEM2 | Mitochondria EM, **mixed FIB-SEM / ssSEM / SBF-SEM** (8 subsets) | 16 × 16 × 16 & 8 × 8 × 30 | *(joint3d SSL, per-vol)* | Liu, P. 2026, Zenodo (MitoEM 2.0, v6, [10.5281/zenodo.20417683](https://zenodo.org/records/20417683)); orig. Wei et al. 2020, *MICCAI* | `convert_mitoem2.py` | `data/MitoEM2` | `joint3d` |

`COSEM3D` (4 nm), `MitoEM2` (8–16 nm), and the Hemibrain / MaleCNS members of
`FLYEM3D` (8 nm) are the image-only `ssl` rungs of the **joint super-resolution
recipe** (`configs/nanocosmos-{16B,4B,2B}.yaml`,
`data.dataset: joint3d`); see [`RESOLUTION_LADDER.md`](./RESOLUTION_LADDER.md).

All scripts live in `scripts/` and write the on-disk convention described
in [On-disk convention](#on-disk-convention). The multi-dataset
"foundation" recipe (`configs/cosmos3nano3d.yaml`) mixes SNEMI3D,
Neurons, MICrONS, CREMI3D, and FLYEM3D in a single run by listing volumes
from several `data/<root>` directories under one datamodule.

---

## On-disk convention

The 3-D training path (`LazyVolDataset`, used whenever
`slice_mode: false` + `patch_size`) reads patches **on demand** from
HDF5, so every volume is stored as **two separate `.h5` files**:

```
<root>/<name>_volume.h5         # EM intensity   (uint8/float)
<root>/<name>_segmentation.h5   # instance ids   (int/uint, 0 = background)
```

- Dataset key inside each file: **`main`** (the loader also falls back to
  `data`/`raw`/`volume`/`image`/`label`).
- Axis order: **`[Z, Y, X]`** (z = section axis).
- A volume is referenced in YAML as `{vol: <name>_volume, seg: <name>_segmentation, root: <dir>}`.
- `LazyVolDataset` finds files by base name with extensions
  `.h5/.hdf5/.tif/.tiff` — so a packed/nested HDF5 (e.g. CREMI's native
  `.hdf` with `volumes/raw` + `volumes/labels/neuron_ids`) must be
  **converted** first; the CREMI/FIB scripts do this at download time.

The dataset/datamodule classes (`SNEMI3DDataset`, `MICRONSDataset`,
`CREMI3DDataset`, `FLYEM3DDataset`, `NeuronsDataset` and their
`*DataModule`s) only differ in metadata; the CREMI3D and FLYEM3D leaves
are thin metadata subclasses of `MICRONSDataset`/`MICRONSDataModule`.

---

## SNEMI3D (AC3 / AC4)

- **What:** The SNEMI3D challenge crops from Kasthuri et al. 2015 mouse
  somatosensory cortex (**ssSEM**, ATUM tape-collecting + SEM). `AC4` = train (1024 × 1024 × 100, EM +
  labels); `AC3` = test (1024 × 1024 × 100, **EM only** — labels were
  never publicly released).  In the joint configs `AC4` is the labeled `sft`
  holdout (`val_volumes`) and the label-less `AC3_inputs` is an image-only
  `ssl` train source.
- **Resolution:** 6 × 6 × 30 nm (anisotropic).
- **Source:** `snemi.zip` (rhoana / Zenodo). AC3/AC4 sit at Y ≈ 5440 in
  the `kasthuri11` volume, outside the GCS ground-truth cylinder.
- **Citation:** Kasthuri, N. et al. (2015), *Saturated Reconstruction of
  a Volume of Neocortex*, Cell 162(3):648-661.

```bash
python scripts/download_snemi3d.py --source snemi      # AC3 EM + AC4 EM/labels
python scripts/download_snemi3d.py --link /scratch/SNEMI3D   # or symlink existing
```

Files land in `data/SNEMI3D/` (e.g. `AC4_inputs`, `AC4_labels`).

---

## Neurons (Kasthuri annotated cylinder)

- **What:** The densely-annotated cylinder from the **same** Kasthuri
  2015 volume, fetched from the public `kasthuri2011` Google bucket. A
  single large crop `5000 × 2900 × 300` at start `(x=3000, y=7200, z=950)`
  inside the annotated region (X≈3000–8000, Y≈7200–10100, Z≈950–1250).
- **Resolution:** 6 × 6 × 30 nm (as downloaded / as used in the config).
- **Source:**
  `gs://neuroglancer-public-data/kasthuri2011/{image_color_corrected, ground_truth}`.

```bash
python scripts/download_snemi3d.py --source neurons
# custom crop:
python scripts/download_snemi3d.py --source neurons --start 3000 7200 950 --size 5000 2900 300
# everything (SNEMI3D + neurons):
python scripts/download_snemi3d.py --source all
```

Files land in `data/SNEMI3D/` (config volume
`neurons_5000x2900x300_x3000_y7200_z950_{volume,segmentation}`).

---

## MICrONS (minnie65)

- **What:** IARPA MICrONS mouse primary visual cortex (V1), ~1 mm³,
  ~120k neurons, automated serial-section TEM (**ssTEM**). We use
  representative sub-volumes for training.
- **Resolution:** **8 × 8 × 40 nm** — this is the EM *imagery*
  resolution (mip 0 of the released precomputed bucket). The often-quoted
  **4 × 4 × 40 nm** is the *annotation/coordinate frame*, not the image
  voxel size.
- **Segmentation versions:** `v117`, `v343`, `v943`, `v1300` (default,
  latest, Jan 2025).
- **Splits:** 12 pre-defined `4096 × 4096 × 800` crops (10 train + 2
  test) at disjoint XY positions / cortical depths; file names encode the
  origin, e.g.
  `minnie65_mip0_4096x4096x800_x50000_y60000_z16000_volume.h5`.
- **Source:** AWS / GCS public buckets via `cloud-volume`
  (`.../iarpa_microns/minnie/minnie65/em`).
- **Citation:** MICrONS Consortium (2025), *Functional connectomics spanning
  multiple areas of mouse visual cortex*, Nature (preprint bioRxiv
  2021.07.28.454025).

```bash
python scripts/download_microns.py --split                       # 10 train + 2 test, v1300
python scripts/download_microns.py --size 4096 4096 800 --seg-version 1300   # custom
python scripts/download_microns.py --seg-version all             # all 4 seg versions
```

Files land in `data/MICRONS/`. Crop size guide (mip0, uint8 EM +
uint64 seg): `512³` ≈ 1.1 GB, `1024³` ≈ 9 GB, `2048³` ≈ 72 GB,
`4096×4096×800` ≈ tens of GB per crop.

---

## CREMI3D

- **What:** CREMI (MICCAI 2016), adult *Drosophila* brain ssTEM.
  - **A, B, C** — labelled TRAINING volumes (`1250 × 1250 × 125`, dense
    neuron ids) → `train_volumes`.
  - **A+, B+, C+** — padded TEST volumes; public EM only (challenge
    withholds the labels) → converted **image-only**.  No GT = no seg
    metrics, so they are never in any `test_volumes`; instead the joint
    configs list them under `data.branches.ssl` (image-only reconstruction).
- **Resolution:** 4 × 4 × 40 nm (anisotropic; 10:1 z:xy).
- **Source:** `https://cremi.org/static/data/sample_{A,B,C}_20160501.hdf`
  (raw + labels packed in one nested `.hdf`).
- **Conversion:** the script downloads the official `.hdf` and writes the
  nanocosmos convention (`cremi3d_sample_<X>_{volume,segmentation}.h5`, key
  `main`, `[Z,Y,X]`), mapping CREMI's "no-data" marker to background.
- **Citation:** Funke, Saalfeld, Bock, Turaga, Perlman — CREMI Challenge,
  https://cremi.org/.

```bash
# downloads + converts all six (A,B,C labelled + A+,B+,C+ image-only test)
python scripts/download_cremi3d.py --out-dir data/CREMI3D
# training only:
python scripts/download_cremi3d.py --out-dir data/CREMI3D --samples A B C
# reuse already-downloaded .hdf (skip the network):
python scripts/download_cremi3d.py --out-dir data/CREMI3D --hdf-dir /scratch/CREMI3D
```

Files land in `data/CREMI3D/` (`cremi3d_sample_A_volume`, … and the
image-only `cremi3d_sample_A+_volume`, …).

---

## FLYEM3D (FIB-25)

- **What:** Janelia FlyEM 7-column *Drosophila* medulla FIB-SEM
  reconstruction. Dense neuron instance labels.
- **Resolution:** **8 × 8 × 8 nm isotropic** at mip 0 (doubles per mip,
  7 mip levels). Full volume `6446 × 6643 × 8090` voxels (x,y,z).
- **Segmentation coverage:** the ground truth covers only **~8.65%** of
  the full volume; the labeled bounding box is `x[1856:5024]
  y[1664:4288] z[1472:8000]` (≈ 25 × 21 × 52 µm) and is ≈55% filled.
  Fully-dense crops top out around `1024³` (≈100% fg); the chosen
  **primary training core is `1536³` at origin `(2304, 2048, 6144)`**
  (≈12.3 µm, **~99.7% foreground**).
- **Source:** `gs://neuroglancer-public-data/flyem_fib-25/{image, ground_truth}`
  (Neuroglancer precomputed) via `cloud-volume`.
- **Citation (FIB-25):** Takemura, S. et al. (2015), PNAS 112(44):13711-13716,
  doi:10.1073/pnas.1509820112.
- **Other FLYEM3D members:** Hemibrain — Scheffer, L.K. et al. (2020),
  *eLife* 9:e57443 (FIB-SEM); MaleCNS — Berg, S. et al. (2025), *bioRxiv*
  2025.10.09.680999 (eFIB-SEM, 8 nm isotropic).

- **Joint-recipe SSL crop set (image-only).** Hemibrain and MaleCNS each
  contribute **4 train + 1 val** `1024³` crops; the FIB-25 surround now
  contributes **2 train + 1 val** (the two empty corner tiles
  `x1024_y4096_z4096` and `x4096_y1024_z4096` were removed after the content
  gate flagged them). Listed in `configs/nanocosmos-{16B,4B,2B}.yaml`
  (`data.branches.ssl` / `val_volumes`). **Caveat:** Hemibrain / MaleCNS tissue
  does **not** fill its bounding box, so corner / off-tissue origins (e.g.
  `x4000_y4000_z4000`) download as all-zero tiles (CloudVolume
  `fill_missing=True`). Pick origins inside the imaged region (Hemibrain
  ~`x12–24k`; MaleCNS ~`x16–28k, y20–30k, z20–36k`) and confirm a non-zero
  central crop before adding it. The `data.ssl_min_std` content gate rejects
  flat / mostly-empty crops at train time (and `data.ssl_min_autocorr` rejects
  noise-dominated ones — see [SSL crop cleaning](#ssl-crop-cleaning-content--structure-gates)),
  but empty *volumes* should still be removed from the config (see
  `doc/data.csv` for the current census).

Recommended: fetch the native cube **once**, then generate all variants
locally with `--from-local` (no re-download).

```bash
# 1. primary 1536^3 dense core (native 8 nm isotropic)
python scripts/download_flyem3d.py --out-dir data/FLYEM3D --name flyem3d \
    --origin 2304 2048 6144 --size 1536 1536 1536 --mip 0

# 2. isotropic orientation variants (thin axis z / y / x) from the local cube
python scripts/download_flyem3d.py --from-local \
    data/FLYEM3D/flyem3d_8nm_x2304_y2048_z6144_volume.h5 \
    --name flyem3d --orientations z y x

# 3. anisotropic 32 nm copies, all z-stride-4 phase offsets (p0..p3)
python scripts/download_flyem3d.py --from-local \
    data/FLYEM3D/flyem3d_8nm_x2304_y2048_z6144_volume.h5 \
    --name flyem3d --z-stride 4
```

Files land in `data/FLYEM3D/`
(`flyem3d_8nm_x2304_y2048_z6144_{volume,segmentation}`). `--size` is
clamped to the volume bounds; the script loads the whole crop into RAM,
so it is built for crops, not the full petavoxel volume (the full image
is ~346 GB / seg ~2.8 TB at mip 0).

---

## SSL crop cleaning (content + structure gates)

The `ssl` branch is **image-only** (COSEM3D, MitoEM2, FlyEM Hemibrain/MaleCNS
surround, CREMI+ test), and its only supervision is reconstructing the clean EM
target — so a junk crop trains and *visualises* as junk. `LazyVolDataset` gates
every SSL crop at read time and re-samples up to `max_foreground_retries` (50),
keeping the **best-seen** crop if all fail. Three complementary gates run on the
per-volume `[0, 1]` scale (config keys under `data.`):

| gate | config key | rejects | how |
| --- | --- | --- | --- |
| non-zero | `ssl_min_foreground` (0.8) | mostly-empty / zero-padded crops | fraction of non-zero voxels |
| contrast | `ssl_min_std` (0.05) | **flat** resin / embedding medium | fraction of 4×16×16 blocks whose local std clears the threshold (`content_frac`) |
| **structure** | `ssl_min_autocorr` (0.5) | **noise-dominated** crops (detector / resin grain) | lag-1 spatial autocorrelation |

**Why the structure gate is needed.** A variance/std test *cannot* tell real
ultrastructure from noise — random detector/resin grain has **high** local
variance, so it passes `ssl_min_std` with `content_frac = 1.0`. This surfaced as
noise-dominated SSL panels traced to **COSEM3D** volumes (`jrc_hela-3`,
`jrc_jurkat-1`): crops landing in the embedding **resin** around the cell are
low-contrast grain that per-crop display normalisation stretches to full
contrast. Real EM is spatially coherent (neighbouring voxels correlated) while
grain is ~white noise, so the **lag-1 spatial autocorrelation** separates them:

| crop type | `content_frac` (`ssl_min_std`) | lag-1 autocorr |
| --- | --- | --- |
| resin / detector grain | 1.00 (passes) | ~0.23–0.28 |
| genuine ultrastructure | 0.97–1.00 | ~0.68–0.79 |

The gap is wide, so `ssl_min_autocorr: 0.5` sits safely between. Validation on
real crops (40 random each): `jrc_hela-3` z8192 accept-rate 33/40 → **11/40**
(the 22 culled crops are all `content_frac 1.0`, autocorr < 0.4 — exactly the
noise), while structured volumes are untouched (`jrc_jurkat-1` 24→24,
`flyem3d_hemibrain` 40→40). Tune `ssl_min_autocorr` down (e.g. 0.45) if the
retry budget is exhausted on a legitimately grainy source.

**Gates ≠ volume curation.** These operate per-crop; an entirely empty/junk
*volume* should still be removed from the config (corner/off-tissue Hemibrain/
MaleCNS tiles, empty MitoEM2 crops). See `doc/data.csv` for the current census.

**Re-collecting bad crops from the cloud.** `scripts/collect_meaningful_crops.py`
automates the fix for CloudVolume-backed sources (COSEM3D, FLYEM3D): instead of
hand-picking a replacement `--origin`, it probes a grid of candidate crop cells
with several small random sub-patches (matching the SSL training-patch scale),
scores each with the *same* `content_frac` + `autocorr` (+ non-zero) gates, and
only pays for a full-crop download on the candidates that pass -- printing
paste-ready config entries exactly like `download_cosem3d.py` /
`download_flyem3d.py`. MitoEM2 has no CloudVolume source (see above), so it is
not supported by this script; audit its already-converted `.h5` files directly
with the same metrics instead.

This was used to fix three confirmed-bad crops (Jul 2026 audit):
* `jrc_hela-3` `x0_y0_z0` / `x0_y0_z8192` (train) and `x8192_y0_z0` (val
  holdout) -- median autocorr ~0.26-0.27 (embedding resin) -- replaced with
  `x4096_y0_z2048` / `x2048_y0_z4096` (train) and `x6144_y0_z0` (val),
  autocorr 0.63-0.75.
* `flyem3d_malecns_8nm_ssl_x20000_y20000_z20000` -- autocorr 0.45 (borderline,
  vs ~0.85-0.95 for the rest of the bucket) -- replaced with
  `x18000_y18000_z30000`, autocorr 0.94.

```bash
# Search + fetch replacement COSEM3D crops (small volumes -- grid-searchable directly):
python scripts/collect_meaningful_crops.py --source cosem3d --dataset jrc_hela-3 --num-crops 2

# FLYEM3D hemibrain/malecns are ~10^5 vox/axis -- MUST restrict the search to the
# known-imaged core (see the FLYEM3D section above) and use a coarse --grid-stride
# for the initial sparse scan (default stride = --size is far too fine at this scale):
python scripts/collect_meaningful_crops.py --source flyem3d --dataset malecns --role ssl \
    --size 1024 1024 1024 --num-crops 1 \
    --search-bounds 14000 30000 18000 32000 18000 38000 --grid-stride 4000 4000 4000
```

### Known NOT-MEANINGFUL crops (excluded from config; files kept on disk)

The `.h5` files below **remain on disk** (not deleted, in case anyone wants to
re-inspect them) but are **removed from every config** -- do not re-add them.
Each was confirmed junk by the audit above (not a one-off borderline score):

| file | root | measured | verdict |
| --- | --- | --- | --- |
| `jrc_hela-3_4x4x3.24nm_x0_y0_z0_volume.h5` | `data/COSEM3D` | autocorr ~0.31 (avg over 15 probes) | embedding resin, ~white noise |
| `jrc_hela-3_4x4x3.24nm_x0_y0_z8192_volume.h5` | `data/COSEM3D` | median autocorr ~0.27 (of 40 probes) | embedding resin, ~white noise |
| `jrc_hela-3_4x4x3.24nm_x8192_y0_z0_volume.h5` | `data/COSEM3D` | autocorr ~0.27 | embedding resin (was the ssl val holdout) |
| `flyem3d_malecns_8nm_ssl_x20000_y20000_z20000_volume.h5` | `data/FLYEM3D` | autocorr ~0.45 | outlier vs ~0.85-0.95 for the rest of the bucket |
| `mitoem2_jurkat_train02_volume.h5` | `data/MitoEM2` | autocorr ~0.00; min=0 max=255, all 256 values present, std≈73.9 (≈ theoretical max) | **pure uniform random noise** -- corrupt in the upstream `.nii.gz` release. **Confirmed persistent** in the Jul 2026 MitoEM 2.0 v6 refresh: these are the ONLY 2 of ~90 files in the whole release that fail SHA256 verification against the official `checksums.txt` -- see [MitoEM2 refresh from Zenodo v6](#mitoem2-refresh-from-zenodo-v6-jul-2026) |
| `mitoem2_macro_train02_volume.h5` | `data/MitoEM2` | same as above | same as above |
| `jrc_macrophage-2_4x4x3.36nm_x7952_y0_z0_volume.h5` | `data/COSEM3D` | autocorr ~0.33, 44.7% of voxels outside the cached norm range | resin + severely under-calibrated norm cache (was the ssl val holdout) |
| `jrc_jurkat-1_4x4x3.44nm_x7952_y0_z0_volume.h5` | `data/COSEM3D` | autocorr ~0.47, 14.1% out-of-range | milder version of the same pattern (was the ssl val holdout) |
| `mitoem2_stem_train01_volume.h5` | `data/MitoEM2` | 25-probe autocorr: max 0.49, median 0.40, 0/25 probes reach 0.5 | consistently low-structure, no good sub-region for the per-crop gate to land on. **Confirmed persistent** in the v6 refresh (re-audited fresh conversion, same result) |
| `mitoem2_stem_train02_volume.h5` | `data/MitoEM2` | max 0.49, median 0.44, 0/25 probes reach 0.5 | same as above; persistent in v6 |
| `mitoem2_stem_test01_volume.h5` | `data/MitoEM2` | median 0.45, only 24% of probes reach 0.5 | same pattern, slightly less bad (was the ssl val holdout; no train counterpart survives to pair it with anyway); persistent in v6 |

### Full dataset meaningfulness audit (Jul 2026)

Every volume referenced anywhere in `configs/nanocosmos-2B.yaml` (83 unique
files: `ssl` + `sft` train branches + `val_volumes`) was audited the same way:
~15 random sub-patches per file (scaled to the SSL training-patch size), each
scored for `content_frac` (`ssl_min_std` gate), lag-1 `autocorr`
(`ssl_min_autocorr` gate), non-zero fraction, and **out-of-range %** -- the
fraction of voxels that fall outside the file's *cached* `.h5.norm.json`
min/max (that cache is intentionally a fast 5-small-probe estimate, see
`LazyVolDataset._compute_norm_params`, so a *few percent* drift is normal, not
a defect -- only large drift is worth flagging).

Besides the three crops already fixed above, the sweep found (and, as of this
writing, **all of these have since been fixed too** -- see the updated
"excluded crops" table above and the ledger below):

| file | branch | autocorr | out-of-range | issue | resolution |
| --- | --- | --- | --- | --- | --- |
| `jrc_macrophage-2_..._x7952_y0_z0` (val holdout) | ssl(val) | **0.33** | **44.7%** | resin + severely under-calibrated norm cache (cached 217-232 vs observed 217-255) | replaced with `x2048_y0_z4096` (autocorr 0.85) |
| `jrc_jurkat-1_..._x7952_y0_z0` (val holdout) | ssl(val) | 0.47 (borderline) | 14.1% | same pattern, milder | replaced with `x0_y0_z2048` (autocorr 0.84) |
| `mitoem2_stem_train01_volume` | ssl | **0.37** | -- | uniformly low structure (0/25 probes reach 0.5) | removed (no cloud source) |
| `mitoem2_stem_train02_volume` | ssl | 0.44 (borderline) | -- | uniformly low structure (0/25 probes reach 0.5) | removed (no cloud source) |
| `mitoem2_stem_test01_volume` (val) | ssl(val) | 0.47 (borderline) | -- | only 24% of probes reach 0.5 | removed (no cloud source, no surviving train counterpart) |
| `jrc_jurkat-1_..._x4096_y0_z0` / `_x4096_y0_z4096` (in-use train) | ssl | 0.59-0.66 (OK) | 12-13% | content is fine; the norm cache is meaningfully too narrow for these tiles (real range extends to 255, cache tops out at 238) | left as-is -- content is genuinely fine, this is a norm-cache calibration note, not a removal candidate |

**Fixing `collect_meaningful_crops.py` itself.** Chasing the `jrc_macrophage-2`
replacement surfaced a real bug in the tool: it scored `content_frac` against
a **hardcoded 0-255 range**, but `jrc_macrophage-2` (like `jrc_jurkat-1`) is a
high-key source living in a much narrower band (e.g. ~140-250); dividing by
the wrong (too-wide) range shrank every candidate's normalised local std well
below the 0.05 threshold, rejecting every candidate regardless of true
quality (`autocorr` itself is scale-invariant -- a ratio of covariance to
variance -- so it was unaffected). Fixed by adding `_estimate_norm_range()`
(mirrors `LazyVolDataset._compute_norm_params`'s handful-of-probes approach)
and using it for `content_frac` scoring instead of a fixed range.

Two things audited and judged **not** a problem, despite showing up as flagged
rows in the raw sweep:

* **Two `jrc_hela-3` tiles average only 0.50-0.52 autocorr over the *whole*
  2048³ file**: `x4096_y0_z0` (an *original*, pre-existing tile, never
  previously re-checked at this granularity) and `x2048_y0_z4096` (one of the
  *newly-fetched* replacements, which scored 0.75 at the specific sub-region
  `collect_meaningful_crops.py` probed). HeLa cells are ~10-20 µm and a
  2048-vox (~8 µm) COSEM box can straddle both cell and resin, so a
  *whole-file* average is not the right yardstick: the actual per-crop
  `ssl_min_autocorr` gate at train time re-samples within the file until it
  lands a good patch, so a heterogeneous file is fine as long as it has *some*
  good structure to find -- unlike the removed tiles, which were bad almost
  everywhere (median ~0.26-0.31, i.e. no good sub-region to land on).
* **`mitoem2_beta_train01-04` / `beta_test02`** (nz_frac ~0.39-0.46): this is
  the documented irregular nnU-Net crop padding (a non-rectangular organ mask
  padded to a bounding box -- see the MitoEM2 note above), not corruption; the
  per-crop `image_min_foreground` gate already handles it at train time.
* The **`CALIB` mismatches on nearly every other volume** (a few-percent
  cached-vs-observed drift) are expected noise from the intentionally-cheap
  5-probe norm estimate, not a data defect.

All six newly-found issues above have since been resolved: two CloudVolume
replacements (`jrc_macrophage-2`, `jrc_jurkat-1` val holdouts) and three
config removals (`mitoem2_stem_train01/02`, `mitoem2_stem_test01` -- see the
"excluded crops" table above for the exact files kept on disk but no longer
referenced). The `jrc_jurkat-1_x4096_*` norm-cache drift was left as a
documentation note only -- the content itself is fine, just imprecisely
normalised by a few percent, which is within the tolerance the cheap 5-probe
cache was designed to accept.

### Labeled (sft) volume audit -- image + instance-label meaningfulness (Jul 2026)

Every prior audit pass covered only the label-free `ssl` branch (the crop is
its own supervision, so junk is directly visible/trainable-on). This pass
extends the same rigor to all **18 labeled (`sft`) volumes** -- the 15 train
+ `AC4_inputs` + 2 MICrONS val holdouts -- checking both the paired **image**
and the **instance label**.

**Image side (all 18):** re-ran the same `content_frac` / `autocorr` / `nz`
probe sweep on the EM image half of every labeled pair. All pass cleanly
(`content` 0.65-1.00, `autocorr` 0.86-0.98, `nz` 0.75-1.00 across 15 random
probes each) -- no resin/noise/off-tissue pattern like the ones found and
fixed on the `ssl` side.

**Label side (new):** for each labeled volume, sampled 20 random patches and
measured, per patch: foreground fraction, instance count, and the largest
instance's share of the labeled foreground (`max_inst_frac` -- a value near 1
with `n_inst` = 1 would mean a degenerate single-blob "label" rather than a
real instance segmentation). Results: **no volume shows a whole-file
degenerate pattern** -- every dataset's *median* instance count per patch is
healthy (CREMI 21-126, MICrONS 31-48, FLYEM3D 37.5, AC4 21, Neurons 3).

The **minimum** across patches told a different story for two datasets, and
only one of them turned out to already be handled:

* **Neurons (Kasthuri cylinder)**: one probe found `n_inst = 0`, `fg = 0.00`
  (a fully-unlabeled patch) -- real background/gap tissue exists inside the
  annotated bounding box. **Already handled**: the existing
  `sft_min_foreground: 0.8` gate (`min_foreground` +
  `image_min_foreground` in `LazyVolDataset`) rejects/re-samples exactly
  this case -- the same "heterogeneous file, per-crop gate finds the good
  regions" pattern as the `jrc_hela-3` SSL tiles.
* **MICrONS**: `n_inst` drops to 1 with `worst_maxfrac = 1.00` in 10 of 12
  crops -- a small patch landing entirely inside one large dendrite trunk (a
  real, large-scale structure in this tissue). **NOT actually handled**:
  `sft_min_foreground` only checks the label's *non-zero fraction*, and a
  crop filled edge-to-edge by one giant instance has `label_frac ~ 1.0` --
  it passes trivially. A single-instance crop gives `AffinityFGLoss` no
  inter-instance "push" signal to learn from at all, only "pull" -- a real,
  previously un-gated gap.

**Fix: a new instance-diversity gate.** Added `sft_min_instances` /
`sft_max_inst_frac` to `LazyVolDataset` / `Joint3DDataModule` (config keys
under `data.`, scoped to the `sft` branch only, mirroring how
`ssl_min_autocorr` is scoped to `ssl`): a crop is now also rejected if it has
fewer than `sft_min_instances` distinct nonzero instance ids, or if the
single largest instance exceeds `sft_max_inst_frac` of the labeled
foreground. Set to `sft_min_instances: 2` / `sft_max_inst_frac: 0.9` --
every dataset's *median* instance count (20-125) makes this trivially
satisfiable on retry, so it only removes the genuinely degenerate patches.
Validated directly against real MICrONS data: 40 sampled crops with the gate
active had `n_inst` min **14** (previously as low as 1), zero single-instance
crops slipped through.

**An experimental image-label alignment check -- tested, and rejected.** To
rigorously check whether an image/label *pair* could be mismatched or
corrupted (as opposed to each half being individually fine), a metric was
tried: the ratio of mean image-gradient magnitude *at* label-boundary voxels
vs *away* from them (`align_ratio`) -- real segmentation boundaries should
sit on real membrane edges, so a well-aligned pair should score `>> 1`; a
shuffled/misaligned pair should score `~1`. The metric was **validated on a
synthetic case first** (a clean image with a 1-voxel bright membrane exactly
on a label boundary scored 10.7; the same image against a spatially-shifted
copy of the label scored 0.87 -- confirming the method works when the ground
truth is known).

Applied to the real data, though, **every dataset scored close to or below 1**
(CREMI 0.75-0.82, MICrONS 0.78-0.94, FLYEM3D FIB-25 0.90, Neurons/AC4
1.12-1.16) -- including **`flyem3d_8nm_x2304_y2048_z6144`, the professionally
proofread FIB-25 core**, which is about as trusted a ground-truth label as
exists in this codebase. Since even that reference case fails the test, the
honest conclusion is that a crude single-scale finite-difference gradient is
**not sensitive enough for real, noisy, anisotropic EM texture** (unlike the
clean synthetic case) -- **this is a methodology limitation, not a data
defect**, and `align_ratio` was **not** used as a pass/fail signal for any
volume. Documented here so nobody re-derives this same metric later and
mistakenly flags good data as corrupt.

**Verdict:** all 18 labeled volumes are confirmed meaningful -- no volume was
removed or replaced -- but the audit did surface one real gap (single-instance
MICrONS crops), fixed with the new `sft_min_instances` / `sft_max_inst_frac`
gate above rather than a per-volume data change.

---

## File names on disk (per dataset)

Every downloader writes the `<stem>_volume.h5` (+ `<stem>_segmentation.h5` when
labelled) convention above. The stems below are what the scripts emit and what
you list (without the `_volume` / `_segmentation` suffix) under `vol:` / `seg:`
in a config. `x{X}_y{Y}_z{Z}` is the crop origin in voxels; image-only crops
(SSL / CREMI test) have **no** `_segmentation.h5`.

| Dataset | `<stem>` pattern | concrete example | seg? | root |
| --- | --- | --- | --- | --- |
| **SNEMI3D** | `AC4_inputs` / `AC4_labels`; `AC3_inputs` (test, EM only) | `AC4_inputs`, `AC4_labels` | A: yes / AC3: no | `data/SNEMI3D` |
| **Neurons** | `neurons_{X}x{Y}x{Z}_x{X0}_y{Y0}_z{Z0}` | `neurons_5000x2900x300_x3000_y7200_z950` | yes | `data/SNEMI3D` |
| **MICrONS** | `minnie65_mip0_4096x4096x800_x{X}_y{Y}_z{Z}` ; seg adds `_v{ver}` | vol `minnie65_mip0_4096x4096x800_x50000_y60000_z16000_volume`, seg `…_v1300_segmentation` | yes | `data/MICRONS` |
| **CREMI3D** | train `cremi3d_sample_{A,B,C}` ; test (EM only) `cremi3d_sample_{A+,B+,C+}` | `cremi3d_sample_A_volume`, `cremi3d_sample_A_segmentation` | A/B/C: yes / +: no | `data/CREMI3D` |
| **FLYEM3D** · FIB-25 SFT core | `flyem3d_8nm_x{X}_y{Y}_z{Z}` | `flyem3d_8nm_x2304_y2048_z6144` | yes | `data/FLYEM3D` |
| **FLYEM3D** · FIB-25 SSL surround | `flyem3d_8nm_ssl_x{X}_y{Y}_z{Z}` | `flyem3d_8nm_ssl_x4000_y4000_z2000_volume` | no | `data/FLYEM3D` |
| **FLYEM3D** · FIB-25 z-stride variants | `flyem3d_z32xy8nm[_xz/_yz][_p0..p3]_x{X}_y{Y}_z{Z}` | `flyem3d_z32xy8nm_p0_x2304_y2048_z6144` | yes | `data/FLYEM3D` |
| **FLYEM3D** · Hemibrain | `flyem3d_hemibrain_8nm[_ssl]_x{X}_y{Y}_z{Z}` | `flyem3d_hemibrain_8nm_ssl_x12000_y12000_z12000_volume` | sft: yes / ssl: no | `data/FLYEM3D` |
| **FLYEM3D** · MaleCNS | `flyem3d_malecns_8nm[_ssl]_x{X}_y{Y}_z{Z}` | `flyem3d_malecns_8nm_ssl_x20000_y20000_z20000_volume` | sft: yes / ssl: no | `data/FLYEM3D` |
| **COSEM3D** | `{jrc_id}_{rx}x{ry}x{rz}nm_x{X}_y{Y}_z{Z}` (image only; `--resample-isotropic` → `{jrc_id}_4nm_…`) | `jrc_hela-3_4x4x3.24nm_x0_y0_z0_volume` | no | `data/COSEM3D` |
| **MitoEM2** | `mitoem2_{subset}_{train,test}{NN}_volume` (image only; converted from nnU-Net `.nii.gz`) | `mitoem2_mossy_train01_volume` / `mitoem2_pyra_test01_volume` | no | `data/MitoEM2` |

Notes on the stem fields:
- **MICrONS** seg encodes the seg version: image `…_volume.h5`, seg
  `…_v1300_segmentation.h5` (so `vol:` and `seg:` differ by `_v{ver}`).
- **FLYEM3D** stems: `8nm` = native isotropic, `z32xy8nm` = z-strided 32 nm
  (FIB-25 only); `_ssl` = image-only (no seg); the `_xz` / `_yz` / `_p{n}`
  suffixes are FIB-25 orientation / z-phase augmentation variants. Hemibrain /
  MaleCNS carry the dataset name in the stem so they never collide with FIB-25.
- **MitoEM2** ships as nnU-Net datasets (`Dataset0NN_ME2-*/imagesTr/*.nii.gz`);
  the images are converted to the standard `mitoem2_{subset}_train{NN}_volume.h5`
  (key `main`, axes transposed `X,Y,Z` → `Z,Y,X`) and used **image-only in the
  `ssl` branch**.  The folder's own split is honoured: `imagesTr` → ssl **train**,
  `imagesTs` → ssl **validation** holdout (`task: ssl` recon).  As of the Jul
  2026 refresh (see below) the joint configs carry **30 train / 10 val**
  volumes; see `doc/data.csv` for the exact per-subset counts.  Two native
  resolutions: `[16,16,16]` (Beta/Jurkat/Macro/Podo/Sperm) and `[30,8,8]`
  (Mossy/Pyra/Stem).  Labels (mito/boundary) are unused.
  **`jurkat_train02` / `macro_train02` REMOVED** (no longer in the config):
  audited as **pure uniform random noise** (min=0, max=255, all 256 values
  present, std≈73.9 ≈ the theoretical max for uniform `[0,255]` data) -- a
  genuine, **checksum-confirmed** bug in the upstream MitoEM 2.0 release, not
  a conversion artifact -- see
  [MitoEM2 refresh from Zenodo v6](#mitoem2-refresh-from-zenodo-v6-jul-2026).
  MitoEM2 has **no CloudVolume source** to re-fetch a replacement crop from
  (see [SSL crop cleaning](#ssl-crop-cleaning-content--structure-gates)), so
  these two are simply dropped -- `jurkat` and `macro` each contribute their
  other, genuinely good `train01` volume instead (see below).  They previously
  passed the old `ssl_min_std`/foreground gates trivially (random noise has
  maximal variance) but fail `ssl_min_autocorr` (~0.0 vs ~0.5-0.9 for every
  other MitoEM2 volume).

### MitoEM2 refresh from Zenodo v6 (Jul 2026)

The original MitoEM2 data on disk was replaced end-to-end with a fresh
download of **MitoEM 2.0 v6** (Liu, P., published May 27, 2026,
[10.5281/zenodo.20417683](https://zenodo.org/records/20417683), CC-BY-4.0),
to check whether the `jurkat_train02` / `macro_train02` noise corruption
(discovered in the earlier local copy) was a stale/local issue or an upstream
one, and to pick up any new volumes the v6 release added.

**Procedure:** downloaded all 8 `Dataset0NN_ME2-*.zip` archives (~17.5 GB) plus
the release's own `checksums.txt` / `metadata.csv`, verified every extracted
file's SHA256 against `checksums.txt`, converted with `convert_mitoem2.py`,
then re-ran the full content/autocorr/nz audit (see
[SSL crop cleaning](#ssl-crop-cleaning-content--structure-gates)) on all 45
converted volumes before touching the config. Old data (48 GB, zips +
extracted + converted) was deleted only after the new data passed on disk (16
GB after dropping the now-redundant zips / extracted `.nii.gz`).

**Findings:**

1. **`jurkat_train02` / `macro_train02` noise is upstream, not local, and
   persists in v6.** SHA256 verification against the release's own
   `checksums.txt` found these are the **only 2 of ~90 files in the entire
   release that fail** -- i.e. Zenodo's own checksum manifest doesn't match
   what's actually packaged for these two files. Direct inspection confirmed
   the packaged data is pure uniform random noise (min=0, max=255, all 256
   values present, std≈73.9), identical in character to the old copy. This is
   a genuine, reproducible bug in the officially published dataset.
2. **A new upstream metadata bug**: `Dataset001_ME2-Beta`'s NIfTI headers ship
   voxel spacing `(1.0, 1.0, 1.0)` instead of the documented 16 nm isotropic
   (`metadata.csv` says `16×16×16`; Jurkat/Macro/Podo/Sperm/Mossy/Pyra headers
   are all correct). `convert_mitoem2.py` reads `native_resolution` straight
   from the NIfTI header zooms, so naively trusting it would silently mis-tag
   every beta volume as 1 nm voxels -- badly distorting the resolution-ladder
   pooling. **The config overrides beta's `native_resolution` to `[16, 16,
   16]` manually** (from `metadata.csv`) rather than trusting the (buggy)
   auto-derived value.
3. **`jurkat_train01` / `macro_train01` are good and were restored.** An older
   doc note said these were "removed as empty" (in whatever earlier release
   version that referred to); freshly audited against v6 they are genuine,
   well-structured EM (autocorr 0.79 / 0.59) -- so both are back in the
   config, meaning `jurkat` / `macro` each now contribute 1 real train crop
   again (down from the illusion of "2" when one was actually noise).
4. **v6 ships additional test crops** not present in the old copy:
   `beta_test01` / `beta_test03` (previously only `test02` existed),
   `jurkat_test01`, `macro_test01` -- all audited good and added to
   `val_volumes`.
5. **`stem_train01` / `stem_train02` / `stem_test01` persist as low-structure**
   in v6 (same autocorr ~0.37-0.49 as before) -- still excluded; see the
   "excluded crops" table above.

Net effect on the config across this whole cleanup pass: MitoEM2 SSL
**train 32 → 30** (`-2` corrupt `jurkat_train02`/`macro_train02`, `-2`
low-structure `stem_train01`/`train02`, `+2` restored `jurkat_train01`/
`macro_train01`; wash on `beta`/`podo`/`sperm`/`mossy`/`pyra`), **val 7 → 10**
(`-1` low-structure `stem_test01`, `+4` new v6 test crops
`beta_test01`/`test03`, `jurkat_test01`, `macro_test01`).
- **COSEM3D** keeps the upstream `jrc_*` id and its exact (near-cubic) voxel in
  the stem; verify the printed entry, since the per-volume z (3.24 / 3.36 /
  3.44 nm) lands in the `res` tag.

Each downloader **prints the exact `vol:` / `seg:` / `root:` block to paste**
into the config after a successful crop (the coords are only known after the
clamp), and `scripts/download_all.py --verify-only` lists every `.h5` actually
on disk with its shape / resolution / dtype / seg-fg%.

---

## How datasets enter a training run

1. **Download** with the script(s) above into the matching `data/<root>`.
2. **List volumes** in the config under `data.train_volumes` /
   `val_volumes` / `test_volumes` as `{vol, seg, root}` triples (the
   combine recipe mixes roots in one run).
3. **Per-dataset native resolution** goes in `data.resolution_map`
   (`(z, y, x)` nm), keyed by a **prefix of the volume name** (e.g.
   `flyem3d`, `cremi3d`, `minnie65`, `neurons`, `AC`). It is consumed
   only by the `resolution_zoom` augmentation.
4. **Resolution policy** (`cosmos3nano3d.yaml` / `cosmospredict3d.yaml`):
   `resolution_zoom_mode: union` with `resolution_zoom_prob` **0.9** (nano) /
   **0.5** (predict) — the fraction of training patches that are jittered
   (the rest are fed at native scale). Each anisotropic patch is resampled to
   a **random target inside the shared union envelope** `z ∈ [30,40]`,
   `xy ∈ [4,8]` nm (the union of all native resolutions); the `z` and `xy`
   targets are sampled **independently** (log-uniform), so every dataset's
   *output* resolution lands in that envelope while the per-dataset *zoom*
   (`= native / target`) differs. Because the affinity offsets are defined in
   **voxels**, harmonising onto a common space gives them a *consistent
   physical meaning across datasets*, and the random target doubles as
   scale/anisotropy augmentation. Train-only (validation is always native).

   Per-dataset behaviour (verified empirically against the config):

   | dataset (`resolution_map` key) | native z,y,x | zoom z | zoom xy | output (z / xy) |
   | --- | --- | --- | --- | --- |
   | `AC` / `neurons` | 30,6,6 | 0.75–1.00 | 0.75–1.50 | 30–40 / 4–8 |
   | `minnie65` | 40,8,8 | 1.00–1.33 (up) | 1.00–2.00 (up) | 30–40 / 4–8 |
   | `cremi3d` | 40,4,4 | 1.00–1.33 (up) | 0.50–1.00 (down ≤2×) | 30–40 / 4–8 |
   | `flyem3d` (native) | 8,8,8 | — skipped (isotropic) — | — | native 8³ |
   | `flyem3d_z32` (12 variants) | 32,8,8 | 0.80–1.07 | 1.00–2.00 (up) | 30–40 / 4–8 |

   (zoom > 1 = upsample/finer; < 1 = downsample/coarser. Anisotropy ratio
   `z:xy` of the output spans ~3.75:1 to 10:1 since z and xy are independent.)
   - **Isotropic volumes (FIB-25 `8×8×8`) are skipped** by the union
     resample (`z==y==x` in `resolution_map`): upsampling their fine 8 nm z
     to the 30–40 nm envelope would be a ~5× downsample, blowing up the
     pre-zoom safe-crop and destroying their isotropy. FIB's anisotropic
     contribution comes from the **z-strided `[32,8,8]` copy** (key
     `flyem3d_z32`, `--z-stride 4`, all phase offsets), which participates
     normally. The isotropic FIB variants
     are augmented by their octahedral orientation copies instead (see the
     FLYEM3D section / `download_flyem3d.py --orientations`).
   - Safe-crop (pre-zoom read) is bounded at ≈ `(107, 512, 512)`; the 512
     is driven by CREMI's 4 nm → 8 nm 2× downsample.
   - Legacy `resolution_zoom_mode: ratio` (anisotropy-preserving, single
     scale factor) is still available.

For the recipe to add a brand-new dataset (preprocessor → leaf dataset →
leaf datamodule → YAML), see [`CONTRIBUTING.md`](./CONTRIBUTING.md).
