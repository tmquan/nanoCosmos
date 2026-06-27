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
| MitoEM2 | Mitochondria EM, **mixed FIB-SEM / ssSEM / SBF-SEM** (8 subsets) | 16 × 16 × 16 & 8 × 8 × 30 | *(joint3d SSL, per-vol)* | Liu et al. 2025, *bioRxiv* (MitoEM 2.0); orig. Wei et al. 2020, *MICCAI* | `convert_mitoem2.py` | `data/MitoEM2` | `joint3d` |

`COSEM3D` (4 nm), `MitoEM2` (8–16 nm), and the Hemibrain / MaleCNS members of
`FLYEM3D` (8 nm) are the image-only `ssl` rungs of the **joint super-resolution
recipe** (`configs/nanocosmos-16B.yaml` / `nanocosmos-2B.yaml`,
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
  never publicly released).
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
    withholds the labels) → converted **image-only**.  They are **not**
    listed in any config's `test_volumes` (no GT = no metrics); run **blind
    inference** on them separately.
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
  y[1664:4288] z[1472:8000]` (≈ 25 × 21 × 52 µm) and is ~55% filled.
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
  `ssl` branch**.  The folder's own split is honoured: `imagesTr` → ssl **train**
  (34 vols), `imagesTs` → ssl **validation** holdout (11 vols, `task: ssl` recon).
  Two native resolutions: `[16,16,16]` (Beta/Jurkat/Macro/Podo/Sperm) and
  `[30,8,8]` (Mossy/Pyra/Stem).  Labels (mito/boundary) are unused.
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
