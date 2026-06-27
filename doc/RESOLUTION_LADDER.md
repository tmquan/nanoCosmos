# The resolution ladder (nanoCosmos as a super-resolution foundation model)

**The organizing axis is *voxel size*, not voxel shape (upsampling).**
nanoCosmos is a single backbone that **super-resolves EM toward a common
small-voxel grid** and segments there. (Convention: *small voxel size* =
fine, e.g. 4 nm; *large voxel size* = coarse, e.g. 30–40 nm.) Every dataset is
placed on a **resolution ladder** by its voxel size and assigned a role by
where it sits:

- the **finest** data anchors **SSL** (self-supervised: degrade → reconstruct);
- the **middle** of the ladder does **all three** — it is upsampled (SR
  forward), self-supervised (SSL), *and* downsampled (to match coarser labels);
- the **coarsest** labeled data is **SFT segmentation** — the small-voxel
  prediction is pooled back to native for the loss.

**Role rule (the roles stack, they are not exclusive):**
- **SSL** is gated by *voxel size* — any rung fine enough can self-supervise
  (degrade → reconstruct), with or without labels.
- **SFT** is gated by *labels* — **any rung that has neuron-instance labels
  does SFT too**, regardless of where it sits on the ladder (predict at the
  small-voxel grid, pool to its native grid, supervise).

So a labeled rung (FIB-25, SNEMI3D) is **both** a SSL source **and** an SFT
source. COSEM is SSL-only — not because of its position, but because its
labels are organelles, not neuron instances.

---

## 1. What the datasets are

| dataset | what it is | imaging | voxel (z, y, x) nm | labels |
| --- | --- | --- | --- | --- |
| **COSEM / OpenOrganelle** (`jrc_*`) | cell-biology volumes (HeLa, Jurkat, macrophage, …) | FIB-SEM | ~4 near-cubic (z **3.24–5.24**) | organelle classes (not neurons) |
| **FIB-25** | FlyEM *Drosophila* medulla (7-column) | FIB-SEM | **8 × 8 × 8** | a dense proofread **core** (~1536³, ~99.7 % fg); the **rest of the 6446×6643×8090 volume is UNSEGMENTED** |
| **Hemibrain** | FlyEM *Drosophila* **central brain** (~25 k neurons) | FIB-SEM (hot-knife slabs) | **8 × 8 × 8** (verified from its `info`) | proofread neuron instances (`gs://neuroglancer-janelia-flyem-hemibrain`) |
| **MaleCNS** | FlyEM **full male *Drosophila* CNS** (central brain + optic lobes + ventral nerve cord); v1.0, 2026, CC-BY | FIB-SEM | **8 × 8 × 8** | proofread neuron instances (`gs://flyem-male-cns`); the largest labeled neuropil EM |
| **SNEMI3D / Neurons** | mouse cortex (Kasthuri) | ssTEM | 30 × 6 × 6 | neuron instances |
| **CREMI** | *Drosophila* brain | ssTEM | 40 × 4 × 4 | neuron instances |
| **MICrONS** (minnie65) | mouse visual cortex | ssTEM (aligned) | 40 × 8 × 8 | neuron instances |

> **Hemibrain** and **MaleCNS** are Janelia FlyEM connectomics releases — large,
> proofread, **8 nm isotropic** neuropil FIB-SEM. They are domain-matched
> (neuropil) SSL sources *and*, because they carry neuron labels, SFT sources.
> They are the scalable 8 nm counterpart to FIB-25.

---

## 2. Dataset census (sorted finest → coarsest by z)

Voxel sizes `(z, y, x)` in nm, **exact**. Fine grid (SR target) = **4 nm cubic**
(see §2.1). `up (z·xy) = native / 4` per axis is the factor the model upsamples
to reach the grid; it is also the **pool factor** used to bring the prediction
back to native for that rung's loss (`< 1` ⇒ that axis is mildly *downsampled*).

`AVAIL_SIZE` is the **real full volume** (`x×y×z` vox, verified from each
`info`). `USED_SIZE` is the **selective, representative portion we actually
fetch** — we do **not** download whole volumes (Hemibrain/MaleCNS are
peta-voxel; even FIB-25 is ~3.5e11 vox). Especially for SSL a handful of
crops suffices. `Nx A×B×C` = N crops of that size. **USED_SIZE values below are
a starting proposal — tune per compute/coverage.**

| rank | volume | (z,y,x) nm | up z·xy | role | AVAIL_SIZE (x×y×z vox) | USED_SIZE (x×y×z vox) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **COSEM `jrc_hela-3`** | (**3.24**,4,4) | 0.81·1 | SSL (anchor) | 12400×1000×12000 | 5× 2048×2048×2048 |
| 2 | **COSEM `jrc_macrophage-2`** | (**3.36**,4,4) | 0.84·1 | SSL (anchor) | 10000×2000×11087 | 5× 2048×2048×2048 |
| 3 | **COSEM `jrc_jurkat-1`** | (**3.44**,4,4) | 0.86·1 | SSL (anchor) | 10000×3000×8560 | 5× 2048×2048×2048 |
| 4 | **FIB-25** | (8,8,8) | 2·2 | SSL + SFT | 6446×6643×8090 | SFT 1× 1536³ (core); SSL 4× 1024³ (surround) |
| 5 | **Hemibrain** | (8,8,8) | 2·2 | SSL (+SFT) | 34427×39725×41394 | 5× 1024³ (4 SSL + 1 test) |
| 6 | **MaleCNS** | (8,8,8) | 2·2 | SSL (+SFT) | 94088×78317×134576 | 5× 1024³ (4 SSL + 1 test) |
| 7 | **SNEMI3D / Neurons** | (30,6,6) | 7.5·1.5 | SFT | 5000×2900×300 (Kasthuri); AC4 1024×1024×100 | full (train vol + AC4 val) |
| 8 | **CREMI** | (40,4,4) | 10·1 | SFT | 3× 1250×1250×125 (A/B/C) | full 3× 1250×1250×125 |
| 9 | **MICrONS** | (40,8,8) | 10·2 | SFT | petascale (full minnie65) | 1–10× 4096×4096×800 crops |

Notes:
- Sizes are `x×y×z` (Neuroglancer `info` order); the voxel column stays `(z,y,x)`.
- COSEM SSL budget = **5× 2048×2048×2048 per volume**, BUT COSEM volumes are
  thin in **y** (1000–3000 vox), so the y extent is **capped to the volume**:
  effective crops are ≈ 2048 × min(2048, y_avail) × 2048 — i.e. `jrc_hela-3`
  ≈ 2048×1000×2048, `jrc_macrophage-2` ≈ 2048×2000×2048, `jrc_jurkat-1`
  2048×2048×2048 (full 2k³ fits, y=3000). The 5 crops tile x·z.
- Labels: COSEM = organelle (SSL-only); FIB-25 = proofread **core** only (the
  surround is unlabeled → SSL); the rest carry neuron instances.
- Hemibrain / MaleCNS take **5× 1024³** each, with **1 of the 5 held out for
  test** (4 train + 1 eval). At 8 nm a 1024³ crop is ≈ 8.2 µm — the same
  physical extent as COSEM's 2048³ at 4 nm, so the rungs stay comparable.
- The COSEM rungs have `up z < 1`: their **z is finer than 4 nm**, so they are
  *mildly downsampled* onto the grid (≤ 1.24×) — they are the anchor that
  defines what 4 nm ultrastructure looks like, not an SR target.
- CREMI is the split case: xy already at the grid (4 nm, `up=1`) but z is the
  coarsest (40 nm, `up=10`) — almost pure z-super-resolution.
- The coarser a rung, the larger its `up` factor (and the more its
  segmentation leans on the prior learned from the finer rungs).

### 2.1 Fine grid: 4 nm cubic (why not 3.2 nm?)

The **smallest voxel dimension anywhere** is COSEM's z (**3.24 nm** on
`jrc_hela-3`; 3.36 / 3.44 on the others). So one could set the grid to ~3.2 nm
"the smallest voxel". We do **not**, and the choice is per-axis:

- **xy:** the finest *native xy* is 4 nm (COSEM **and** CREMI). A 3.2 nm grid
  would upsample **every** dataset's xy (4 → 3.2 = 1.25×, even COSEM/CREMI) —
  fabricating in-plane detail nobody measured — and inflate the voxel count
  ~1.25³ ≈ **1.95×** for the same field of view. Not worth it.
- **z:** COSEM's 3.24–3.44 nm z is only ≤ 24 % finer than 4 nm. Putting it on a
  4 nm grid is a ≤ 1.24× **downsample** — negligible next to the 2–10×
  *up*sampling every other rung needs, and COSEM still anchors the texture.

So: **4 nm cubic.** A `z = 3.2 nm` grid is not justified by a ≤ 0.76 nm gain on
one source at the cost of upsampling everyone's xy; COSEM's z is simply
resampled to 4 nm. That z→4 nm resample happens at train time on the 4 nm grid;
`download_cosem3d.py --resample-isotropic` can instead bake an exact 4 nm cubic
voxel into the files (**off by default**, since the train-time resample already
handles the ≤ 1.24× difference).

---

## 3. Role bands

### Band A — finest → **SSL** (label-free super-resolution prior)
Sources (image-only; `RandResolutionDegraded` synthesises the degraded→clean
pair):
- **COSEM 4 nm** (`jrc_hela-3`, `jrc_macrophage-2`, `jrc_jurkat-1`, …) — genuine
  4 nm small-voxel texture no 8 nm volume can give;
- **FIB-25 — the UNSEGMENTED surround** (and the core image): the dense
  proofread core is ~1536³, but the full FIB-25 volume is 6446×6643×8090, so the
  vast majority is **unlabeled 8 nm neuropil** — a large, free, domain-matched
  SSL source. Use the whole image volume image-only;
- **Hemibrain / MaleCNS images** — petascale 8 nm neuropil, image-only for SSL.

```
small-voxel x ──RandResolutionDegraded(zf, misalign, missing, noise)──► large-voxel input
       ──backbone──► x̂ (small-voxel)        L1(pool(x̂→grid), x)        (no labels)
```

This is the petascale-able pretraining: it teaches **fine ultrastructure +
z-continuity** with zero annotation. COSEM contributes 4 nm texture; FIB-25 /
Hemibrain / MaleCNS contribute domain-matched neuropil at 8 nm.

### Band B — middle → **upsample / SSL / downsample** (the bridge)
FIB-25 / Hemibrain / MaleCNS (8 nm, *labeled*) are the pivot rungs. Each plays
every role:
- **upsample**: an SR target/input toward the 4 nm grid;
- **SSL**: degrade → reconstruct (self-supervised; the unsegmented FIB-25
  surround makes this nearly free at scale);
- **downsample**: their 8 nm neuron labels supervise segmentation by pooling
  the small-voxel prediction back to 8 nm.

SNEMI3D (6 nm xy) lives here too — fine-ish xy, coarse z, labeled.

### Band C — coarsest → **SFT segmentation** (downsample the prediction)
CREMI, MICrONS (and FIB / SNEMI / Hemibrain / MaleCNS also contribute here):

```
native x ──resample→ 4 nm grid ──backbone──► head (small-voxel seg)
                                              │ adaptive_avg_pool3d → (d, h, w)_native
                                              ▼
                                    AffinityFGLoss vs native labels
```

The prediction is pooled to the rung's **native (z, y, x)** before the loss —
on all three axes, since coarser sets are also coarser in xy (MICrONS 8 nm,
`up_xy=2`).

---

## 4. How this maps onto the implemented code

The pieces in `nanocosmos/losses/joint3d.py` + `nanocosmos/transforms/degrade.py`
**already implement this ladder**:

| ladder concept | implementation |
| --- | --- |
| SSL (any fine-enough image) | `Joint3DReconSegLoss` task `ssl` + `RandResolutionDegraded` |
| SFT at native via pooling | task `sft` — pool prediction to the label grid |
| SFT where native == 4 nm grid | `sft` with pool factor 1 (the `up=1` special case) |
| pool factor | **derived from the GT shape** (`labels.shape[-3:]` / `recon_image.shape[-3:]`) |

The all-axis pool `Joint3DReconSegLoss._pool_to(x, (d, h, w))` via
`adaptive_avg_pool3d` handles every rung (z *and* xy), with the factor read
from the ground-truth shape (no `z_sections` key). Recon (`raw`) pools the same
way to the reconstruction target's grid. `tests/test_joint.py` covers both.

---

## 5. Plan of record

1. **Census + fine grid.** Adopt §2; fine grid = **4 nm cubic** (§2.1).
   (Revisit FOV/memory: 160³ @ 4 nm = 0.64 µm context — see §6.)
2. **Acquire SSL data.**
   - `scripts/download_cosem3d.py` — 4 nm COSEM3D cubes (`jrc_hela-3`,
     `jrc_macrophage-2`, `jrc_jurkat-1`).
   - **FIB-25 image, including the unsegmented surround** — `download_flyem3d.py`
     (image-only crops outside the proofread core), for 8 nm neuropil SSL.
   - optionally **Hemibrain / MaleCNS** image crops (8 nm), for scale.
3. **Acquire SFT data.** FIB-25 core + SNEMI3D + CREMI + MICrONS
   (+ Hemibrain / MaleCNS labels if desired).
4. **Config.** `configs/nanocosmos-16B.yaml`: `ssl` = COSEM + unsegmented
   FIB-25 (+ Hemibrain / MaleCNS); `sft` = every labeled rung, each carrying its
   native `(z, y, x)` so the pool factor is derived per volume.
5. **Curriculum.** Phase 1 SSL (label-free, 4 nm grid) → Phase 2 add SFT on the
   labeled rungs (predict 4 nm, pool to native), keep SSL live (joint).
6. **Integration layer.** Multi-task datamodule (resample each volume onto the
   4 nm grid, keep native labels, round-robin task-homogeneous batches) +
   `Joint3DModule` routing to `Joint3DReconSegLoss`.

---

## 6. Open knob (FOV vs resolution)

A 4 nm grid at 160³ sees only 0.64 µm — small context for large neurites.
Levers: bigger voxel patch (more memory), non-cubic voxel patch (more z-planes
since z is synthesized), or a slightly larger-voxel grid (e.g. 6 nm: 0.96 µm
FOV, only CREMI/COSEM mildly downsampled). Decide alongside the 16B memory
budget.
