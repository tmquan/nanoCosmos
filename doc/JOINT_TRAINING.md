# Joint reconstruction + segmentation — loss mechanics

This is the mechanics of the nanoCosmos training loss. For the governing
framing (the **resolution ladder**, the dataset census, role assignment, and
the curriculum) read [`RESOLUTION_LADDER.md`](./RESOLUTION_LADDER.md) first;
this doc only details how `Joint3DReconSegLoss` and the batch contract work.

**Voxel-size convention** (used throughout): *small voxel size* = fine /
high-detail (e.g. 4 nm); *large voxel size* = coarse (e.g. 30–40 nm). The
network predicts on a fixed **small-voxel grid** (the finest voxel size, 4 nm;
`nanocosmos-16B.yaml` uses a `[200, 256, 256]` patch, the `2B`/`4B` configs a
z-heavier `[400, 256, 256]`) and every loss term pools that prediction **down**
to wherever its ground truth lives.

---

## 1. The single head

One backbone, one head, on the fixed small-voxel grid (`D = H = W`):

```
head : [B, N_AFF + 2, D, H, W]
        └ aff  [0 : N_AFF]    per-offset affinity   (logits)   ─┐ small-voxel
        └ sem  [N_AFF]        foreground / boundary (logit)    ─┘ segmentation
        └ raw  [N_AFF + 1]    reconstructed small-voxel EM (linear)  ← reconstruction
```

Same channel layout as the plain affinity recipe (`nanocosmos.losses._common`),
with one reinterpretation: `raw` is the **reconstructed small-voxel EM** (the
generative, super-resolution output), not the input reconstruction. No
wrapper/decoder change is needed — the existing `Cosmos3Nano3DWrapper` already
emits `[B, N_AFF+2, D, H, W]` at the input grid, and the input grid is the
small-voxel cube.

---

## 2. Two branches (routed per task-homogeneous batch)

A round-robin multi-task sampler yields one branch per step. The roles
**stack** by dataset (a labeled small-voxel rung like FIB-25 is in *both*).

### `ssl` — self-supervised reconstruction (gated by *voxel size*)
```
small-voxel x ──RandResolutionDegraded──► large-voxel input ──backbone──► raw = x̂
                                                                            │
                                                    L1(pool(raw→grid), x)  ◄─┘   (no labels)
```
Sources: COSEM 4 nm + unlabeled FlyEM 8 nm (**FIB-25 / Hemibrain / MaleCNS**
neuropil) + CREMI A+/B+/C+ (padded, image-only) + SNEMI3D **AC3** (label-less
test half) + the full MitoEM2 8–16 nm ladder — all domain-matched EM. The
degradation (`nanocosmos/transforms/degrade.py::RandResolutionDegraded`)
composes z slab-integration + decimation, per-section `grid_sample`
misalignment, missing/duplicated sections, and section noise/contrast — not
clean z-decimation alone — so the reconstruction transfers to real ssTEM/SBEM
(a sim-to-real contribution).

### `sft` — segmentation (gated by *labels*)
```
native x ──resample→ small-voxel grid ──backbone──► head
                                                     │ adaptive_avg_pool3d → labels.shape[-3:]
                                                     ▼
                                          AffinityFGLoss vs native labels
                          + raw data-consistency:  pool(raw) → native EM, L1   (if recon_image given)
```
The head is pooled to the **label grid** (factor derived from `labels.shape`,
so factor 1 when the labels are already small-voxel). **Plus** a
data-consistency term on the always-present `raw` head: pool the predicted
small-voxel `raw` down to the **original large-voxel EM** and match it — the
super-resolved reconstruction must agree with what was actually measured when
downsampled ("do no harm"), and this keeps the raw head trained on sft.

### Why pool with `adaptive_avg_pool3d`
A large voxel **integrates** the small sub-voxels it spans — an *average*, not
a point sample. `adaptive_avg_pool3d` averages over each (possibly fractional)
window on **all three axes** (so larger-voxel-xy rungs like MICrONS pool
in-plane too), with no aliasing — unlike a single `grid_sample` tap. It accepts
any output shape, so an arbitrary (non-integer, dataset-dependent) factor needs
no kernel bookkeeping. (`grid_sample` *is* the right tool on the **degradation**
side, for sub-pixel section misalignment.)

---

## 3. Joint training

All branches **backprop jointly** into the shared backbone (no detach): the
segmentation gradient shapes the reconstruction and vice-versa.

```
total = weight_ssl · ( weight_rec · L1_recon )                                  # ssl
      + weight_sft · ( weight_seg · (aff + sem) [+ weight_rec · L1_dataconsist]) # sft
```

`weight_ssl` / `weight_sft` are applied **per task-homogeneous batch** as an
outer multiplier on that branch's subtotal (the shipped joint configs use
`weight_ssl: 10.`, `weight_sft: 1.0`, so an ssl step carries 10× the gradient
scale of an sft step). This is independent of how *often* each branch is drawn
(that is the sampler's job, below).

### 3.1 Sampling & balancing (`balance`, `sample_weight`, `subset_weights`)

A custom round-robin batch sampler draws every batch from a **single group** so
each batch is task- and shape-homogeneous (required by the loss/VAE). The
`data.balance` knob chooses what a group is:

- `resolution` (code default): one group per `(task, native_resolution)`. Within
  a group, volumes are picked **weighted by voxel count** (big volumes dominate).
- `volume` (**shipped joint configs**): one group per volume, every volume
  scheduled **equally**; each batch is drawn from a single volume.
- `subset`: one group per domain (e.g. `cremi3d`, `minnie65`, `mitoem2_pyra`),
  every subset scheduled equally.

A group's schedule share is `num_samples · branch.sample_weight · multiplier`,
where `multiplier` is the per-volume `sample_weight` (volume balance) or
`subset_weights[subset]` (subset balance), **normalised by the branch mean** so
weights are *relative ratios* and the virtual-epoch length stays fixed. Bump a
volume/subset's weight to deliberately over-sample it (e.g. overfit CREMI/SNEMI
for a segmentation sanity check). `trainer.limit_train_batches` caps the actual
epoch length, so adding groups does not lengthen epochs.

---

## 4. Batch contract (what the datamodule provides)

`Joint3DReconSegLoss.forward(head, targets)` consumes, per (homogeneous) batch:

| key | branches | meaning |
| --- | --- | --- |
| `task` | all | `"ssl"` or `"sft"` (one per batch) |
| `recon_image` | ssl (req), sft (opt) | clean EM recon target on its own grid; `raw` is pooled to match. ssl: clean small-voxel EM (main SR). sft: the **original large-voxel EM** → `raw` data-consistency. |
| `labels` | sft | instance ids at the **native** grid; the head is pooled to `labels.shape[-3:]` |
| `sem_label` | sft (opt) | boundary-eroded foreground (same native grid as `labels`) that supervises the **sem head only** — see §4.1 |
| `_cached_targets` | sft | from `loss.build_targets(labels)` (precomputed affinity target) |

- **SSL** sample (COSEM 4 nm, unlabeled FlyEM 8 nm, or MitoEM2 8–16 nm — all
  image-only) → `RandResolutionDegraded` writes the degraded `image` + pristine
  `recon_image`; no `labels`.
- **SFT** sample (any labeled rung) → `image` resampled to the small-voxel cube,
  `labels` kept on the **native** grid; the loss pools the prediction to the
  label grid (factor 1 when native == small-voxel). Optionally pass the
  original native EM as `recon_image` for the raw data-consistency term.

---

## 4.1 Boundary-aware sem supervision (`find_boundaries` / `boundary_target`)

Without help, the `sem` head's target (`labels > 0`) is almost all-foreground
after `min_foreground` cropping — a near-degenerate "predict 1 everywhere"
objective. Two `data` knobs fix this on the **sft** branch:

- `find_boundaries` — per-sample probability of eroding membrane voxels (the
  shipped configs use `1.0` = always).
- `boundary_target` —
  - `semantic` (joint default): the datamodule copies `label → sem_label` and
    erodes the membranes **only** in `sem_label`. The sem head learns thin
    membrane gaps; the instance `label` stays **pristine** so the affinity
    target is unaffected.
  - `both`: erode the shared `label` (affects sem **and** affinity).

Erosion runs on the **native** label grid with the group's native resolution,
so the anisotropy guard (xy-only when z is ≥2× coarser) applies per dataset.
The sem loss, the val sem metric, and the `true/sem` TensorBoard panel all use
`sem_label` when present, falling back to `labels` otherwise.

**No-full-erase guard.** Erosion only opens gaps *between* touching instances —
it must never delete an instance outright. On near-isotropic volumes (FIB-25
8 nm, where the xy-only guard does *not* trigger) a structure thin along z would
otherwise have every voxel flagged as boundary and vanish from `sem_label`,
turning real tissue into false background (an all-black `true/sem` region).
`FindBoundariesd` therefore restores any connected component the boundary pass
would remove completely.

## 4.2 SSL crop gates (`ssl_min_foreground`, `ssl_min_std`)

The label-less `ssl` branch can sample crops that are uninformative for
reconstruction in two distinct ways, each with its own gate (both applied only
to label-less volumes in `LazyVolDataset`; failing crops are re-sampled, and the
best-seen crop is kept once `max_foreground_retries` is exhausted):

- `ssl_min_foreground` — **legacy non-zero gate** used only when `ssl_min_std`
  is `0`. Rejects literally zero-padded / black crops (non-zero voxel fraction
  below the threshold).
- `ssl_min_std` (shipped `0.05`, the active gate) — **local content gate**.
  A global non-zero (or even global-std) test passes a crop that is mostly flat
  resin/embedding medium as long as *some* region is textured. Instead, the crop
  is normalised to `[0, 1]` (per-volume), tiled into `4×16×16` blocks, and each
  block is marked *content* when its local std `>=` `ssl_min_std`; the crop must
  then have a **content fraction `>=` `ssl_min_foreground`** (so the two knobs
  compose: `ssl_min_std` sets the texture threshold, `ssl_min_foreground` the
  required fraction). This rejects flat / half-empty crops (e.g. a COSEM cell
  edge against resin) that the non-zero gate misses. Best-seen ranks by content
  fraction.

For the **sft** branch, `sft_min_foreground` (shipped `0.8`) is a **dual** gate:
a crop must have BOTH its label-foreground fraction AND its image non-zero
fraction `>=` the threshold. The label half rejects background-heavy / unlabeled
crops (e.g. sparse MICrONS regions that render as half-black `true/label` /
`true/sem` panels); the image half rejects zero-padded EM. As with ssl, the
best-seen crop is kept once `max_foreground_retries` is exhausted, so genuinely
sparse volumes degrade gracefully rather than looping. (The older label-only
`min_foreground` still works and is used as the fallback when
`sft_min_foreground` is unset.)

---

## 5. Status

The whole joint recipe is implemented and unit-tested (`tests/test_joint*.py`):

- `nanocosmos/losses/joint3d.py` — `Joint3DReconSegLoss` (`ssl`/`sft` routing,
  `_pool_to` all-axis pool to the GT grid, recon on ssl + raw
  data-consistency on sft, joint backprop, channel-layout delegation,
  deterministic `canonical_loss_keys`).
- `nanocosmos/transforms/degrade.py` — `RandResolutionDegraded`;
  `nanocosmos/transforms/fine_grid.py` — `ToFineGridd` (resample to the
  small-voxel grid, keep native labels / recon target).
- `nanocosmos/datamodules/joint3d.py` — `Joint3DDataModule`: per-branch volume
  lists, round-robin task-homogeneous batches (`_RoundRobinBatchSampler`).
- `nanocosmos/modules/joint3d.py` — `Joint3DModule` (Cosmos-3 Nano backbone)
  and `JointPredict3DModule` (Cosmos-Predict 2.5 2B backbone); both route to
  `Joint3DReconSegLoss` and handle the label-free `ssl` step + eval.
- `nanocosmos/callbacks/tensorboard/joint3d_logger.py` — `Joint3DImageLogger`
  logs **both** `ssl` and `sft` panels each epoch (task-namespaced tags).
- `train.py` selects the module by `model.type` (`joint3d` / `joint3d_2b`); see
  `configs/nanocosmos-16B.yaml` and `configs/nanocosmos-2B.yaml`.
- Data: `scripts/download_cosem3d.py` (4 nm COSEM3D), `scripts/download_flyem3d.py`
  (FlyEM 8 nm: fib25 / hemibrain / malecns), `scripts/convert_mitoem2.py`
  (MitoEM2 nnU-Net `.nii.gz` → image-only h5).
