# Joint reconstruction + segmentation — loss mechanics

This is the mechanics of the nanoCosmos training loss. For the governing
framing (the **resolution ladder**, the dataset census, role assignment, and
the curriculum) read [`RESOLUTION_LADDER.md`](./RESOLUTION_LADDER.md) first;
this doc only details how `Joint3DReconSegLoss` and the batch contract work.

**Voxel-size convention** (used throughout): *small voxel size* = fine /
high-detail (e.g. 4 nm); *large voxel size* = coarse (e.g. 30–40 nm). The
network predicts on a fixed **small-voxel grid** (the finest voxel size, e.g.
160³ at 4 nm) and every loss term pools that prediction **down** to wherever
its ground truth lives.

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

### `dapt` — self-supervised reconstruction (gated by *voxel size*)
```
small-voxel x ──RandResolutionDegraded──► large-voxel input ──backbone──► raw = x̂
                                                                            │
                                                    L1(pool(raw→grid), x)  ◄─┘   (no labels)
```
Sources: COSEM 4 nm + **unlabeled FIB-25** (domain-matched neuropil). The
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
total = w_dapt · ( w_recon · L1_recon )                              # dapt
      + w_sft  · ( w_seg · (aff + sem) [+ w_recon · L1_dataconsist]) # sft
```

---

## 4. Batch contract (what the datamodule provides)

`Joint3DReconSegLoss.forward(head, targets)` consumes, per (homogeneous) batch:

| key | branches | meaning |
| --- | --- | --- |
| `task` | all | `"dapt"` or `"sft"` (one per batch) |
| `recon_image` | dapt (req), sft (opt) | clean EM recon target on its own grid; `raw` is pooled to match. dapt: clean small-voxel EM (main SR). sft: the **original large-voxel EM** → `raw` data-consistency. |
| `labels` | sft | instance ids at the **native** grid; the head is pooled to `labels.shape[-3:]` |
| `_cached_targets` | sft | from `loss.build_targets(labels)` (precomputed affinity target) |

- **DAPT** sample (COSEM or **unlabeled FIB**) → `RandResolutionDegraded` writes
  the degraded `image` + pristine `recon_image`; no `labels`.
- **SFT** sample (any labeled rung) → `image` resampled to the small-voxel cube,
  `labels` kept on the **native** grid; the loss pools the prediction to the
  label grid (factor 1 when native == small-voxel). Optionally pass the
  original native EM as `recon_image` for the raw data-consistency term.

---

## 5. Status

Implemented and unit-tested (`tests/test_joint.py`, 14 tests):

- `nanocosmos/losses/joint.py` — `Joint3DReconSegLoss` (`dapt`/`sft` routing,
  `_pool_to` all-axis pool to the GT grid, recon on dapt + raw
  data-consistency on sft, joint backprop, channel-layout delegation,
  deterministic `canonical_loss_keys`).
- `nanocosmos/transforms/degrade.py` — `RandResolutionDegraded`.
- `scripts/download_cosem3d.py` — fetch 4 nm COSEM3D cubes for the DAPT branch;
  `scripts/download_flyem3d.py` — FlyEM 8 nm (fib25 / hemibrain / malecns).

Remaining: the multi-task datamodule (per-branch volume lists, resample to the
small-voxel grid + keep native labels, `RandResolutionDegraded` for `dapt`,
round-robin task-homogeneous batches) + the Lightning module (route to
`Joint3DReconSegLoss`, handle the label-free `dapt` step + eval) + `train.py`
wiring; see `configs/nanocosmos-16B.yaml`.
