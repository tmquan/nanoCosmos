# Nanocosmos — File Structure

What lives where.  For the *why* behind the layout (design patterns,
conventions, adding-new-X checklists), see
[`ORGANIZATION.md`](./ORGANIZATION.md).

---

## Top-level

```
nanocosmos/
├── LICENSE
├── README.md
├── pyproject.toml          # package metadata
├── requirements.txt        # pinned runtime dependencies
├── configs/                # Hydra YAMLs (§ Configuration)
├── data/                   # (gitignored) raw volumes
├── outputs/                # (gitignored) training artefacts
├── logs/                   # (gitignored) TensorBoard / wandb logs
├── doc/                    # this folder
├── scripts/                # CLI entrypoints (§ Scripts)
├── tests/                  # pytest suite (§ Tests)
└── nanocosmos/               # importable Python package (§ Package)
```

---

## Configuration

`configs/` is composed via Hydra's `defaults:` list.  The inheritance
chain is `default → <dataset> → <project>`.

| File                  | Purpose                                                                     |
| --------------------- | --------------------------------------------------------------------------- |
| `default.yaml`        | Every knob with a sensible default.  Base for every experiment.             |
| `snemi3d.yaml`        | SNEMI3D dataset overrides + shared **model/loss** hyperparameters (`cosmos3nano3d`). |
| `combine.yaml`        | Multi-dataset training (SNEMI3D + neurons + MICrONS).                       |
| `cosmospredict3d.yaml`| Flattened standalone Cosmos-Predict 2.5 (2B) baseline recipe.               |
| `cosmos3nano3d.yaml`  | Flattened standalone Cosmos3-Nano (16B) recipe.                             |

---

## Scripts

CLI entry points.  Run with `python -m nanocosmos.<module>` where
applicable or directly: `python scripts/<name>.py`.

| Script                         | Purpose                                           |
| ------------------------------ | ------------------------------------------------- |
| `scripts/train.py`             | Hydra-driven training loop (Lightning Trainer).   |
| `scripts/download_snemi3d.py`  | Fetch SNEMI3D volumes via cloudvolume.            |
| `scripts/download_microns.py`  | Fetch MICrONS volumes + segmentations.            |
| `scripts/download_zenodo_582636.py` | Generic Zenodo downloader, currently pointing at record 582636 (X-ray uCT of an assembly of rice grains, used as a 3D instance-segmentation benchmark with densely touching objects). |

---

## Tests

| File                               | Covers                                                     |
| ---------------------------------- | ---------------------------------------------------------- |
| `tests/test_losses.py`             | `AffinityFGLoss` (channel layout, raw-logit head, uint8 affinity target + validity mask, chunked-loss parity, scalar keys, backward) + `DiceBCEFocalLoss`. |
| `tests/test_tensorboard_heads.py`  | `_log_predictions` panel set (true/pred aff, sem, raw, Mutex Watershed `pred/label`). |
| `tests/test_datasets.py`           | `CircuitDataset` abstract contract (resolution, anisotropy, length virtualisation). |
| `tests/test_datamodules.py`        | `CircuitDataModule` augmentation pipeline (via a synthetic in-memory dataset). |
| `tests/test_preprocessors.py`      | HDF5 / NRRD / TIFF / NfTy converters.                      |
| `tests/test_utils.py`              | label / io / parallel helpers.                             |

Tests for the freeze schedule and `sliding_window_inference` are not
yet shipped (tracked under the
audit overhaul backlog -- see
[`doc/CONTRIBUTING.md`](./CONTRIBUTING.md) for where to land them).

---

## Package: `nanocosmos/`

Top-level `nanocosmos/__init__.py` re-exports the most common symbols
(wrappers, Lightning modules, losses, datamodules).

Subpackages, in the order a new contributor would typically explore:

### `nanocosmos/transforms/` — deterministic ops

Pure functions and MONAI `Transformd` wrappers used by the datamodules
and the loss targets.  No learnable state.

| File                       | Purpose                                                                |
| -------------------------- | ---------------------------------------------------------------------- |
| `edt.py`                   | GPU/CPU distance-transform + filter/label/centroid utilities (cucim / scipy). |
| `find_boundaries.py`       | Connectivity-1 inner/outer boundary masks (cucim / skimage / torch).   |
| `label.py`                 | Relabel / remap / consolidate instance ids.                            |
| `rand_crop_foreground.py`  | Random crop biased toward foreground voxels.                           |
| `rand_transpose_xy.py`     | Random xy-transpose augmentation.                                      |
| `resolution_zoom.py`       | Per-axis resolution scaling for multi-resolution training.             |

### `nanocosmos/datasets/` — MONAI `CacheDataset`s

| File            | Purpose                                                                         |
| --------------- | ------------------------------------------------------------------------------- |
| `base.py`       | `CircuitDataset` abstract base — declares `paper`, `resolution`, `labels`, etc. |
| `snemi3d.py`    | SNEMI3D dataset leaf.                                                           |
| `microns.py`    | MICrONS dataset leaf.                                                           |
| `neurons.py`    | Internal "neurons" volume leaf.                                                 |
| `lazy.py`       | `LazyVolDataset` — on-demand loading for very large volumes.                    |

### `nanocosmos/datamodules/` — Lightning `DataModule`s

| File            | Purpose                                                             |
| --------------- | ------------------------------------------------------------------- |
| `base.py`       | `CircuitDataModule` — MONAI augmentation pipeline + split logic.    |
| `snemi3d.py`    | SNEMI3D datamodule leaf.                                            |
| `microns.py`    | MICrONS datamodule leaf.                                            |
| `neurons.py`    | Internal neurons datamodule leaf.                                   |

### `nanocosmos/losses/` — affinity + sem + raw loss

| File            | Purpose                                                                        |
| --------------- | ------------------------------------------------------------------------------ |
| `_common.py`         | Single source of truth for the head layout: `AFFINITY_OFFSETS` / `N_PULL`, `AFF_SLICE` / `SEM_SLICE` / `RAW_SLICE`, `HEAD_CHANNELS`, the affinity-target / validity-mask builders, and slicing helpers. The head emits raw logits / linear values (no activation in `forward`). |
| `affinity.py`        | `AffinityFGLoss` — the head's supervisor: masked + offset-weighted (pull/push) affinity composite (BCE + soft-Dice + focal), a `DiceBCEFocalLoss` foreground (`sem`) term, and an L1 `raw` reconstruction term.  Emits `loss/aff`, `loss/sem`, `loss/raw`. |
| `dice_bce_focal.py`  | `DiceBCEFocalLoss` — composite logit-input supervisor used by `AffinityFGLoss` for the `sem` head.  Logit-stable BCE (`binary_cross_entropy_with_logits`) plus MONAI's `DiceLoss(sigmoid=False)` and a focal path, both on `sigmoid(logits)`; `lambda_{bce,dice,focal}` + `gamma` parameterise the mix. |

### `nanocosmos/metrics/` — per-head eval metrics

| File            | Purpose                                                             |
| --------------- | ------------------------------------------------------------------- |
| `semantic.py`   | Per-class IoU, Dice, pixel accuracy.                                |
| `instance.py`   | Adapted Rand Error, Variation of Information, optimal split/merge. |

### `nanocosmos/models/` — backbone wrappers

`models/base.py::BaseModel` is the abstract contract (forward →
`[B, HEAD_CHANNELS, *spatial]` raw-logit head tensor,
`get_output_channels()`).

#### `models/cosmos_2_5_common/` — shared scaffolding for the Cosmos 2.5 family

Both Cosmos-Transfer 2.5 and Cosmos-Predict 2.5 share the same
underlying base DiT, Wan VAE, feature-extraction hooks, decoder
adapter, freeze plumbing and HF auto-pull path.  Those live here so
each backbone-specific package only owns its true delta.

| File                  | Purpose                                                               |
| --------------------- | --------------------------------------------------------------------- |
| `__init__.py`         | Re-exports the shared symbols.                                        |
| `wrapper_base.py`     | `_BaseCosmos25Wrapper` — abstract wrapper with extension hooks.       |
| `variants.py`         | `_VariantConfigBase` dataclass (each backbone extends as needed).     |
| `layers.py`           | Shared primitives (`_NORM`, `_PointwiseLinear`, `_adapt_to_rgb`).     |
| `decoder.py`          | `_FeatureProjector3D` / `_DecoderAdapter3D` (VAE decoder + affinity + sem + raw head). |
| `standalone_dit.py`   | Random-init `_StandaloneDiT3D` fallback for `pretrained=False`.       |
| `hf_loader.py`        | Rank-aware HF snapshot download (ignores `text_encoder/*`).           |

#### `models/cosmos_transfer_2_5/` — Cosmos-Transfer 2.5 3-D wrapper

The Transfer-specific delta on top of `cosmos_2_5_common`: the
ControlNet residual branch (`CosmosControlNetModel`) loaded from a
sibling HF revision and summed into the base DiT every
`controlnet_block_every_n` blocks.

| File                  | Purpose                                                               |
| --------------------- | --------------------------------------------------------------------- |
| `__init__.py`         | Re-exports `CosmosTransfer3DWrapper`.                                 |
| `wrapper.py`          | `CosmosTransfer3DWrapper` — adds ControlNet load / freeze / forward.  |
| `variants.py`         | `_VariantConfig(_VariantConfigBase)` adding `hf_revision_controlnet`. |

#### `models/cosmos_predict_2_5/` — Cosmos-Predict 2.5 3-D wrapper

The base DiT in NVIDIA's Cosmos 2.5 stack (no ControlNet).  Inherits
everything from `_BaseCosmos25Wrapper` without overriding any
extension hooks.

| File                  | Purpose                                                               |
| --------------------- | --------------------------------------------------------------------- |
| `__init__.py`         | Re-exports `CosmosPredict3DWrapper`.                                  |
| `wrapper.py`          | `CosmosPredict3DWrapper` — thin subclass of `_BaseCosmos25Wrapper`.   |
| `variants.py`         | Predict-specific variant registry (`nvidia/Cosmos-Predict2.5-2B`).    |

#### `models/vista/` — Vista3D wrapper + head

| File                       | Purpose                                                             |
| -------------------------- | ------------------------------------------------------------------- |
| `__init__.py`              | Re-exports `Vista3DWrapper`, `VistaTaskHead3D`.                     |
| `wrapper.py`               | `Vista3DWrapper` — SegResNetDS2 backbone + the affinity + sem + raw head. |
| `heads.py`                 | `VistaTaskHead3D` (MONAI `UnetrBasicBlock`).                        |
| `hf_loader.py`             | MONAI `VISTA3D-HF` encoder download + partial-load.                 |

### `nanocosmos/modules/` — Lightning modules

`modules/base.py::BaseCircuitModule` captures the full training /
validation / test loop shared by every architecture.  Each arch gets
its own package with a freeze-/optim-aware `base.py` and a
concrete `module.py`.

| Path                                  | Purpose                                                      |
| ------------------------------------- | ------------------------------------------------------------ |
| `modules/base.py`                       | `BaseCircuitModule` — loop + head-oriented scalar logging. |
| `modules/cosmos_2_5_common/base.py`     | `BaseCosmosModule` — freeze schedule + optim param-group split (shared by Predict and Transfer). |
| `modules/cosmos_transfer_2_5/base.py`   | Back-compat re-export of `BaseCosmosModule`.               |
| `modules/cosmos_transfer_2_5/module.py` | `CosmosTransfer3DModule` — concrete Lightning class.       |
| `modules/cosmos_predict_2_5/module.py`  | `CosmosPredict3DModule` — concrete Lightning class.        |
| `modules/vista/base.py`                 | `BaseVistaModule` — Vista-specific freeze schedule.        |
| `modules/vista/module.py`               | `Vista3DModule` — concrete Lightning class.                |

### `nanocosmos/callbacks/` — Lightning callbacks

| Path                              | Purpose                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| `callbacks/memory.py`             | Per-epoch GPU/CPU memory logger.                                        |
| `callbacks/tensorboard/`          | `ImageLogger` — hierarchical TB visualisation (package).                |
| `callbacks/tensorboard/image_logger.py` | `ImageLogger` callback (the public class).                        |
| `callbacks/tensorboard/tags.py`   | `TagContext` — single source of `{stage}/{mode}/[{head}/]{panel}`.      |
| `callbacks/tensorboard/heads.py`  | `_log_predictions` — emits `true/{image,label,aff/*}`, `pred/{sem,raw,aff/*}`, and the Mutex Watershed `pred/label/{pre,mul}` panels; `aff_panel_indices` selects which affinity offsets to show. |
| `callbacks/tensorboard/viz.py`    | Colour-map, overlay, tile builders.                                     |

### `nanocosmos/inference/` — sliding-window + Mutex Watershed

| File                    | Purpose                                                             |
| ----------------------- | ------------------------------------------------------------------- |
| `sliding_window.py`     | Blended sliding-window inference over arbitrarily large volumes.    |
| `mutex_watershed.py`    | Parameter-free Mutex Watershed agglomeration of the predicted affinities into instance ids (`mutex_watershed` functional + `MutexWatershed` nn.Module). The eval / inference instance-segmentation step; see [`MUTEXWATERSHED.md`](./MUTEXWATERSHED.md). |

### `nanocosmos/preprocessors/` — format converters

`preprocessors/base.py::BasePreprocessor` declares the `save` / `load`
/ `validate` / `get_shape` / `get_metadata` interface.

| File         | Purpose                                     |
| ------------ | ------------------------------------------- |
| `hdf5.py`    | HDF5 preprocessor (primary format).         |
| `nrrd.py`    | NRRD preprocessor (medical imaging format). |
| `tiff.py`    | Multi-page TIFF preprocessor.               |
| `nfty.py`    | NfTy / neurofitty volumetric format.        |

### `nanocosmos/utils/` — miscellaneous helpers

| File            | Purpose                                                 |
| --------------- | ------------------------------------------------------- |
| `io.py`         | Volume read / write façade over `preprocessors/*`.      |

### `nanocosmos/visualizer/` — interactive web volume renderer

| Path                      | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| `app.py`                  | FastAPI server exposing volume tiles.              |
| `__main__.py`             | `python -m nanocosmos.visualizer` entrypoint.        |
| `volume_loader.py`        | Lazy chunked HDF5 loader for the server.           |
| `static/index.html`       | Single-page UI.                                    |
| `static/app.js`           | UI wiring + camera controls.                       |
| `static/volume_renderer.js` | WebGL 3-D volume raymarcher.                     |
| `static/style.css`        | Dark-mode layout.                                  |

---

## File count per subsystem (informational)

| Subsystem                               | .py files  |
| --------------------------------------- | ---------- |
| `nanocosmos/transforms/`                  |  7 (incl. `__init__`)                                |
| `nanocosmos/models/cosmos_2_5_common/`    |  7 (incl. `__init__`)                                |
| `nanocosmos/models/cosmos_transfer_2_5/`  |  3 (`__init__`, `wrapper`, `variants`)               |
| `nanocosmos/models/cosmos_predict_2_5/`   |  3 (`__init__`, `wrapper`, `variants`)               |
| `nanocosmos/models/vista/`                |  4 (incl. `__init__`)                                |
| `nanocosmos/callbacks/tensorboard/`       |  6 (incl. `__init__`)                                |
| `nanocosmos/losses/`                      |  4 (`__init__`, `_common`, `affinity`, `dice_bce_focal`) |
| `nanocosmos/datamodules/` + `datasets/`   |  5 + 7 (datasets incl. `lazy.py`, `_patches.py`)     |
| `nanocosmos/preprocessors/`               |  6 (incl. `__init__`, `base`)                        |
| `nanocosmos/modules/`                     |  2 (top-level) + per-arch packages (`cosmos_2_5_common`, `cosmos_transfer_2_5`, `cosmos_predict_2_5`, `cosmos_3_nano`, `vista`) |
| `nanocosmos/metrics/`                     |  3                                                   |
| `nanocosmos/inference/`                   |  3                                                   |
| `nanocosmos/utils/`                       |  4 (incl. `__init__`)                                |
| `nanocosmos/callbacks/`                   |  2 (top-level: `__init__`, `memory.py`) + `tensorboard/` package above |
| `nanocosmos/visualizer/`                  |  4 py + 4 static                                     |

---

## See also

- [`ORGANIZATION.md`](./ORGANIZATION.md) — design patterns, conventions, and
  "how to add a new …" checklists.
- `configs/*.yaml` — every knob is documented inline.
- `nanocosmos/losses/affinity.py` — `AffinityFGLoss` consumes the model's
  single `[B, HEAD_CHANNELS, …]` head tensor plus a target dict and
  returns `loss/aff`, `loss/sem`, `loss/raw`.
