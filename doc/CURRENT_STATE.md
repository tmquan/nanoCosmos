# nanoCosmos — Current State

> **Standalone, verified reference (2026-07-03).** This document consolidates and supersedes the drifted `doc/` tree. It was written directly against the code and configs; **where any older doc disagrees, this document and the code win.** Authoritative source files are cited inline throughout.
>
> Scope: what the repo *is today* — architecture, the model/checkpoint surface, the data pipeline, and how to train/evaluate. For the list of stale docs and code this replaces, see [`doc/AUDIT_REPORT.md`](AUDIT_REPORT.md).

---

## Overview & Purpose

**nanocosmos** is a PyTorch Lightning + Hydra research infrastructure for **3-D electron-microscopy (EM) connectomics segmentation**. It trains volumetric models that emit a per-voxel **affinity + semantic + reconstruction** head, then agglomerates the predicted affinities into instance labels with a **Mutex Watershed** at evaluation and inference time. The distinctive bet of the codebase is to build these segmentation models on top of NVIDIA's **Cosmos** video-diffusion backbones (a VAE tokenizer + a DiT transformer), repurposing pretrained generative-video weights as 3-D feature extractors for dense EM segmentation, with a lightweight non-Cosmos reference backbone (Vista3D / SegResNetDS2) for fast local iteration.

The package is `import nanocosmos` (version `0.1.0`, MIT); the canonical training entry point is `python scripts/train.py --config-name <recipe>` (Hydra `@hydra.main`, default config `configs/default.yaml`).

### High-level goal

Given raw 3-D EM volumes, produce **instance** segmentations (individual neurons/organelles) and **semantic** foreground maps. Rather than regressing labels directly, every model predicts a shared fixed-layout head and defers instance formation to a graph-agglomeration step:

- **Head layout** (authoritative: `nanocosmos/losses/_common.py`). The forward pass emits one `[B, HEAD_CHANNELS, *spatial]` tensor of **raw logits / linear values, with no activation applied in `forward`**. The default layout is `HEAD_CHANNELS = N_AFF + 2 = 16`:
  - channels `0 .. N_AFF-1` (default `N_AFF = 14`): per-offset **affinity** logits, `P(label[v] == label[v+offset])`, for a fixed anisotropy-aware offset set (`AFFINITY_OFFSETS`: `N_PULL` short-range pull edges + long-range push edges);
  - channel `N_AFF`: **sem** — foreground/boundary logit;
  - channel `N_AFF+1`: **raw** — linear L1 reconstruction of the normalized input EM intensity in `[-1, 1]`.

  `sigmoid` is applied only downstream (loss, metrics, Mutex Watershed, TensorBoard). Note the head width is not simply read from config: `BaseCircuitModule` (`nanocosmos/modules/base.py`) derives `head_channels = <num affinity offsets> + 2` from the active loss's offset set, so config recipes that change the offset list (e.g. the 30-offset recipes use `head_channels: 32`) stay consistent automatically. `HEAD_LAYOUT`/`slice_head` (re-exported from `nanocosmos.losses`) own the canonical channel split.

- **Loss.** `AffinityFGLoss` supervises the affinity + foreground head (with a raw-reconstruction term); the joint SSL+SFT recipes use `Joint3DReconSegLoss`, which wraps the same affinity loss and adds branch routing plus small-voxel→native-grid pooling.

- **Instances.** `nanocosmos.inference.MutexWatershed` (Wolf et al. 2018) agglomerates the predicted affinities into instances; it has torch / cupy / numpy backends selected by config. `sliding_window_inference` handles blended patch fusion at inference time.

### Model families and how they relate

Five end-to-end backbone wrappers live under `nanocosmos/models/`, all sharing a common contract (`BaseModel`) and the same post-backbone refinement head (`VistaTaskHead3D`). Geometry numbers below are authoritative from each family's `variants.py`.

| Family | Wrapper | Backbone / provenance | Key geometry | Shared scaffolding |
|---|---|---|---|---|
| **Cosmos-Predict 2.5 (2B)** | `CosmosPredict3DWrapper` | Base DiT + Wan VAE (`nvidia/Cosmos-Predict2.5-2B`, `diffusers/base/post-trained`) | hidden 2048 / 28 layers / 16 heads; latent 16ch, 8× spatial / 4× temporal | `cosmos_2_5_common` |
| **Cosmos 3 Nano (16B)** | `Cosmos3NanoWrapper` (alias `Cosmos3Nano3DWrapper`) | `Cosmos3OmniTransformer` omni MoT, init from Qwen3-VL 8B (`nvidia/Cosmos3-Nano`) + Wan2.2-TI2V VAE | hidden 4096 / 36 / 32 (8 KV, GQA); latent 48ch, 16× spatial / 4× temporal | `cosmos_3_common` |
| **Cosmos 3 Edge (4B)** | `Cosmos3EdgeWrapper` | Announced-but-unreleased tier; warm-started by **structurally reducing the loaded Nano transformer** (depth/width truncation) and reusing Nano's VAE | hidden 2048 / 28 / 16 (8 KV) | `cosmos_3_common` |
| **Cosmos 3 Super (64B)** | `Cosmos3SuperWrapper` | Omni MoT, init from Qwen3-VL 32B (`nvidia/Cosmos3-Super`) + Wan2.2-TI2V VAE; needs FSDP to train | hidden 5120 / 64 / 64 (8 KV); MLP ratio 5.0 | `cosmos_3_common` |
| **Vista3D (reference)** | `Vista3DWrapper` | MONAI `SegResNetDS2` (optional `MONAI/VISTA3D-HF` encoder warm start) | conv backbone; no DiT/VAE | `vista` |

Relationships:

- The three **Cosmos-3 tiers (Nano / Edge / Super)** are thin specializations (`wrapper.py` + `variants.py`) over a shared `Cosmos3OmniWrapper` (`cosmos_3_common`); they differ only in the variant geometry and, for Edge, the reduce-from-Nano warm start. **Edge has no released weights** — it is approximated by reducing Nano, or trained from scratch with `pretrained=false`.
- The **Cosmos-2.5 Predict (2B)** wrapper and all Cosmos-3 wrappers descend from `_BaseCosmos25Wrapper` (`cosmos_2_5_common`), which owns the VAE→DiT→feature-projector→decoder-adapter→head pipeline. All Cosmos wrappers share the freeze schedule, `dit_backbone_lr` param-group split, and optional FP8 path via `BaseCosmosModule`.
- **Vista3D** is deliberately outside the Cosmos stack — a fast, small backbone for local iteration — but reuses the identical `VistaTaskHead3D`, loss, and Mutex Watershed eval, so results are comparable.
- Every wrapper projects backbone features through the **same** `VistaTaskHead3D` and emits the **same** head layout, so the loss, metrics, and instance agglomeration are backbone-agnostic.

### Two recipe families (config-selected)

Hydra config drives all dispatch via inline registries in `scripts/train.py`: `cfg.data.dataset` → datamodule, `cfg.model.type` → Lightning module (default `joint3d_2b`).

1. **Single-task affinity recipes** (`default.yaml`, `snemi3d.yaml`, `combine.yaml`, `cosmospredict3d.yaml`, `cosmos3nano3d.yaml`): one dataset family (SNEMI3D / MICrONS / FLYEM3D / CREMI3D / neurons) with a chosen Cosmos or Vista backbone, supervised by `AffinityFGLoss`.
2. **Joint reconstruction + segmentation ("nanoCosmos") recipes** (`nanocosmos-2B.yaml` / `-4B.yaml` / `-16B.yaml`): one backbone on a fixed fine voxel grid, trained on a `joint3d` multi-task datamodule with two round-robin branches — **SSL** (self-supervised reconstruction, no labels) and **SFT** (segmentation, labels pooled to native grid) — supervised by `Joint3DReconSegLoss`. `joint3d_2b`→Cosmos-Predict 2B, `joint3d`→Nano 16B, `joint3d_edge`→Edge 4B, `joint3d_super`→Super 64B. The `Joint3D*` modules subclass `Cosmos3Nano3DModule` and swap only the backbone wrapper and loss.

> **Authoritative sources.** Head/channel layout and offsets: `nanocosmos/losses/_common.py`. Config→code dispatch and CLI: `scripts/train.py` (`build_datamodule`, `build_module`). Backbone geometries and provenance: `nanocosmos/models/*/variants.py`. Public API surface: `nanocosmos/__init__.py`. Where this document and inline docstrings disagree with those files, the code wins.

**Caveat surfaced during verification:** `scripts/train.py` still references `nanocosmos/transforms/skeleton.py` in comments (lines ~13/57), but no such module exists in the tree — treat the "kimimaro/EDT skeleton transform" mentions as stale.

---

## Repository Architecture & Layout

nanoCosmos is a PyTorch Lightning + Hydra codebase for volumetric connectomics instance segmentation. A single dense head emits a `[B, HEAD_CHANNELS, *spatial]` tensor (affinity + semantic-foreground + raw-reconstruction channels), supervised by an affinity loss and agglomerated into instances at eval time by the Mutex Watershed. Everything in the package is designed so that swapping the backbone (Cosmos-Predict 2.5, Cosmos-3 Nano/Edge/Super, or Vista3D) is a config-only change.

> Authority note: The Python package under `nanocosmos/` is the source of truth for behavior; the Hydra YAML files under `configs/` are the source of truth for which classes are instantiated and with what hyperparameters. `scripts/train.py` is the source of truth for how config keys map to classes. Where this document and any older top-level docs disagree, the code and configs win.

### Package layout

The package root is `nanocosmos/`. Its `__init__.py` is a public-API facade that re-exports the dataset, preprocessor, datamodule, loss, model-wrapper, and Lightning-module classes and sets `__version__ = "0.1.0"`.

| Package | Role | Key classes / entry points |
|---|---|---|
| `models/` | Backbone **wrappers** — the entire network (encoder + decoder + dense head) as an `nn.Module`. | `BaseModel` (abstract), tier wrappers, shared `cosmos_2_5_common/`, `cosmos_3_common/`, and per-tier `cosmos_3_{nano,edge,super}/`, `cosmos_predict_2_5/`, `vista/` subpackages. |
| `modules/` | Lightning **modules** — the train/eval loop wrapping one model wrapper + one loss + a Mutex Watershed agglomerator. | `BaseCircuitModule`, `BaseCosmosModule`, `BaseVistaModule`, concrete per-architecture modules, and joint recipes in `joint3d.py`. |
| `datasets/` | Dataset classes. Eager MONAI `CacheDataset`-backed `CircuitDataset` family + lazy HDF5 `LazyVolDataset`; `_patches.py` is a patch-index helper. | `CircuitDataset`, `SNEMI3DDataset`, `MICRONSDataset`, `CREMI3DDataset`, `FLYEM3DDataset`, `NeuronsDataset`, `LazyVolDataset`. |
| `datamodules/` | Lightning `DataModule`s — own the MONAI augmentation pipeline and DataLoader config, one per dataset plus a joint multi-task one. | `CircuitDataModule` (base), per-dataset modules, `Joint3DDataModule`. |
| `transforms/` | Connectomics-specific MONAI dict (`MapTransform`) augmentations, plus EDT dispatch helpers. | `Labeld`, `FindBoundariesd`, `RandSpatialCropForegroundd`, `RandTransposeXYd`, `RandResolutionZoomd`, `RandMissingSliced`, `RandResolutionDegraded`, `ToFineGridd`. |
| `losses/` | Loss functions + canonical channel-layout module. | `AffinityFGLoss`, `DiceBCEFocalLoss`, `Joint3DReconSegLoss`; `_common.py` defines `HEAD_CHANNELS`/`HEAD_LAYOUT`/`slice_head`/`AFFINITY_OFFSETS`. |
| `metrics/` | Segmentation metrics, per-point and per-batch. | Instance (ARI/AMI/VOI/TED) in `instance.py`; semantic (Dice/IoU) in `semantic.py`. |
| `callbacks/` | Lightning callbacks wired in declaratively from config. | `CudaEmptyCacheCallback`, `CudaMemoryLoggerCallback`, and the `tensorboard/` subpackage's `ImageLogger` / `Joint3DImageLogger`. |
| `inference/` | Post-training path. | `sliding_window_inference` (blended patch fusion) + `MutexWatershed` (affinity→instance agglomeration; torch/cupy/numpy backends). |
| `preprocessors/` | Format-agnostic volume I/O. | `BasePreprocessor` + `TIFFPreprocessor`, `HDF5Preprocessor`, `NRRDPreprocessor`, `NFTYPreprocessor`. |
| `visualizer/` | Standalone FastAPI + WebGL volume viewer (`python -m nanocosmos.visualizer`). **Not imported by training.** | `app.py`, `volume_loader.py`, `__main__.py`. |
| `utils/` | Cross-cutting helpers that dispatch to preprocessors. | `find_folder`, `load_volume`, `save_volume` (`io.py`). |

The key conceptual distinction is **`models/` (a wrapper = the whole `nn.Module` network) vs `modules/` (a Lightning module = the training recipe that owns a wrapper).** A Lightning module stores its wrapper as `self.model`, so all trainable weights live under the `model.` state-dict prefix.

### Class inheritance hierarchies

**Lightning modules (`modules/`).** All modules share one train/eval loop defined once in `BaseCircuitModule`:

```
pl.LightningModule
└── BaseCircuitModule                       (modules/base.py)  — shared loop; declares _model_cls/_loss_cls contract
    ├── BaseCosmosModule                    (modules/cosmos_2_5_common/base.py) — freeze schedule, dit_backbone_lr param split, fp8 hookup; the ONLY place that calls save_hyperparameters()
    │   ├── CosmosPredict3DModule           _model_cls=CosmosPredict3DWrapper, _loss_cls=AffinityFGLoss
    │   ├── Cosmos3NanoModule               (alias Cosmos3Nano3DModule)  _model_cls=Cosmos3NanoWrapper
    │   ├── Cosmos3EdgeModule               _model_cls=Cosmos3EdgeWrapper
    │   ├── Cosmos3SuperModule              _model_cls=Cosmos3SuperWrapper
    │   └── Joint3DModule                   (modules/joint3d.py, subclass of Cosmos3Nano3DModule)  _loss_cls=Joint3DReconSegLoss
    │       ├── JointPredict3DModule        _model_cls=CosmosPredict3DWrapper
    │       ├── JointEdge3DModule           _model_cls=Cosmos3EdgeWrapper
    │       └── JointSuper3DModule          _model_cls=Cosmos3SuperWrapper
    └── BaseVistaModule                     (modules/vista/base.py)
        └── Vista3DModule                   _model_cls=Vista3DWrapper, _loss_cls=AffinityFGLoss
```

Concrete modules are declarative: they set two class attributes, `_model_cls` (the wrapper) and `_loss_cls` (the loss), and inherit the loop. `BaseCircuitModule.__init__` derives the head width from the loss's affinity-offset set (`head_channels = len(offsets) + 2`) and injects it into the model config, overriding any stale `model.head_channels` value with a warning — so the wrapper, loss, and Mutex Watershed always agree on one channel count. The joint modules subclass `Cosmos3Nano3DModule` (thus inherit `save_hyperparameters()`), swapping only `_model_cls`/`_loss_cls` and the target/metric routing. Note the joint families use `head_channels: 32` (30 affinity offsets + sem + raw) whereas the default single-dataset recipe uses `head_channels: 16` (14 offsets + 2).

**Model wrappers (`models/`).** `BaseModel` (in `models/base.py`) is the abstract contract — a `forward(x) -> [B, HEAD_CHANNELS, *spatial]` head tensor of raw logits/linear values plus `get_output_channels()`. Per its own docstring, the concrete production wrappers inherit directly from `torch.nn.Module` rather than `BaseModel`, but honor the same single-tensor forward contract:

```
nn.Module
├── _BaseCosmos25Wrapper                    (models/cosmos_2_5_common/wrapper_base.py) — shared Cosmos scaffold: fallback down-proj, DiT, feature projector, decoder adapter, VAE
│   ├── CosmosPredict3DWrapper              (models/cosmos_predict_2_5/wrapper.py)
│   └── Cosmos3OmniWrapper                  (models/cosmos_3_common/wrapper.py) — omni backbone + latent-patch repatch + residual-VAE handling
│       ├── Cosmos3NanoWrapper              (alias Cosmos3Nano3DWrapper)  variant "NANO"
│       ├── Cosmos3EdgeWrapper              variant "EDGE" (reduce_omni_transformer from Nano)
│       └── Cosmos3SuperWrapper             variant "SUPER"
├── Vista3DWrapper                          (models/vista/wrapper.py) — SegResNetDS2 backbone + shared head
├── VistaTaskHead3D                         (models/vista/heads.py) — shared dense head used by BOTH Vista and the Cosmos decoder adapter
├── _StandaloneDiT3D / _DiTBlock            (standalone_dit.py) — random-init DiT fallback when pretrained weights unavailable
└── _DecoderAdapter3D / _FeatureProjector3D / _ProgressiveUpsampler3D  (decoder.py)
```

The Cosmos families deliberately expose underscore-prefixed helpers (`_BaseCosmos25Wrapper`, `_DecoderAdapter3D`, `_VariantConfig`, layer/decoder primitives) that are used across sibling modules. Back-compat aliases keep both names live: `Cosmos3Nano3DWrapper = Cosmos3NanoWrapper` and `Cosmos3Nano3DModule = Cosmos3NanoModule`.

**Datasets and datamodules.** Both use inheritance to share loading/augmentation logic:

```
monai CacheDataset + Randomizable + ABC        pl.LightningDataModule + ABC
└── CircuitDataset (datasets/base.py)           └── CircuitDataModule (datamodules/base.py)
    ├── SNEMI3DDataset                              ├── SNEMI3DDataModule
    ├── MICRONSDataset                              ├── MICRONSDataModule
    │   ├── CREMI3DDataset                           │   ├── CREMI3DDataModule
    │   └── FLYEM3DDataset                           │   └── FLYEM3DDataModule
    └── NeuronsDataset                              └── NeuronsDataModule
LazyVolDataset (datasets/lazy.py) is a plain      Joint3DDataModule (datamodules/joint3d.py) is a
torch Dataset over HDF5 (not a CircuitDataset).   plain pl.LightningDataModule (not a CircuitDataModule);
                                                  it uses a round-robin batch sampler and does NOT call
                                                  save_hyperparameters().
```

`CircuitDataModule.get_train_transforms` assembles the MONAI `Compose` pipeline from the `transforms/` package plus MONAI built-ins (elastic deformation uses MONAI's `Rand3DElasticd` directly).

**Preprocessors.** `BasePreprocessor` (abstract) → `TIFFPreprocessor`, `HDF5Preprocessor`, `NRRDPreprocessor`, `NFTYPreprocessor`. `utils/io.py` dispatches `load_volume`/`save_volume` to the right leaf by format.

### How Hydra config wires it together

The single training entry point is `scripts/train.py` (`@hydra.main`, default config `configs/default.yaml`). Config keys select classes through **inline registry dicts inside `train.py`** (deliberately not an import-time registry — a new dataset/model is a one-line edit there plus a leaf class):

- **`cfg.data.dataset`** → `build_datamodule()`. Map: `snemi3d`/`microns`/`flyem3d`/`cremi3d`/`neurons` → the matching `*DataModule` (kwargs filtered to the class signature via `inspect.signature`); `joint3d` is a distinct branch built directly from the `data.branches`/`data.degrade` schema (a different config shape).
- **`cfg.model.type`** → `build_module()` (default `joint3d_2b`). Map includes `vista3d`, `cosmos3edge3d`, `cosmos3nano3d`, `cosmos3super3d`, `cosmospredict3d`, the four joint recipes (`joint3d`→Nano/16B, `joint3d_edge`→Edge/4B, `joint3d_super`→Super/64B, `joint3d_2b`→Predict/2B), plus verbose/legacy aliases (`cosmos3_nano_3d`, `joint_predict3d`, etc.). The selected module receives four config dicts: `model_config`, `optimizer_config`, `loss_config`, `training_config`.
- **`cfg.model.type` startswith `joint`** → `setup_callbacks()` picks `Joint3DImageLogger`, otherwise `ImageLogger`.
- **`cfg.training.strategy`** → `ddp` (`DDPStrategy`), `fsdp` (`FSDPStrategy`), or passthrough. **`cfg.training.mutex_watershed`** (strides, backend, etc.) → the module's `MutexWatershed` agglomerator. **`cfg.logger`** / **`cfg.training.profiler`** select TensorBoard-vs-W&B and the profiler.

Config wiring inside the modules: `model.type` is popped and used only for dispatch; `loss.type` is a readability hint that is stripped before forwarding (the module class fixes the loss class via `_loss_cls`); and `model.head_channels` is overridden by the loss-offset-derived value as noted above.

**Config families** (`configs/`):
- **Inheritance chain**: `default.yaml` (base: snemi3d + cosmos3nano3d) ← `snemi3d.yaml` ← `combine.yaml` (data-only overrides; no top-level `model.type`, so it resolves to the inherited default module). Standalone flattened recipes `cosmospredict3d.yaml` (2B backbone) and `cosmos3nano3d.yaml` (16B) carry no `defaults:` and pin one backbone each.
- **Joint resolution-ladder family**: `nanocosmos-2B.yaml` / `nanocosmos-4B.yaml` / `nanocosmos-16B.yaml` — all use `data.dataset: joint3d` with a single backbone (2B / Edge-4B / Nano-16B) predicting reconstruction + segmentation on a fixed fine grid, with SSL and SFT branches.

**Serialization note:** `BaseCosmosModule.__init__` is the only place that calls `save_hyperparameters()`, so all Cosmos/Joint module checkpoints carry `hyper_parameters`, but `Vista3DModule` checkpoints do not (neither `BaseCircuitModule` nor `BaseVistaModule` calls it). Weights always serialize under the `model.` prefix; the highest-churn checkpoint prefixes are `model.dit.*` (class/shape varies with `pretrained`, `variant`, Cosmos-3 latent-patch repatch, Edge structured reduction, and FP8 swap) and `model.decoder_adapter.*`.

**Secondary entry points:** `python -m nanocosmos.visualizer` (FastAPI viewer, default `127.0.0.1:8899`) and the standalone dataset-download/conversion utilities under `scripts/` (each guarded by `if __name__ == "__main__"`, not imported by the package).

Relevant authoritative files: `/localhome/local-tranminhq/nanoCosmos/nanocosmos/__init__.py`, `/localhome/local-tranminhq/nanoCosmos/nanocosmos/modules/base.py`, `/localhome/local-tranminhq/nanoCosmos/nanocosmos/models/base.py`, `/localhome/local-tranminhq/nanoCosmos/nanocosmos/modules/joint3d.py`, `/localhome/local-tranminhq/nanoCosmos/scripts/train.py` (registries at lines 237/313), and `/localhome/local-tranminhq/nanoCosmos/configs/`.

---

## Model & Module Reference (Checkpoint Surface)

This section is the authoritative map of nanoCosmos's concrete model families, their `LightningModule` wrappers, and the serialized surface that a maintainer must preserve to keep checkpoints loadable. Where this document and the code disagree, **the code is authoritative** — every claim below was verified against the files cited inline.

### Layering at a glance

Every training run instantiates one `LightningModule` whose trainable weights live under the single attribute `self.model` (a `nn.Module` "wrapper"). Three inheritance spines exist:

```
pl.LightningModule
└── BaseCircuitModule            (nanocosmos/modules/base.py:60)   — shared train/eval/log loop, NO save_hyperparameters()
    ├── BaseCosmosModule         (nanocosmos/modules/cosmos_2_5_common/base.py:54) — the ONLY save_hyperparameters() caller
    │   ├── CosmosPredict3DModule (modules/cosmos_predict_2_5/module.py:13)
    │   ├── Cosmos3NanoModule     (modules/cosmos_3_nano/module.py:12; alias Cosmos3Nano3DModule)
    │   ├── Cosmos3EdgeModule     (modules/cosmos_3_edge/module.py:15)
    │   ├── Cosmos3SuperModule    (modules/cosmos_3_super/module.py:12)
    │   └── Joint3DModule         (modules/joint3d.py:42) → JointPredict3DModule, JointEdge3DModule, JointSuper3DModule
    └── BaseVistaModule          (nanocosmos/modules/vista/base.py:18) — NO save_hyperparameters()
        └── Vista3DModule         (modules/vista/module.py:8)
```

Note that `BaseCosmos3Module` referenced by config is a thin re-export of `BaseCosmosModule`; the Cosmos-3 nano/super modules import `BaseCosmosModule` via `nanocosmos.modules.cosmos_3_common.base`.

### LightningModule `__init__` contract

`BaseCircuitModule.__init__` (`modules/base.py:112`) is the single signature every module inherits — subclasses do **not** override it, they only set the class attributes `_model_cls` and `_loss_cls` (plus `_SPATIAL_DIMS = 3`):

```python
def __init__(self, model_config=None, optimizer_config=None,
             loss_config=None, training_config=None, **kwargs)
```

`BaseCosmosModule.__init__` (`modules/cosmos_2_5_common/base.py:73`) wraps that same 4-dict signature, adding two behaviors before delegating to the base:

1. It pops `model_config["hf_token"]`, calls `self.save_hyperparameters()`, then re-inserts the token (`base.py:82-86`). **The HF token is deliberately excluded from the saved hyperparameters** while all four config dicts are captured.
2. After the base builds `self.model`, if a pretrained VAE encoder is present it freezes `self.model._fallback_down` (`base.py:96-97`), and it optionally FP8-swaps the DiT (`base.py:128-135`).

Concrete Cosmos/Joint/Vista modules carry no `__init__` of their own. The full class → wrapper → loss table:

| Module | file:line | `_model_cls` | `_loss_cls` | `save_hyperparameters()`? |
|---|---|---|---|---|
| `CosmosPredict3DModule` | `modules/cosmos_predict_2_5/module.py:13` | `CosmosPredict3DWrapper` | `AffinityFGLoss` | yes (inherited) |
| `Cosmos3NanoModule` (=`Cosmos3Nano3DModule`) | `modules/cosmos_3_nano/module.py:12` | `Cosmos3NanoWrapper` | `AffinityFGLoss` | yes |
| `Cosmos3EdgeModule` | `modules/cosmos_3_edge/module.py:15` | `Cosmos3EdgeWrapper` | `AffinityFGLoss` | yes |
| `Cosmos3SuperModule` | `modules/cosmos_3_super/module.py:12` | `Cosmos3SuperWrapper` | `AffinityFGLoss` | yes |
| `Joint3DModule` | `modules/joint3d.py:42` | `Cosmos3NanoWrapper` (inherited) | `Joint3DReconSegLoss` | yes |
| `JointPredict3DModule` | `modules/joint3d.py:136` | `CosmosPredict3DWrapper` | `Joint3DReconSegLoss` | yes |
| `JointEdge3DModule` | `modules/joint3d.py:151` | `Cosmos3EdgeWrapper` | `Joint3DReconSegLoss` | yes |
| `JointSuper3DModule` | `modules/joint3d.py:165` | `Cosmos3SuperWrapper` | `Joint3DReconSegLoss` | yes |
| `Vista3DModule` | `modules/vista/module.py:8` | `Vista3DWrapper` | `AffinityFGLoss` | **NO** |

The joint modules subclass `Cosmos3Nano3DModule` and change only task routing (`_loss_offsets`, `_prepare_targets`, `_accumulate_metrics`); they add no new submodules, so their state_dict layout is identical to a plain Cosmos module with the same wrapper (`modules/joint3d.py:1-27`).

### The canonical head width (loss-driven, not config-driven)

`BaseCircuitModule.__init__` derives the unified head width from the **loss's affinity offset set**, not from `model.head_channels` (`modules/base.py:133-148`): `head_channels = n_affinity_offsets + 2` (one semantic channel + one raw-reconstruction channel). If `model_config["head_channels"]` disagrees it is overwritten with a warning. The default offset set has 14 entries, so `HEAD_CHANNELS = N_AFF + 2 = 16` (`losses/_common.py:81-104`; verified `N_AFF = 14`). The 30-offset configs (`cosmospredict3d.yaml`, the joint recipes) therefore pin `head_channels = 32`.

**Checkpoint rule:** the final head's Conv3d out-channels are fixed by the loss offset count. Loading a 16-channel checkpoint into a 32-offset run (or vice versa) will mismatch the head's final conv — the offset set is part of the checkpoint contract even though it lives in `loss_config`.

### Wrapper families and their state_dict layout

All weights are stored under the `model.` prefix. `self.criterion` (loss) and `self.agglomerator` (`MutexWatershed`) are parameter-free and contribute no meaningful keys.

#### Cosmos 2.5 / 3 family — `_BaseCosmos25Wrapper` (`models/cosmos_2_5_common/wrapper_base.py:83`)

This is the shared parent of **every Cosmos wrapper** and defines the bulk of the layout. Its `__init__` (`wrapper_base.py:106`) signature:

```python
def __init__(self, in_channels=1, head_channels=HEAD_CHANNELS, feature_size=64,
             variant="2B", dtype="bf16", pretrained=True,
             freeze_dit_backbone=False, freeze_vae_decoder=False, freeze_vae_encoder=True,
             gradient_checkpointing=False, feature_layers=None, cache_dir=None,
             hf_token=None, dropout=0.0, input_supersample=1, highres_skip=False,
             highres_skip_channels=8, vae_input_pm1=True, vae_symmetrize_z=False,
             decode_chunk=16, **kwargs)
```

`BaseCosmosModule._build_model` (`base.py:104-136`) forwards a fixed subset of these from `model_config` (it does not pass `input_supersample`, `highres_skip*`, or `vae_input_pm1`, which fall back to their defaults unless a subclass adds them via `_extra_model_kwargs`).

Registered submodules (state_dict prefixes under `model.`):

- **`_fallback_down`** (`wrapper_base.py:231`) — `Conv3d(3, lc*2) → _NORM → GELU → _PointwiseLinear`. **Always constructed**, even when a pretrained VAE encoder is loaded (then frozen at `base.py:97`).
- **`dit`** (set in `_build_backbone`) — the transformer. **Class is conditional:** a diffusers class (e.g. `CosmosTransformer3DModel`, `Cosmos3OmniTransformer`) when `pretrained=True`, else the random-init `_StandaloneDiT3D` (`standalone_dit.py:83`). The two key sets are **mutually incompatible**.
- **`feature_projector`** (`wrapper_base.py:245`) — `_FeatureProjector3D`, forced `.float()`. Its first Linear's in-features = `hidden_dim * len(feature_layers)`, so changing the *number* of hooked feature layers changes this layer's shape.
- **`decoder_adapter`** (`wrapper_base.py:254`) — `_DecoderAdapter3D` (below).
- **`vae_encoder` / `vae_decoder`** — registered as submodules only on the pretrained path (`None`, no keys, on the standalone path). The VAE decoder body is aliased under two prefixes: `model.vae_decoder.*` and `model.decoder_adapter.decoder_body.*`.

Deliberately **not** registered (kept off the state_dict): `self._vae_ref = [vae]` is a plain Python list (`wrapper_base.py:585`), and `_hook_block_container` is set via `object.__setattr__`.

#### `_DecoderAdapter3D` (`models/cosmos_2_5_common/decoder.py:100`)

Construction is heavily conditional (the top checkpoint-fragility hotspot):

- **Pretrained path** (`vae_decoder is not None`): `to_latent = _PointwiseLinear(feature_size, latent_channels)`; `decoder_body = vae_decoder` (the Wan decoder); the pretrained final conv is snapshotted as `original_conv_out` (frozen) and the body's own `conv_out`/`output_conv`/`proj_out`/`final_conv` is replaced with `nn.Identity()` (`decoder.py:206-226`).
- **Standalone path**: `to_latent = None`; `decoder_body = _ProgressiveUpsampler3D(...)` (a `ModuleList` of `num_stages` upsampling blocks).
- **`skip_stem`** exists only when `highres_skip=True` (`decoder.py:176-188`); its presence adds `model.decoder_adapter.skip_stem.*` keys *and* widens the head's first-conv in-channels (`head_in = _hidden_ch + skip_channels`).
- **`head = VistaTaskHead3D(in_channels=head_in, out_channels=head_channels, refine_channels=feature_size, dropout=dropout)`**, forced `.float()`.

#### `CosmosPredict3DWrapper` (`models/cosmos_predict_2_5/wrapper.py:21`)

Pure subclass of `_BaseCosmos25Wrapper`; only sets `_variant_configs = _VARIANT_CONFIGS`, no extra submodules. Uses diffusers `CosmosTransformer3DModel` + `AutoencoderKLWan`. The `2B` variant geometry (`variants.py:41-46`): hidden 2048 / 28 layers / 16 heads / latent 16 / spatial 8× / temporal 4×.

#### Cosmos-3 family — `Cosmos3OmniWrapper` (`models/cosmos_3_common/wrapper.py:124`)

Subclass of `_BaseCosmos25Wrapper` with `__init__(self, *args, variant="NANO", **kwargs)` (`wrapper.py:161`). After the base build it runs three post-load mutations:

1. **`_repatch_to_unit_latent_patch()`** (`wrapper.py:169, :212`) — rebuilds `dit.proj_in`/`dit.proj_out` as new `nn.Linear` layers with reshaped/averaged weights (latent patch 2→1). **Idempotent** and a no-op on the standalone DiT. A Cosmos-3 checkpoint's `model.dit.proj_in/proj_out` shapes therefore differ from raw HF weights.
2. **`_maybe_wrap_residual_vae_decoder()`** (`wrapper.py:173, :701`) — swaps `decoder.conv_out` (an `nn.Identity`) for a `_CacheTolerantIdentity` (**no params**) and wraps `decoder.forward` (a Python attribute, not serialized). No new keys.
3. **`_install_time_embedder_dtype_guard()`** (`wrapper.py:177, :663`) — a forward-pre-hook, not serialized.

Tier wrappers subclass `Cosmos3OmniWrapper`, set the registry + default variant (all verified against `variants.py`):

| Wrapper | file:line | default variant | hidden / layers / heads / latent / spatial / temporal |
|---|---|---|---|
| `Cosmos3NanoWrapper` (=`Cosmos3Nano3DWrapper`) | `cosmos_3_nano/wrapper.py:19` | `"NANO"` | 4096 / 36 / 32 / 48 / 16× / 4× |
| `Cosmos3EdgeWrapper` | `cosmos_3_edge/wrapper.py` | `"EDGE"` | 2048 / 28 / 16 / 48 / 16× / 4× |
| `Cosmos3SuperWrapper` | `cosmos_3_super/wrapper.py` | `"SUPER"` | 5120 / 64 / 64 / 48 / 16× / 4× |

**`Cosmos3EdgeWrapper` fragility:** it overrides `_post_load_diffusers` (`wrapper.py:37`) to replace `self.dit` with a structurally reduced transformer via `reduce_omni_transformer` when `cfg.reduce_from_parent` is set — Edge downloads the **Nano** repo and truncates depth 36→28 and width (`models/cosmos_3_common/reduce.py`, `child.load_state_dict(..., strict=False)`). An Edge checkpoint's `model.dit.*` keys/shapes are the reduced-Edge geometry, produced dynamically at construction; raw Nano weights do not match Edge shapes.

#### Vista family — `Vista3DWrapper` (`models/vista/wrapper.py:25`)

```python
def __init__(self, in_channels=1, head_channels=HEAD_CHANNELS, feature_size=64,
             encoder_name="vista3d", dropout=0.0, pretrained=False,
             hf_repo_id=DEFAULT_VISTA3D_REPO, hf_revision=DEFAULT_VISTA3D_REVISION,
             cache_dir=None, hf_token=None, **kwargs)
```

Submodules:
- **`backbone`** — **conditional class** (`wrapper.py:98-124`): `SegResNetDS2(init_filters=feature_size, blocks_down=(1,2,2,4,4), norm="instance", dsdepth=1)` when `encoder_name in ("vista3d","segresnet_ds2")` and MONAI provides it; otherwise falls back to `SegResNet`. The two classes have different `model.backbone.*` key sets. Set `feature_size=48` to cleanly load the pretrained `MONAI/VISTA3D-HF` encoder (upstream `init_filters=48`).
- **`head = VistaTaskHead3D(in_channels=feature_size, out_channels=head_channels, refine_channels=feature_size, dropout=dropout)`** (`wrapper.py:91`).

`BaseVistaModule._build_model` (`vista/base.py:29`) forwards `in_channels, head_channels, feature_size, encoder_name, dropout, pretrained, cache_dir, hf_token`.

#### `VistaTaskHead3D` (`models/vista/heads.py:48`) — shared by Vista and the Cosmos decoder

```python
def __init__(self, in_channels, out_channels, refine_channels=None,
             dropout=0.0, norm_name="instance")
```

The `block` `nn.Sequential` has **index-shifting conditional layers** (`heads.py:101-133`):
- A width-matcher `Conv3d(in_channels, refine, 1)` is prepended **only when `in_channels != refine_channels`**.
- Two residual `UnetrBasicBlock(spatial_dims=3, in=refine, out=refine, kernel=3, res_block=True, norm=instance)`.
- `Dropout3d` inserted **only when `dropout > 0`**.
- Final `Conv3d(refine, out_channels, 1)`.

**Checkpoint rule:** `...head.block.{N}.*` indices are renumbered by whether `in_channels == refine_channels` and whether `dropout > 0`. Changing `feature_size`, the decoder's hidden width, `highres_skip`, or dropout between save and load renumbers head keys even when the shapes would otherwise fit.

### What `save_hyperparameters()` actually persists

`BaseCosmosModule.__init__` is the **only** module-level `save_hyperparameters()` call (`modules/cosmos_2_5_common/base.py:84`). It captures the four config dicts (`model_config`, `optimizer_config`, `loss_config`, `training_config`) into the checkpoint's `hyper_parameters`, minus `hf_token`. Consequences:

- All Cosmos and Joint checkpoints carry `hyper_parameters`.
- **`Vista3DModule` checkpoints carry NO `hyper_parameters`** — neither `BaseCircuitModule` nor `BaseVistaModule` calls it.
- On the datamodule side, the `CircuitDataModule` family (SNEMI3D/MICRONS/CREMI/FLYEM/Neurons) saves hyperparameters, but **`Joint3DDataModule` does not** (`datamodules/joint3d.py`).

There are **no** `torch.save` / `torch.load` / `load_from_checkpoint` calls anywhere in the package — checkpoint write/read is entirely delegated to Lightning's `ModelCheckpoint` / `Trainer(ckpt_path=...)`. The only explicit `load_state_dict` sites are two warm-start weight-porting utilities that run at construction, before any Lightning checkpoint is written: `reduce_omni_transformer` (Edge DiT reduction, `models/cosmos_3_common/reduce.py`) and `load_pretrained_vista3d_encoder` (`models/vista/hf_loader.py`). Both use `strict=False`.

### Checkpoint-compatibility rules a maintainer must respect

1. **`model.dit.*` is the highest-risk prefix.** Its class and key set change with `pretrained` (diffusers vs `_StandaloneDiT3D`), with `variant`, with the Cosmos-3 unit-latent-patch repatch (rebuilt `proj_in`/`proj_out`), with the Edge structured reduction, and with the FP8 module-swap (`fp8: true` replaces DiT `nn.Linear` with `Float8Linear`, preserving FQNs but changing the parameter/buffer set). A bf16 and an fp8 checkpoint of the same DiT are not interchangeable.
2. **`model.decoder_adapter.*` is second.** The presence of `to_latent`, `original_conv_out`, and `skip_stem`, and the class of `decoder_body` (Wan decoder vs `_ProgressiveUpsampler3D`), all flip on `pretrained` and `highres_skip`.
3. **Head width is pinned by the loss offset count** (`head_channels = n_offsets + 2`), not by `model.head_channels`. Keep the offset set stable across save/resume.
4. **Head key indices shift** with `(in_channels == refine_channels)` and `(dropout > 0)`; do not change `feature_size`/dropout/skip between train and resume without re-mapping keys.
5. **`vae_symmetrize_z` is a fresh-run-only setting.** It shifts the latent distribution (adds no keys), so a checkpoint trained with it OFF will not resume cleanly with it ON (documented at `wrapper_base.py:169-173`).
6. **Back-compat aliases** — `Cosmos3Nano3DWrapper = Cosmos3NanoWrapper` (`cosmos_3_nano/wrapper.py:30`) and `Cosmos3Nano3DModule = Cosmos3NanoModule` (`cosmos_3_nano/module.py:27`) exist so older configs/checkpoints keep resolving. Keep them.

For any layout question, the authoritative sources are the wrapper `__init__` methods (`wrapper_base.py`, `decoder.py`, `heads.py`, the per-tier `wrapper.py`/`variants.py`) and the two `__init__` methods that own hyperparameter serialization (`modules/base.py:112`, `modules/cosmos_2_5_common/base.py:73`).

---

## Data Pipeline & Datasets

nanoCosmos loads 3-D electron-microscopy (EM) connectomics volumes, augments them through a MONAI transform pipeline, and feeds either 2-D slices or 3-D patches into training. Two dataset families and two datamodule families coexist; the flagship joint reconstruction+segmentation recipes use the lazy dataset plus a multi-task round-robin datamodule. The code under `nanocosmos/datasets/`, `nanocosmos/datamodules/`, `nanocosmos/transforms/`, `nanocosmos/preprocessors/`, and `scripts/` is authoritative for everything below.

### On-disk convention

Every dataset resolves per-volume entries against a flat `data_root` directory. A volume list is a list of dicts, each with a `vol` key (raw EM basename) and an optional `seg` key (instance-label basename); label-less (SSL / image-only) volumes omit `seg`. Optional per-volume keys are `root` (override `data_root` for that volume) and `find_boundaries` (eager datasets only, see below). File lookup (`nanocosmos/utils/io.py::find_folder`) is **non-recursive** — it tries `root/{base}{ext}` for the supported extensions in priority order (`.h5`, `.hdf5`, `.hdf`, `.tiff`, `.tif`, `.nrrd`, `.nhdr`, `.npy`, `.npz`).

The canonical stored layout produced by the download scripts is HDF5 with dataset key `main` and axis order `[Z, Y, X]`. Segmentation is `int64` with `0 = background`; raw EM is typically `uint8`.

### Datasets (`nanocosmos/datasets/`)

Two families with different semantics (`datasets/__init__.py` is the public surface):

**1. Eager `CircuitDataset` family** — subclasses of MONAI `CacheDataset`, preload entire volumes into RAM at `__init__`, then serve crops through the transform pipeline. Best for small-to-medium datasets and 2-D slice mode.

- `CircuitDataset` (`base.py`) — abstract base. Requires each leaf to implement `paper`, `resolution` (a `{"x","y","z"}` dict), `labels`, `data_files`, and `_prepare_data`. Supports a virtual epoch length (`_virtual_len`): `__getitem__` wraps the index modulo the real dataset length, so `num_samples` can decouple epoch size from volume count.
- `SNEMI3DDataset` (`snemi3d.py`) — Kasthuri et al. 2015, 6×6×30 nm, labels `["background","neuron"]`. Loads HDF5 or TIFF. Normalizes each volume to [0,1] via per-volume min/max. In `slice_mode=True` emits per-z-slice 2-D entries; otherwise one 3-D entry per volume.
- `MICRONSDataset` (`microns.py`) — MICrONS minnie65, resolution set to **8×8×40 nm** (the code comment is explicit that this is the released mip-0 EM voxel size, not the 4×4×40 nm annotation frame). Adds a 3-D patch mode: when `slice_mode=False` and `patch_size` is set, it tiles each volume with `generate_patch_indices` (`_patches.py`, overlap-fraction grid, default 0.25). Loads HDF5/TIFF/NRRD.
- `CREMI3DDataset` (`cremi3d.py`) and `FLYEM3DDataset` (`flyem3d.py`) — thin **metadata-only** subclasses of `MICRONSDataset` (loading/patching/normalization inherited verbatim). CREMI is 4×4×40 nm ssTEM *Drosophila*; FLYEM is **8×8×8 nm** near-isotropic FIB-SEM (FIB-25 / Hemibrain / MaleCNS).
- `NeuronsDataset` (`neurons.py`) — Kasthuri 2015 dense cylinder, 6×6×30 nm, same patch/slice logic as MICRONS.

`generate_patch_indices` (`_patches.py`) is the single shared tiling helper for MICRONS/Neurons: it produces `(z,y,x)` slice triples covering the whole volume, shifting the trailing patch backward at edges so no zero-padding is introduced.

**2. Lazy `LazyVolDataset`** (`lazy.py`) — a plain `torch.utils.data.Dataset` that reads only the requested patch from disk per `__getitem__`, keeping RAM constant regardless of volume count/size. This is the loader used by all 3-D patch training (including every joint recipe). Key mechanics:

- Stores only metadata (`_VolumeHandle`: paths, shape, HDF5 key) — HDF5 keys are auto-resolved (`main`/`data`/`raw`/`volume`/`image`/`label`, else first dataset). Volumes smaller than `patch_size` on any spatial axis are skipped with a warning (no lazy zero-padding).
- Per-worker thread-local caches of HDF5 file handles (`swmr=True`, `locking=False`, 128 MB chunk cache) and dataset objects; TIFF via memmap.
- **Volume picking is voxel-count-weighted** (`_pick_volume`, `np.searchsorted` over cumulative voxels, PCG64 RNG seeded by index).
- **Per-volume normalization** to [0,1] using min/max estimated from a handful of small centered probes (default 5 probes of ≤512² per z-quantile), cached to a `<file>.norm.json` sidecar so subsequent ranks/runs load instantly.
- **Crop-quality rejection sampling** (up to `max_foreground_retries`, default 50; falls back to best-seen crop). Gates:
  - Labeled volumes: `min_foreground` (label non-zero fraction), plus optional `image_min_foreground` (image non-zero), plus optional instance-diversity gates `min_instances` and `max_inst_frac` (reject crops filled by one giant instance that gives the affinity loss no push signal).
  - Label-less volumes: `image_min_foreground` (non-zero), `image_min_std` (block-wise contrast — rejects flat resin/embedding medium), and `image_min_autocorr` (lag-1 in-plane spatial autocorrelation — rejects noise-dominated crops that pass the variance gate).
- `deterministic=True` seeds per-sample RNG by index (used for validation groups).

### Preprocessors (`nanocosmos/preprocessors/`)

Format-isolation layer used by the eager `CircuitDataset` leaves. `BasePreprocessor` (`base.py`) defines the `load` / `validate` / `save` / `get_shape` contract; concrete leaves are `HDF5Preprocessor` (`hdf5.py`, default key `main`, extensions `.h5/.hdf5/.hdf/.he5`), `TIFFPreprocessor` (`tiff.py`), `NRRDPreprocessor` (`nrrd.py`), and `NFTYPreprocessor` (`nfty.py`, NIfTI). Note there are two distinct I/O paths: the preprocessor classes (used by eager datasets) and the standalone suffix-dispatch functions in `utils/io.py` (`load_volume`/`save_volume`, used by scripts/utilities); the lazy dataset uses its own inline h5py/tifffile readers, not the preprocessors.

### Transforms (`nanocosmos/transforms/`)

Connectomics-specific MONAI `MapTransform` (dict-in/dict-out) wrappers that slot into `monai.transforms.Compose`. Public surface (`transforms/__init__.py`):

- `Labeld` (`label.py`) — connected-component **relabel after crop** (value-based connectivity; cucim GPU with skimage CPU fallback). Splits instances that became disconnected by cropping, optional `min_voxels` fragment removal.
- `FindBoundariesd` (`find_boundaries.py`) — probabilistically zeros instance-boundary voxels so touching instances get a gap. **Anisotropy-aware**: when `pixel_size` z-spacing is >2× the min xy-spacing it switches to xy-only boundary detection (avoids physically-thick z boundaries). Includes a no-full-erase guard that restores any instance erosion would delete entirely. The module-level `find_boundaries()` function is also called eagerly by the `CircuitDataset` leaves at load time when a volume spec sets `find_boundaries > 0`.
- `RandSpatialCropForegroundd` (`rand_crop_foreground.py`) — `RandSpatialCropd` with rejection sampling on foreground fraction (`min_foreground`, up to 50 attempts, best-seen fallback). Used by the eager pipeline.
- `RandTransposeXYd` (`rand_transpose_xy.py`) — random Y↔X transpose, completing the dihedral-8 XY symmetry group alongside flips + rot90.
- `RandResolutionZoomd` (`resolution_zoom.py`) — multi-dataset resolution harmonization. Per-volume native resolution is looked up from `resolution_map` by **longest-prefix match on the `volume` key**. Two modes: `ratio` (legacy, scale-jitter preserving each volume's anisotropy) and `union` (independently sample z and xy targets onto a shared envelope; **isotropic volumes such as FIB-25 8×8×8 are skipped**). Labels use nearest-neighbour interpolation with large-ID→compact remapping to avoid float32 precision loss (e.g. MICrONS uint64 IDs). `DEFAULT_TARGET_RANGE` is z 30–40, xy 6–8 nm.
- `RandMissingSliced` (`missing_slice.py`) — CREMI-style missing/damaged z-section defect, **image-only** (labels stay intact so the model learns to bridge connectivity). Fill modes `zero`/`mean`/`replicate`.
- `RandResolutionDegraded` (`degrade.py`) — the self-supervised (SSL) supervisor for the joint recipe. Takes a small-voxel patch and manufactures a realistic large-voxel acquisition (z slab-average decimation via `adaptive_avg_pool3d`, per-section jitter, missing/duplicated sections, per-section noise + gamma, then trilinear z-upsample back to the grid). Writes the degraded volume to `image` and a pristine copy to `recon_image` (the L1 reconstruction target). `zf` (voxel-size factor) is sampled log-uniformly over `zf_range` (default 2–10). Label-free.
- `ToFineGridd` (`fine_grid.py`) — the resolution-ladder resampler. Places `image` on the fixed fine grid (trilinear), keeps `label`/`sem_label` at native resolution (the segmentation loss pools the fine head back down), and puts `recon_image` on the coarser of native/fine. On the `sft` branch it captures the clean native image as `recon_image` before resampling; on `ssl` the degrade transform already wrote it.

`edt.py` provides GPU/CPU dispatch helpers (EDT, gaussian, centroid, CC-label) via cucim with scipy/skimage fallback; forked DataLoader workers use the CPU path since CUDA contexts don't survive `fork()`.

### Datamodules (`nanocosmos/datamodules/`)

**1. `CircuitDataModule` (`base.py`)** — the shared base for the single-dataset leaves. Owns the MONAI pipeline and DataLoader wiring. Selects between eager and lazy datasets per leaf. Train pipeline order (verified in `get_train_transforms`):

```
EnsureChannelFirst → [FindBoundaries (boundary_target="both")]
→ SpatialPad → [RandSpatialCropForeground | RandSpatialCrop] (safe or patch size)
→ [Labeld + ResolutionZoom] → [CenterSpatialCrop(patch_size)]
→ spatial aug (flip×3, rot90, transposeXY, [elastic])
→ Labeld (CC-relabel) → intensity aug (contrast, gaussian noise)
→ [RandMissingSlice] → [sem_label copy + FindBoundaries]
→ EnsureType
```

When resolution zoom can downsample (zoom < 1), an enlarged **safe** crop (`_safe_patch_size`, computed from the finest native resolution across `resolution_map`) is taken first so the subsequent zoom + center-crop never introduce zero-padding. `boundary_target` selects whether membrane erosion hits the shared instance `label` (`both` — affects sem + affinity targets + val GT) or a separate `sem_label` copy (`semantic` — instance label stays pristine). The val pipeline uses `find_boundaries` prob 1.0 and no missing-slice augmentation.

DataLoaders use `forkserver` multiprocessing, `prefetch_factor` (default 6), `drop_last=True` on train, and a NUMA-binding `worker_init_fn`. `bind_local_numa` pins each rank's process (by `LOCAL_RANK`) and its workers to the GPU-local NUMA socket on multi-socket nodes; best-effort, no-ops on single-socket/non-Linux.

**2. Single-dataset leaves** — `SNEMI3DDataModule`, `MICRONSDataModule`, `NeuronsDataModule`, `CREMI3DDataModule` (subclass of MICRONS), `FLYEM3DDataModule` (subclass of MICRONS). Each leaf switches between eager (2-D slice mode) and lazy (`_use_lazy = not slice_mode and patch_size is not None`, 3-D patch mode) at `setup`. In lazy mode they build `LazyVolDataset` splits (default `num_samples=16000`, reading `_effective_read_size()` so the safe-crop margin is honored). `MICRONSDataModule` requires a non-empty `train_volumes` in lazy mode.

**3. `Joint3DDataModule` (`joint3d.py`)** — the multi-task datamodule for the flagship `nanocosmos-2B/4B/16B` recipes; a plain `pl.LightningDataModule` (does **not** call `save_hyperparameters()`). It runs the network on a **fixed fine grid** (`patch_size` @ cubic `pixel_size`, e.g. `[200,256,256]` @ 4 nm). Two branches under `data.branches`:
- `ssl` — label-free reconstruction: reads native crops, applies `RandResolutionDegraded`, then `ToFineGridd` (recon target already set).
- `sft` — segmentation: reads native `image`+`label`, CC-relabels, optional boundary erosion (per this group's native resolution), then `ToFineGridd` (captures clean native image as recon target).

Volumes are bucketed into **round-robin groups** per the `balance` policy — `resolution` (one group per `(task, native_resolution)`; within a group volumes are voxel-count-weighted), `subset` (one group per domain, scheduled equally, weighted by `subset_weights`), or `volume` (one group per volume, weighted by per-volume `sample_weight`). A subset name is derived from the volume stem (`_derive_subset`) or overridden by an explicit `subset` field. Each group is single-task and single-resolution, so **every batch is task- and shape-homogeneous** — the contract `Joint3DReconSegLoss` / `Joint3DModule` require. `_RoundRobinBatchSampler` interleaves per-group batch counts (shuffled on train, sequential on val), and each branch's `sample_weight` scales its group lengths (normalized by the branch mean so weights are relative). Empty groups (all volumes skipped) are logged and dropped rather than aborting. Each group is a `LazyVolDataset` at that group's native patch size (`native_patch = round(fine_patch × fine_nm / native_res)`), with the crop-quality gates wired per task (`sft_min_foreground`/`sft_min_instances`/`sft_max_inst_frac` for sft; `ssl_min_foreground`/`ssl_min_std`/`ssl_min_autocorr` for ssl). DataLoaders use `use_distributed_sampler: false` (the batch sampler is custom).

### Download & conversion scripts (`scripts/`)

All are standalone (`__main__`-guarded), write into the `data/<DATASET>` roots, and normalize into the canonical HDF5 (`main`, `[Z,Y,X]`) layout the loaders consume. `--out-dir` defaults are relative (`data/...`).

- `download_all.py` — orchestrator over the per-dataset downloaders into the resolution-ladder layout, then verifies every produced `.h5` (key/shape/dtype/resolution provenance/foreground %). Verification is config-driven against the roots in `configs/nanocosmos-16B.yaml`. Supports `--verify-only`, `--dry-run`, `--datasets`.
- `download_cosem3d.py` — COSEM/OpenOrganelle ~4 nm near-cubic FIB-SEM (HeLa/Jurkat/macrophage) via CloudVolume; image-only SSL (smallest-voxel ladder rung). S3 bucket `janelia-cosem-datasets`.
- `download_flyem3d.py` — FlyEM 8 nm FIB-SEM (`fib25`/`hemibrain`/`malecns`) via CloudVolume; `--role` = `sft` (image+seg) or `ssl` (image-only).
- `download_cremi3d.py` — downloads CREMI `.hdf` (A/B/C labeled train; A+/B+/C+ image-only test) and splits the nested `volumes/raw` + `volumes/labels/neuron_ids` into separate `_volume.h5` / `_segmentation.h5`.
- `download_microns.py` — MICrONS minnie65 EM + static segmentation (default version v1300) via bossdb/GCS.
- `download_snemi3d.py` — Kasthuri 2015 data at 6×6×30 nm: AC3/AC4 from snemi.zip and the annotated `neurons` cylinder from GCS.
- `download_zenodo_582636.py` — a rice-grain X-ray micro-CT record used as a non-connectomics instance-segmentation smoke test (parallel, MD5-verified, resumable); also a template for other Zenodo records.
- `convert_mitoem2.py` — converts MitoEM2 nnU-Net `.nii.gz` images (transposing `(X,Y,Z)`→`(Z,Y,X)`, reading `native_resolution` from voxel spacing) to image-only `mitoem2_<subset>_<crop>_volume.h5` for the SSL branch; validates and overwrites truncated outputs.
- `collect_meaningful_crops.py` — probes a CloudVolume grid with cheap small patches and only fetches full crops passing the same content/autocorr/non-zero gates as `LazyVolDataset` (avoids blind re-downloading of resin/off-tissue regions).

### How volumes/patches flow into training

1. `scripts/train.py` selects a datamodule from `cfg.data.dataset` (special-casing `joint3d` → `Joint3DDataModule`).
2. `setup()` builds datasets from the configured `train/val/test_volumes` (or the joint `branches`), applying per-volume `root`/`native_resolution`.
3. For 3-D patch training, `LazyVolDataset` reads a random (voxel-count-weighted, quality-gated) native patch per sample and normalizes it to [0,1].
4. The MONAI `Compose` pipeline augments the patch and constructs loss targets: instance `label` (CC-relabeled), optional eroded `sem_label`, and (joint recipe) the fine-grid `image` plus the `recon_image` reconstruction target and a `task` tag.
5. The DataLoader collates task/shape-homogeneous batches (round-robin sampler for the joint recipe; standard shuffled loader otherwise) and hands them to the Lightning module, which routes `image` to the backbone and `label`/`sem_label`/`recon_image` to the affinity+foreground / joint recon-seg loss.

Relevant files: `/localhome/local-tranminhq/nanoCosmos/nanocosmos/datasets/{base,lazy,microns,snemi3d,cremi3d,flyem3d,neurons,_patches}.py`; `/localhome/local-tranminhq/nanoCosmos/nanocosmos/datamodules/{base,snemi3d,microns,joint3d}.py`; `/localhome/local-tranminhq/nanoCosmos/nanocosmos/transforms/{label,find_boundaries,rand_crop_foreground,rand_transpose_xy,resolution_zoom,missing_slice,degrade,fine_grid,edt}.py`; `/localhome/local-tranminhq/nanoCosmos/nanocosmos/preprocessors/`; `/localhome/local-tranminhq/nanoCosmos/nanocosmos/utils/io.py`; `/localhome/local-tranminhq/nanoCosmos/scripts/download_*.py`, `convert_mitoem2.py`, `collect_meaningful_crops.py`.

---

## Training, Config & Inference

This section is authoritative to the code at `/localhome/local-tranminhq/nanoCosmos`. The single source of truth for the run wiring is `scripts/train.py`; config defaults live in `configs/*.yaml`; the eval-time agglomeration lives in `nanocosmos/modules/base.py` + `nanocosmos/inference/`.

### Launching training

Training is a single Hydra entry point: `scripts/train.py` (decorated `@hydra.main(version_base=None, config_path="../configs", config_name="default")`). Run a recipe by name:

```bash
python scripts/train.py                                   # default.yaml
python scripts/train.py --config-name cosmospredict3d     # 2B Cosmos-Predict baseline
python scripts/train.py --config-name nanocosmos-2B       # joint recon+seg (2B)
# Hydra dotted overrides work on any key:
python scripts/train.py --config-name snemi3d data.batch_size=8 training.max_epochs=200
python scripts/train.py training.fast_dev_run=true
```

`main()` (train.py:820) does, in order: install runtime patches, create the run directory, seed, build the datamodule, build the module, optionally `torch.compile`, build callbacks/logger/profiler, build the trainer, resolve any checkpoint, then `trainer.fit(...)` wrapped in crash recovery.

- **Run directory**: `Path(output_dir) / "{timestamp}_{experiment_name}"`, default `output_dir="outputs"` (train.py:829). Checkpoints go to `<run_dir>/checkpoints`, logs to `<run_dir>/logs`, and a `final_model.ckpt` is always saved on rank 0 at the end (train.py:886), plus a `crash_recovery.ckpt` if `fit` throws (train.py:805).
- **Runtime patches** (`_install_runtime_patches`, train.py:128) force `torch.load(..., weights_only=False)` (Lightning ckpts pickle `defaultdict`/`DictConfig`), allow-list OmegaConf types for the unpickler, silence known Lightning/MONAI warnings, and set `torch.set_float32_matmul_precision("high")`.
- **NCCL / NVLink hardening** is set at *module import time* (train.py:66–94, via `os.environ.setdefault` so it is overridable): `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`, `TORCH_NCCL_TRACE_BUFFER_SIZE=2048`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`. These work around an Xid 145 NVLink fault on the target B300 node; they are node-specific defaults, not universal requirements.

**Multi-GPU with NUMA (`numa_run.sh`)**: this wrapper is hardcoded to one dual-socket 8-GPU B300 node (GPUs 0–3 → NUMA node0, 4–7 → node1, split at `LOCAL_RANK < 4`) and requires `numactl`. Launch with torchrun so every rank runs the wrapper and Lightning reuses the existing process group:

```bash
torchrun --standalone --nproc_per_node=8 --no-python ./numa_run.sh
```

Note `numa_run.sh` hardcodes `--config-name cosmos3nano3d`; edit the script to change the recipe.

### Config-driven dispatch

All architecture/dataset selection is by config value, resolved through inline registries in `train.py` (not an import-time plugin registry):

- **`cfg.data.dataset`** → `build_datamodule()` (train.py:237). Map: `snemi3d`→`SNEMI3DDataModule`, `microns`→`MICRONSDataModule`, `flyem3d`→`FLYEM3DDataModule`, `cremi3d`→`CREMI3DDataModule`, `neurons`→`NeuronsDataModule`. `joint3d` is a distinct branch (train.py:258) building `Joint3DDataModule` from a different schema (`data.branches` / `data.degrade` / SSL+SFT crop gates), bypassing the shared single-dataset kwargs. Single-dataset modules are filtered to the kwargs each class actually accepts (train.py:308).
- **`cfg.model.type`** → `build_module()` (train.py:313), default `joint3d_2b`. Map: `vista3d`→`Vista3DModule`, `cosmos3edge3d`→`Cosmos3EdgeModule`, `cosmos3nano3d`→`Cosmos3Nano3DModule`, `cosmos3super3d`→`Cosmos3SuperModule`, `cosmospredict3d`→`CosmosPredict3DModule`, `joint3d`→`Joint3DModule`, `joint3d_edge`→`JointEdge3DModule`, `joint3d_super`→`JointSuper3DModule`, `joint3d_2b`→`JointPredict3DModule`, plus legacy aliases (`cosmos3_nano_3d`, `joint_predict3d`, etc.). Each concrete module fixes its own `_model_cls`/`_loss_cls`; `loss.type` is stripped before forwarding (it is a readability hint only).
- **`cfg.training.strategy`** → `setup_strategy()` (train.py:541): `ddp`→`DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True, timeout=30min)`; `fsdp`→`FSDPStrategy` (FULL_SHARD, `use_orig_params=True`, bf16 `MixedPrecision`, auto-wraps only `Cosmos3VLTextMoTDecoderLayer` from diffusers) for the 15.2B Nano backbone; any other value is passed through.
- **`cfg.logger`** → `setup_logger()`: `tensorboard`→`TensorBoardLogger`, `wandb`→`WandbLogger`, else `True`. **`cfg.training.profiler`** → `simple`/`advanced`/`pytorch` profiler or `None`.
- **`torch.compile`** (`_maybe_compile`, train.py:378) compiles only the DiT backbone (`module.model.dit`) when `training.compile` is truthy, and is skipped entirely when `model.gradient_checkpointing` is set (dynamo cannot trace the checkpoint HOP). `true`→`reduce-overhead`; a string is used as the mode verbatim.

`build_trainer` (train.py:630) forces `use_distributed_sampler=False` for the `joint3d` dataset (its custom round-robin batch sampler is not a `BatchSampler` subclass); it is otherwise `True` and always overridable.

**Checkpoint resume** (`_resolve_checkpoint`, train.py:695): `training.resume_from_checkpoint` is a full Lightning resume (optimizer/epoch/scheduler) passed to `trainer.fit(ckpt_path=...)`. `+ckpt_path=...` is a weights-only warm start applied in place with `strict=False`, optionally filtered by the mutually-exclusive `+ckpt_path_skip_prefixes` / `+ckpt_path_only_prefixes` (operating on `module.state_dict()` keys, e.g. `model.dit.`). The two mechanisms are mutually exclusive.

### Available configs

Two families. **Inheritance chain**: `default` ← `snemi3d` ← `combine` (via Hydra `defaults:`). **Flattened standalone recipes** (no `defaults:`): `cosmospredict3d`, `cosmos3nano3d`, and the three joint recipes `nanocosmos-{2B,4B,16B}`. Inspect the merged tree with `python scripts/show_config.py --config-name <recipe>` (referenced in `default.yaml`).

| Config | `model.type` → module | `data.dataset` | Notes (verified) |
|---|---|---|---|
| `default.yaml` | `cosmos3nano3d` → `Cosmos3Nano3DModule` | `snemi3d` | Root defaults; `head_channels: 16`, `feature_size: 64`, `precision: bf16-mixed`, `strategy: ddp`, AdamW `lr 1e-3` + cosine, `max_epochs 100`, MWS `strides [1,4,4]` `size_filter 50`. |
| `snemi3d.yaml` | `cosmos3nano3d` | `snemi3d` | `defaults: [default]`; Nano 16B end-to-end on SNEMI3D+neurons+MICrONS, `bf16-true`, cosine-warmup. |
| `combine.yaml` | inherits `snemi3d` | inherits | `defaults: [snemi3d]`; **data-only override** (drops AC4 from train, keeps in val/test). Has no `model.type`/`dataset` of its own — it resolves to the inherited `cosmos3nano3d`/`snemi3d`. |
| `cosmospredict3d.yaml` | `cosmospredict3d` → `CosmosPredict3DModule` | `snemi3d` | Flattened; 2B Cosmos-Predict baseline, `variant 2B`, `head_channels 32`, `freeze_dit_backbone: 5` (freeze N epochs), `precision bf16-mixed`, `compile: true`, MWS `strides [1,1,1]` + `gate_with_sem`. |
| `cosmos3nano3d.yaml` | `cosmos3nano3d` | `snemi3d` | Flattened; same combined data recipe as above but Nano 16B, `bf16-true`, `compile: false`, `weight_decay 0.0`. |
| `nanocosmos-2B.yaml` | `joint3d_2b` → `JointPredict3DModule` | `joint3d` | Joint recon+seg on a 4 nm fine grid; SSL+SFT branches, round-robin sampler, `Joint3DReconSegLoss`. |
| `nanocosmos-4B.yaml` | `joint3d_edge` → `JointEdge3DModule` | `joint3d` | Edge tier (`variant EDGE`), reduced-from-Nano DiT; `gradient_checkpointing: [dit]` (per-component list), FP8 knobs present but effectively disabled per config comments. |
| `nanocosmos-16B.yaml` | `joint3d` → `Joint3DModule` | `joint3d` | Nano 16B joint recipe; smaller batch/patch than 2B. |

Cross-cutting invariant: **head width is config-derived, not free**. `BaseCircuitModule.__init__` (base.py:139–148) sets `head_channels = n_affinity_offsets + 2` (aff offsets + sem + raw) from the loss offsets, overriding any stale `model.head_channels` (with a warning on mismatch). Default offset set is 14 → 16 channels; the 30-offset recipes → 32. The optimizer is hardcoded to AdamW in `configure_optimizers`; the `optimizer:` block only tunes hyperparameters and the scheduler wrapper.

### Callbacks & logging

`setup_callbacks()` (train.py:422) builds the list from the `callbacks:` block:

- **`ModelCheckpoint`** (enabled by default): monitors `val/automatic/loss` (`mode: min`, `save_top_k: 3`, `save_last: true`), `auto_insert_metric_name=False`, dir `<output_dir>/checkpoints`.
- **`CudaMemoryLoggerCallback`** (`nanocosmos/callbacks/memory.py`, enabled by default): logs `cuda_memory/*` to the logger every N steps (inherits `training.log_every_n_steps` if unset).
- **`CudaEmptyCacheCallback`**: optional (`cuda_empty_cache_before_val`), clears the allocator around validation.
- **`LearningRateMonitor`** (`logging_interval="step"`, enabled by default).
- **Image logger** (enabled by default): `str(cfg.model.type).startswith("joint")` selects `Joint3DImageLogger`, otherwise `ImageLogger` (train.py:476; both from `nanocosmos.callbacks`, `spatial_dims=3`). Renders per-head panels (true/pred image, label, aff/*, sem, raw) plus the Mutex-Watershed instance segmentation, at epoch end on rank 0. Panel logic is in `nanocosmos/callbacks/tensorboard/heads.py`.
- Always appended: `RichProgressBar()` and `ModelSummary(max_depth=2)`.

Scalar metric tags follow `{stage}/{mode}/...` (e.g. `val/automatic/loss`, `.../ins/metric/{ari,ami,voi,voi_split,voi_merge,ted}`); the hierarchy is authoritative in `nanocosmos/modules/base.py`.

### Inference / eval path

There are two distinct pieces.

**Mutex Watershed (the wired eval-time agglomeration).** `nanocosmos/inference/mutex_watershed.py::MutexWatershed` is a parameter-free affinity→instance agglomerator (Wolf et al. 2018). `BaseCircuitModule` instantiates `self.agglomerator = MutexWatershed(**mws_config)` (base.py:165) with `offsets`/`n_pull` defaulting to the loss's convention so head, target, and agglomerator agree. During validation, `_accumulate_instance_metrics` (base.py:455) runs it on `head_pred[:, aff_slice].sigmoid()` restricted to GT foreground, then scores ARI/AMI/VOI/TED. Constructor knobs (mws_watershed.py:900): `strides` (subsamples push edges only — a throughput lever, default `(1,4,4)`), `size_filter`, `max_push_edges`, `buckets`, `grow_boundaries`/`grow_max_distance`, `gate_with_sem`/`sem_gate_threshold`, and `backend`.

Backends (`_resolve_backend`, mutex_watershed.py:945): `auto` selects **`mws_th`** — a native-torch bucketed Boruvka approximation, the zero-copy default on CUDA inputs — falling back to the exact numba `mws_np` on CPU. `cupy` selects the DLPack `mws_cp` path (falls back to torch if cupy is absent); `cpu` forces `mws_np`. Note: the docstring/config comment near base.py:154 and `default.yaml`:151 say the CUDA default is "mws_cp"; the resolver code is authoritative and returns `"torch"` for `auto`/`gpu`/`torch` on CUDA. Set `training.mutex_watershed.backend` to override.

**Sliding-window inference (standalone utility, not in the training loop).** `nanocosmos/inference/sliding_window.py::sliding_window_inference` runs patch-wise forward passes over a full volume and blends overlapping patches (Gaussian/`create_gaussian_weight`, average, or max), returning the unified head tensor `[C, D, H, W]` (split with `nanocosmos.losses.slice_head`). The caller must move the model to `device` first; stride defaults to `patch_size // 2`. This function is exported from `nanocosmos/inference/__init__.py` but is **not invoked by `train.py` or any module** — its only in-repo callers are `tests/test_sliding_window.py`. It is a building block for external full-volume prediction, not part of the validation metric path (which uses MWS on cropped patches).

### Corrections to the input maps

- The GPU MWS default is `mws_th` (native torch), not `mws_cp` — verified in `_resolve_backend` (mutex_watershed.py:945). The `mws_cp`/`mws_np` phrasing in `base.py` comments and `default.yaml` is stale relative to the resolver.
- `train.py:57` references `nanocosmos/transforms/skeleton.py`, which does not exist in the tree (confirmed stale comment).
- `combine.yaml` has no top-level `model.type`/`dataset`; it inherits `cosmos3nano3d` + `snemi3d` from its `defaults: [snemi3d]` chain (verified), so it does resolve to a valid module at runtime.

Authoritative files: `/localhome/local-tranminhq/nanoCosmos/scripts/train.py`, `/localhome/local-tranminhq/nanoCosmos/numa_run.sh`, `/localhome/local-tranminhq/nanoCosmos/configs/*.yaml`, `/localhome/local-tranminhq/nanoCosmos/nanocosmos/modules/base.py`, `/localhome/local-tranminhq/nanoCosmos/nanocosmos/inference/{mutex_watershed,sliding_window}.py`, `/localhome/local-tranminhq/nanoCosmos/nanocosmos/callbacks/{__init__.py,memory.py,tensorboard/heads.py}`.

---

## Verified environment & quick facts

- **Env:** conda env `nanocosmos` — Python 3.13, torch 2.12.1+cu130, pytorch-lightning 2.6.1, MONAI 1.5.2, 8× CUDA.
- **Tests:** `python -m pytest tests/` → 173 passed (~29 s, CPU-friendly; joint-module/datamodule tests run `fast_dev_run` end-to-end).
- **Checkpoint contract:** Lightning ckpt; weights under `model.*`; `hyper_parameters` = `{model,optimizer,loss,training}_config` (Cosmos/Joint only; Vista saves none). Head width = `len(loss offsets) + 2`. Preserve `model.dit.*`, `model.vae_encoder/decoder.*`, `model.decoder_adapter.*` (incl. `decoder_body.*` alias), `model.feature_projector.*`, `model._fallback_down.*`, and the `Cosmos3Nano3D*` aliases.
- **Known stale comments in code (safe to ignore / clean):** references to `nanocosmos/transforms/skeleton.py` and a `kimimaro/EDT` pipeline (`scripts/train.py`), and to a `cosmos_transfer` package (`models/cosmos_2_5_common`) — neither exists in the tree.
