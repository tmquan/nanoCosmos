# New lease: nanocosmos-4B-pretrain on 8× B300

Start `configs/nanocosmos-4B-pretrain.yaml` on a fresh 8× NVIDIA B300 SXM6
(sm_103) machine. The recipe is Cosmos3-Edge pretraining: every batch is raw
EM, degraded online, reconstructed at 4 nm. There is no segmentation loss.

The crop is `patch_size: [256, 256, 256]` at `pixel_size: [4, 4, 4]` nm.
Lightning uses all visible GPUs (`devices: auto`, `strategy: ddp`). One
`python scripts/train.py` process is the launcher; do not wrap it in
`torchrun`.

## 1. Check the machine

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -c 'import torch' 2>/dev/null || true
nvcc --version || true
df -h "$HOME"
```

Expect 8 B300s, about 275 GiB each, and a driver that can run CUDA 13.4.
Leave several terabytes free under `$HOME`: the EM tree is the large part,
and the Edge snapshot is about 8 GB.

If CUDA 13.4 and Anaconda are not installed yet:

```bash
git clone https://github.com/tmquan/nanoCosmos.git "$HOME/nanoCosmos"
cd "$HOME/nanoCosmos"
sudo ./setup.sh
```

`setup.sh` installs CUDA 13.4.3, cuDNN, and Anaconda at `$HOME/anaconda3`.
Skip it when those are already present.

## 2. Environment

Python 3.13. Keep the cu134 nightly. Replacing it with the Cosmos Framework
cu130 pin, or with a pip resolver that pulls `torch==2.11`, breaks sm_103.

```bash
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda create -n nanocosmos python=3.13 -y
conda activate nanocosmos
pip install uv
cd "$HOME/nanoCosmos"
uv pip install -r requirements.txt --prerelease=allow --index-strategy unsafe-best-match
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_capability())'
```

The print must be `2.16.0.dev20261027+cu134` and capability `(10, 3)`.
`requirements.txt` also pins `diffusers==0.42.1`, which is the release that
reads Edge `hidden_act: relu2` and `qk_norm_for_text: false`.

`nvidia/Cosmos3-Edge` is gated. Export a token that has accepted that model:

```bash
export HF_TOKEN=hf_...
```

The loader reads `HF_TOKEN` (`nanocosmos/models/cosmos_2_5_common/hf_loader.py`).
The first rank downloads the snapshot into the Hugging Face cache. Later
ranks reuse it.

## 3. Data

Every train and val entry in `configs/nanocosmos-4B-pretrain.yaml` is an
HDF5 volume under `$HOME/data`. Roots used by this config:

| Root | What it is |
|---|---|
| `$HOME/data/COSEM3D` | COSEM cells resampled to 4 nm |
| `$HOME/data/FLYEM3D` | FIB-25, Hemibrain, MaleCNS |
| `$HOME/data/CREMI3D` | CREMI padded ssTEM |
| `$HOME/data/SNEMI3D` | SNEMI3D test |
| `$HOME/data/MitoEM2` | MitoEM2 image volumes |

Each file is named `<vol>_volume.h5` next to its `root`, with dataset key
`main` and axes `[Z, Y, X]`.

Copy an existing tree when you have one:

```bash
mkdir -p "$HOME/data"
rsync -a <old-host>:data/COSEM3D "$HOME/data/"
rsync -a <old-host>:data/FLYEM3D "$HOME/data/"
rsync -a <old-host>:data/CREMI3D "$HOME/data/"
rsync -a <old-host>:data/SNEMI3D "$HOME/data/"
rsync -a <old-host>:data/MitoEM2 "$HOME/data/"
ln -sfn "$HOME/data" "$HOME/nanoCosmos/data"
```

Otherwise download from the repo root. `scripts/download_all.py` covers
COSEM, FlyEM, CREMI, and SNEMI3D. MitoEM2 is a separate zip-to-HDF5
conversion. MICrONS and H01 are not in this pretrain config.

```bash
cd "$HOME/nanoCosmos"
python scripts/download_all.py --datasets cosem flyem cremi snemi
python scripts/convert_mitoem2.py --data-dir "$HOME/data/MitoEM2" --extract-zips
```

COSEM tiles whose names contain `_4nm_` were resampled with
`scripts/resample_cosem_isotropic.py`. A fresh `download_cosem3d.py` run
writes that isotropic file when the downloader's resample step is enabled;
do not point the yaml at a native anisotropic crop and only change
`native_resolution` in the config.

Confirm every path the yaml names is a readable HDF5 before launching:

```bash
cd "$HOME/nanoCosmos"
python - << 'PY'
import os, re
from pathlib import Path
import h5py
text = Path("configs/nanocosmos-4B-pretrain.yaml").read_text()
home = os.environ["HOME"]
missing = []
for block in re.split(r"\n\s*- vol:", text)[1:]:
    vol = block.splitlines()[0].strip().split()[0]
    root = re.search(r"root:\s*(\S+)", block).group(1).replace("${oc.env:HOME}", home)
    path = Path(root) / f"{vol}_volume.h5"
    if not path.is_file():
        missing.append(path)
        continue
    with h5py.File(path, "r") as handle:
        assert "main" in handle, path
print(f"missing {len(missing)}")
for path in missing:
    print(path)
PY
```

A missing file is skipped with a warning at startup, so the run will
train on a quieter subset of the ladder. Fix the list before launching.

## 4. Launch

From the repo root, inside `tmux`, with the conda env activated:

```bash
cd "$HOME/nanoCosmos"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate nanocosmos
export HF_TOKEN=hf_...
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD"
python scripts/train.py --config-name nanocosmos-4B-pretrain
```

Leave `CUDA_VISIBLE_DEVICES` unset so DDP takes all 8 GPUs. Logs and
checkpoints go to `$HOME/outputs`.

A healthy step sits near 70 GiB of 275 GiB on each GPU with activation
checkpointing on (`gradient_checkpointing: [dit, decode]`). The first step
is the long one: Edge weights, the Wan encode of a 256³ crop, and the first
checkpointed backward. Sanity validation runs 2 batches before epoch 1.
Each epoch is 500 train batches (`limit_train_batches`). Validation runs
after epochs 10, 20, …, 100 (`check_val_every_n_epoch: 10`,
`limit_val_batches: 100`). Epoch 101 does not validate. `max_epochs` is 101.

`torch.compile` stays off. It does not trace the checkpoint hop.

If a 3D convolution fails with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`,
the process did not see the env's cuDNN plugins. Activate `nanocosmos`
before launch, or:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.13/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 5. Resume

Full trainer state, including epoch and optimizer:

```bash
python scripts/train.py --config-name nanocosmos-4B-pretrain \
  training.resume_from_checkpoint="$HOME/outputs/<run>/checkpoints/last.ckpt"
```

Weights only, new optimizer:

```bash
python scripts/train.py --config-name nanocosmos-4B-pretrain \
  +ckpt_path="$HOME/outputs/<run>/checkpoints/last.ckpt"
```

Use one of those, not both. Checkpoints keep the best 3 by
`val/automatic/loss` and always write `last.ckpt`.

## What this run trains

Frozen: Wan encoder, Wan decoder except the last up-block and output norm,
and the AR reasoner. Trained: DiT generation half (including the patch-1
repatch of `proj_in`), the EM feature projector, and the 32-channel task
head. The loss is L1 on the raw channel (`weight_rec: 1`, `weight_ssl: 5`,
`weight_seg: 0`). The head stays 32 channels so a later joint SFT checkpoint
load does not change shape. Optimizer: AdamW, head lr `2e-4`, DiT lr
`1e-4`, betas `[0.9, 0.99]`, cosine over 101 epochs down to `6e-5`, no
warmup. Precision is `bf16-true`.
