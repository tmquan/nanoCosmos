"""End-to-end test for the joint data path (synthetic volumes, stub backbone).

Writes tiny HDF5 volumes, builds :class:`JointDataModule`, and (a) checks the
per-branch batch contract (task / fine-grid image / native label / recon
target), and (b) runs the *real* ``JointModule`` loop over the datamodule via
``pl.Trainer(fast_dev_run=True)`` on CPU -- i.e. the full "can I train?" path
without the 16B backbone.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl

from nanocosmos.datamodules import JointDataModule
from nanocosmos.modules import JointModule
from nanocosmos.transforms import ToFineGridd

# Small affinity offsets so tiny test grids stay above the offset reach.
_OFFSETS = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [-2, 0, 0], [0, -2, 0], [0, 0, -2]]
_N_PULL = 3
_HEAD = len(_OFFSETS) + 2


def _write_h5(path, arr):
    import h5py
    with h5py.File(str(path), "w") as f:
        f.create_dataset("main", data=arr, compression="gzip")


# ---------------------------------------------------------------------------
# ToFineGridd
# ---------------------------------------------------------------------------

def test_to_fine_grid_shapes():
    t = ToFineGridd(image_size=(40, 32, 32), recon_size=(20, 16, 16),
                    set_recon_from_image=True, task="sft")
    d = t({"image": torch.rand(1, 32, 16, 16), "label": torch.zeros(1, 32, 16, 16)})
    assert tuple(d["image"].shape) == (1, 40, 32, 32)       # -> fine grid
    assert tuple(d["recon_image"].shape) == (1, 20, 16, 16)  # -> recon grid
    assert tuple(d["label"].shape) == (1, 32, 16, 16)        # native, untouched
    assert d["task"] == "sft"


# ---------------------------------------------------------------------------
# JointDataModule
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_root(tmp_path):
    rng = np.random.default_rng(0)
    # dapt source: 4 nm (== fine), image only.
    _write_h5(tmp_path / "cosem_vol.h5", (rng.random((64, 64, 64)) * 255).astype(np.uint8))
    # sft source: 8 nm (coarser), image + segmentation.
    _write_h5(tmp_path / "fly_vol.h5", (rng.random((64, 64, 64)) * 255).astype(np.uint8))
    seg = rng.integers(0, 3, size=(64, 64, 64)).astype(np.int64)
    _write_h5(tmp_path / "fly_seg.h5", seg)
    return str(tmp_path)


def _make_dm(root, num_samples=2):
    return JointDataModule(
        data_root=root,
        patch_size=(32, 32, 32),
        pixel_size=(4.0, 4.0, 4.0),
        branches={
            "dapt": {"batch_size": 1, "sample_weight": 1.0, "volumes": [
                {"vol": "cosem_vol", "root": root, "native_resolution": [4, 4, 4]},
            ]},
            "sft": {"batch_size": 1, "sample_weight": 1.0, "volumes": [
                {"vol": "fly_vol", "seg": "fly_seg", "root": root,
                 "native_resolution": [8, 8, 8]},
            ]},
        },
        degrade={"zf_range": [2.0, 4.0], "missing_prob": 0.0},
        num_workers=0,
        persistent_workers=False,
        num_samples=num_samples,
        val_num_samples=2,
        val_batch_size=1,
    )


def test_joint_datamodule_batch_contract(synth_root):
    dm = _make_dm(synth_root)
    dm.setup()
    tasks = set()
    for batch in dm.train_dataloader():
        task = batch["task"][0] if isinstance(batch["task"], (list, tuple)) else batch["task"]
        tasks.add(task)
        # Image is always on the fine grid (32^3).
        assert tuple(batch["image"].shape) == (1, 1, 32, 32, 32)
        assert "recon_image" in batch
        if task == "sft":
            # 8 nm native label grid: round(32 * 4 / 8) = 16.
            assert tuple(batch["label"].shape[-3:]) == (16, 16, 16)
            assert tuple(batch["recon_image"].shape[-3:]) == (16, 16, 16)
        else:
            # dapt 4 nm == fine grid.
            assert tuple(batch["recon_image"].shape[-3:]) == (32, 32, 32)
    assert tasks == {"dapt", "sft"}   # round-robin yielded both branches


# ---------------------------------------------------------------------------
# Full train path: synthetic data -> JointDataModule -> JointModule -> Trainer
# ---------------------------------------------------------------------------

class _StubBackbone(nn.Module):
    def __init__(self, head_channels):
        super().__init__()
        self._backbone_loaded = False
        self.vae_encoder = None
        self.vae_input_pm1 = False
        self.conv = nn.Conv3d(1, head_channels, 1)

    def forward(self, x):
        return self.conv(x)


class _StubJointModule(JointModule):
    def _build_model(self, model_config):
        return _StubBackbone(model_config["head_channels"])


def test_joint_train_end_to_end(synth_root):
    dm = _make_dm(synth_root, num_samples=2)
    module = _StubJointModule(
        model_config={"pretrained": False},
        optimizer_config={"lr": 1e-4},
        loss_config={"weight_dapt": 1.0, "weight_sft": 1.0,
                     "seg": {"offsets": _OFFSETS, "n_pull": _N_PULL}},
        training_config={"mutex_watershed": {"backend": "cpu"}},
    )
    assert module.model.conv.out_channels == _HEAD
    trainer = pl.Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, dm)
