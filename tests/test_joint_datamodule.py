"""End-to-end test for the joint data path (synthetic volumes, stub backbone).

Writes tiny HDF5 volumes, builds :class:`Joint3DDataModule`, and (a) checks the
per-branch batch contract (task / fine-grid image / native label / recon
target), and (b) runs the *real* ``Joint3DModule`` loop over the datamodule via
``pl.Trainer(fast_dev_run=True)`` on CPU -- i.e. the full "can I train?" path
without the 16B backbone.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl

from nanocosmos.datamodules import Joint3DDataModule
from nanocosmos.modules import Joint3DModule
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
# Joint3DDataModule
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_root(tmp_path):
    rng = np.random.default_rng(0)
    # ssl source: 4 nm (== fine), image only.
    _write_h5(tmp_path / "cosem_vol.h5", (rng.random((64, 64, 64)) * 255).astype(np.uint8))
    # sft source: 8 nm (coarser), image + segmentation.
    _write_h5(tmp_path / "fly_vol.h5", (rng.random((64, 64, 64)) * 255).astype(np.uint8))
    seg = rng.integers(0, 3, size=(64, 64, 64)).astype(np.int64)
    _write_h5(tmp_path / "fly_seg.h5", seg)
    return str(tmp_path)


def _make_dm(root, num_samples=2, *, with_prompts=False, sft_convention=None):
    ssl_vol = {"vol": "cosem_vol", "root": root, "native_resolution": [4, 4, 4]}
    sft_vol = {
        "vol": "fly_vol", "seg": "fly_seg", "root": root,
        "native_resolution": [8, 8, 8],
    }
    if with_prompts:
        ssl_vol["imaging"] = "FIB-SEM"
        sft_vol["imaging"] = "FIB-SEM"
        if sft_convention is not None:
            sft_vol["label_convention"] = sft_convention
    return Joint3DDataModule(
        data_root=root,
        patch_size=(32, 32, 32),
        pixel_size=(4.0, 4.0, 4.0),
        branches={
            "ssl": {"batch_size": 1, "sample_weight": 1.0, "volumes": [ssl_vol]},
            "sft": {"batch_size": 1, "sample_weight": 1.0, "volumes": [sft_vol]},
        },
        degrade={"zf_range": [2.0, 4.0], "missing_prob": 0.0},
        find_boundaries=1.0 if sft_convention else 0.0,
        boundary_target="semantic",
        num_workers=0,
        persistent_workers=False,
        num_samples=num_samples,
        val_num_samples=2,
        val_batch_size=1,
    )


def test_joint_datamodule_ssl_only(synth_root):
    """Pretrain path: an ssl-only branch config never yields labels or sft."""
    dm = Joint3DDataModule(
        data_root=synth_root,
        patch_size=(32, 32, 32),
        pixel_size=(4.0, 4.0, 4.0),
        branches={
            "ssl": {
                "batch_size": 1,
                "sample_weight": 1.0,
                "volumes": [{
                    "vol": "cosem_vol", "root": synth_root,
                    "native_resolution": [4, 4, 4], "imaging": "FIB-SEM",
                }],
            },
        },
        degrade={"zf_range": [2.0, 4.0], "missing_prob": 0.0},
        val_volumes=[{
            "vol": "cosem_vol", "root": synth_root, "task": "ssl",
            "native_resolution": [4, 4, 4], "imaging": "FIB-SEM",
        }],
        num_workers=0,
        persistent_workers=False,
        num_samples=2,
        val_num_samples=2,
        val_batch_size=1,
    )
    dm.setup()
    assert set(dm.branches) == {"ssl"}
    for loader in (dm.train_dataloader(), dm.val_dataloader()):
        for batch in loader:
            task = batch["task"][0] if isinstance(batch["task"], (list, tuple)) else batch["task"]
            assert task == "ssl"
            assert "label" not in batch
            assert "sem_label" not in batch
            assert tuple(batch["image"].shape) == (1, 1, 32, 32, 32)
            assert tuple(batch["recon_image"].shape[-3:]) == (32, 32, 32)


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
            # The recon target is nearest-upsampled to the 4 nm cube.
            assert tuple(batch["label"].shape[-3:]) == (16, 16, 16)
            assert tuple(batch["recon_image"].shape[-3:]) == (32, 32, 32)
            assert tuple(batch["recon_native_size"][0].tolist()) == (16, 16, 16)
        else:
            # ssl 4 nm == fine grid.
            assert tuple(batch["recon_image"].shape[-3:]) == (32, 32, 32)
    assert tasks == {"ssl", "sft"}   # round-robin yielded both branches


# ---------------------------------------------------------------------------
# Full train path: synthetic data -> Joint3DDataModule -> Joint3DModule -> Trainer
# ---------------------------------------------------------------------------

class _StubBackbone(nn.Module):
    def __init__(self, head_channels):
        super().__init__()
        self._backbone_loaded = False
        self.vae_encoder = None
        self.vae_input_pm1 = False
        self.conv = nn.Conv3d(1, head_channels, 1)
        self.prompts_seen = []

    def forward(self, x, prompts=None):
        self.prompts_seen.append(prompts)
        return self.conv(x)


class _StubJointModule(Joint3DModule):
    def _build_model(self, model_config):
        return _StubBackbone(model_config["head_channels"])


def test_joint_datamodule_condition_prompts(synth_root):
    dm = _make_dm(
        synth_root, with_prompts=True, sft_convention="filled",
    )
    dm.setup()
    seen = {}
    for batch in dm.train_dataloader():
        task = batch["task"][0] if isinstance(batch["task"], (list, tuple)) else batch["task"]
        assert "prompt" in batch
        prompt = batch["prompt"][0]
        seen[task] = prompt
    assert seen["ssl"] == "FIB-SEM · ssl"
    assert seen["sft"] == "FIB-SEM · filled"


def test_gapped_skips_sem_label_erosion(synth_root):
    dm = _make_dm(
        synth_root, with_prompts=True, sft_convention="gapped",
    )
    dm.setup()
    for batch in dm.train_dataloader():
        task = batch["task"][0] if isinstance(batch["task"], (list, tuple)) else batch["task"]
        if task == "sft":
            assert "sem_label" not in batch
            assert batch["prompt"][0] == "FIB-SEM · gapped"


def test_joint_train_end_to_end(synth_root):
    dm = _make_dm(synth_root, num_samples=2, with_prompts=True, sft_convention="filled")
    module = _StubJointModule(
        model_config={"pretrained": False},
        optimizer_config={"lr": 1e-4},
        loss_config={"weight_ssl": 1.0, "weight_sft": 1.0,
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
    # Batches carry a prompt for logging; the network only ever gets null text.
    assert module.model.prompts_seen and set(module.model.prompts_seen) == {None}


def test_image_logger_writes_prompt_text(synth_root):
    from nanocosmos.callbacks.tensorboard.joint3d_logger import Joint3DImageLogger

    class _RecordingTB:
        def __init__(self):
            self.text = {}

        def add_text(self, tag, text, global_step=None):
            self.text[tag] = text

        def __getattr__(self, name):
            return lambda *a, **k: None

    tb = _RecordingTB()
    logger_cb = Joint3DImageLogger(spatial_dims=3)
    logger_cb._get_tb = lambda trainer: tb
    dm = _make_dm(synth_root, num_samples=2, with_prompts=True, sft_convention="filled")
    module = _StubJointModule(
        model_config={"pretrained": False},
        optimizer_config={"lr": 1e-4},
        loss_config={"weight_ssl": 1.0, "weight_sft": 1.0,
                     "seg": {"offsets": _OFFSETS, "n_pull": _N_PULL}},
        training_config={"mutex_watershed": {"backend": "cpu"}},
    )
    trainer = pl.Trainer(
        max_epochs=1, limit_train_batches=2, limit_val_batches=0,
        accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[logger_cb],
    )
    trainer.fit(module, dm)
    prompts = {k: v for k, v in tb.text.items() if k.endswith("/prompt")}
    assert prompts.get("train/automatic/ssl/prompt") == "FIB-SEM · ssl"
    assert set(module.model.prompts_seen) == {None}


# ---------------------------------------------------------------------------
# rand_transpose_zyx: cubic FOV, finest native axis moves
# ---------------------------------------------------------------------------

def _recon_for(fine: int, fine_nm: float, zyx):
    return tuple(
        max(1, int(round(fine * fine_nm / max(float(r), fine_nm)))) for r in zyx
    )


def test_rand_transpose_zyx_requires_cubic_fov():
    with pytest.raises(ValueError, match="cubic patch_size"):
        Joint3DDataModule(
            patch_size=(32, 16, 16),
            pixel_size=(4.0, 4.0, 4.0),
            rand_transpose_zyx=True,
        )
    with pytest.raises(ValueError, match="cubic pixel_size"):
        Joint3DDataModule(
            patch_size=(32, 32, 32),
            pixel_size=(4.0, 4.0, 8.0),
            rand_transpose_zyx=True,
        )


def test_anisotropic_ssl_keeps_batch_size(synth_root):
    """Native recon is nearest-upsampled to the cube, so batch 2 stacks."""
    fine = 32
    dm = Joint3DDataModule(
        data_root=synth_root,
        patch_size=(fine, fine, fine),
        pixel_size=(4.0, 4.0, 4.0),
        rand_transpose_zyx=True,
        branches={
            "ssl": {
                "batch_size": 2,
                "sample_weight": 1.0,
                "volumes": [{
                    "vol": "cosem_vol", "root": synth_root,
                    "native_resolution": [16, 4, 4], "imaging": "ssTEM",
                }],
            },
        },
        degrade={
            "zf_range": [2.0, 2.0], "missing_prob": 0.0, "jitter_prob": 0.0,
            "noise_prob": 0.0, "contrast_prob": 0.0, "volume_contrast_prob": 0.0,
        },
        num_workers=0,
        persistent_workers=False,
        num_samples=4,
    )
    dm.setup()
    assert dm._train_groups[-1][2] == 2
    assert dm._group_transform("ssl", (16, 4, 4)).transforms[1].downsample is False
    assert dm._group_transform("ssl", (4, 4, 4)).transforms[1].downsample is True
    batch = next(iter(dm.train_dataloader()))
    assert tuple(batch["image"].shape) == (2, 1, fine, fine, fine)
    assert tuple(batch["recon_image"].shape) == (2, 1, fine, fine, fine)
    for i in range(2):
        nat = tuple(int(v) for v in batch["recon_native_size"][i].tolist())
        assert sorted(nat) == [8, 32, 32]


def test_rand_transpose_zyx_augments_finest_axis(synth_root):
    """Train crops reorient zyx; the 4 nm axes move, val stays native."""
    native = (16, 4, 4)
    fine = 32
    dm = Joint3DDataModule(
        data_root=synth_root,
        patch_size=(fine, fine, fine),
        pixel_size=(4.0, 4.0, 4.0),
        rand_transpose_zyx=True,
        branches={
            "ssl": {
                "batch_size": 1,
                "sample_weight": 1.0,
                "volumes": [{
                    "vol": "cosem_vol", "root": synth_root,
                    "native_resolution": list(native), "imaging": "ssTEM",
                }],
            },
        },
        degrade={
            "zf_range": [2.0, 2.0], "missing_prob": 0.0, "jitter_prob": 0.0,
            "noise_prob": 0.0, "contrast_prob": 0.0, "volume_contrast_prob": 0.0,
            "fine_axis_prob": 0.0,
        },
        val_volumes=[{
            "vol": "cosem_vol", "root": synth_root, "task": "ssl",
            "native_resolution": list(native), "imaging": "ssTEM",
        }],
        num_workers=0,
        persistent_workers=False,
        num_samples=36,
        val_num_samples=4,
        val_batch_size=1,
    )
    dm.setup()
    finest_on_z = False
    coarse_on_z = False
    for batch in dm.train_dataloader():
        assert tuple(batch["image"].shape) == (1, 1, fine, fine, fine)
        assert batch["prompt"][0] == "ssTEM · ssl"
        assert tuple(batch["recon_image"].shape[-3:]) == (fine, fine, fine)
        nat = tuple(batch["recon_native_size"][0].tolist())
        assert sorted(nat) == sorted(_recon_for(fine, 4.0, native))
        finest_on_z = finest_on_z or nat[0] == fine
        coarse_on_z = coarse_on_z or nat[0] == 8
    assert finest_on_z and coarse_on_z

    for batch in dm.val_dataloader():
        assert batch["prompt"][0] == "ssTEM · ssl"
        assert tuple(batch["image"].shape) == (1, 1, fine, fine, fine)
        assert tuple(batch["recon_image"].shape[-3:]) == (fine, fine, fine)
        assert tuple(batch["recon_native_size"][0].tolist()) == (8, 32, 32)
