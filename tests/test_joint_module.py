"""Integration test for the JointModule training/eval loop (stub backbone).

Runs the *real* ``JointModule`` loop through a tiny CPU ``pl.Trainer``
(``fast_dev_run``) with a stub backbone in place of the 16B Cosmos-3 Nano, so
the task routing (``_prepare_targets``), the inherited ``training_step`` ->
``JointReconSegLoss`` path, and the sft metric pooling (``_accumulate_metrics``)
are all exercised without any HuggingFace download or GPU.
"""

import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from nanocosmos.losses import HEAD_CHANNELS
from nanocosmos.modules import JointModule


class _StubBackbone(nn.Module):
    """Stands in for the Cosmos-3 Nano wrapper: [B,1,D,H,W] -> [B,HEAD,D,H,W]."""

    def __init__(self, head_channels: int) -> None:
        super().__init__()
        self._backbone_loaded = False     # skips the post-init fallback freeze
        self.vae_encoder = None
        self.vae_input_pm1 = False
        self.conv = nn.Conv3d(1, head_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _StubJointModule(JointModule):
    def _build_model(self, model_config):
        return _StubBackbone(model_config["head_channels"])


class _BranchDataset(Dataset):
    """A few task-homogeneous samples on a small grid (> the offset reach)."""

    def __init__(self, task: str, n: int = 2, d: int = 32):
        self.task = task
        self.n = n
        self.d = d

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        d = self.d
        sample = {"task": self.task, "image": torch.rand(1, d, d, d)}
        if self.task == "dapt":
            sample["recon_image"] = torch.rand(1, d, d, d)
        else:  # sft
            sample["label"] = torch.randint(0, 3, (d, d, d), dtype=torch.long)
            sample["recon_image"] = torch.rand(1, d, d, d)
        return sample


class _BranchDataModule(pl.LightningDataModule):
    def __init__(self, task: str):
        super().__init__()
        self.task = task

    def train_dataloader(self):
        return DataLoader(_BranchDataset(self.task), batch_size=2)

    def val_dataloader(self):
        return DataLoader(_BranchDataset(self.task), batch_size=2)


def _make_module():
    return _StubJointModule(
        model_config={"variant": "Nano", "pretrained": False},
        optimizer_config={"lr": 1e-4},
        loss_config={"weight_dapt": 1.0, "weight_sft": 1.0, "seg": {}},
        training_config={"mutex_watershed": {"backend": "cpu"}},
    )


def test_head_width_derived_from_nested_seg_offsets():
    m = _make_module()
    assert m.model.conv.out_channels == HEAD_CHANNELS  # 14 aff + sem + raw


@pytest.mark.parametrize("task", ["dapt", "sft"])
def test_joint_module_fast_dev_run(task):
    m = _make_module()
    dm = _BranchDataModule(task)
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # Should run 1 train + 1 val batch through the real JointModule loop
    # (routing + JointReconSegLoss + sft metric pooling) without error.
    trainer.fit(m, dm)


def test_prepare_targets_routing():
    m = _make_module()
    # dapt: recon target, no labels.
    t = m._prepare_targets({"task": ["dapt", "dapt"], "recon_image": torch.rand(2, 1, 16, 16, 16)})
    assert t["task"] == "dapt" and "labels" not in t and "recon_image" in t
    # sft: labels + cached affinity target.
    t = m._prepare_targets({
        "task": ["sft", "sft"],
        "label": torch.randint(0, 3, (2, 32, 32, 32)),
        "recon_image": torch.rand(2, 1, 32, 32, 32),
    })
    assert t["task"] == "sft" and "labels" in t and "_cached_targets" in t
