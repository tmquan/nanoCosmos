"""Tests for the joint reconstruction + segmentation recipe.

Covers the two-branch :class:`JointReconSegLoss` routing / shapes / gradient
flow and the :class:`RandResolutionDegraded` self-supervised (small-voxel ->
large-voxel) degradation transform.
"""

import pytest
import torch

from nanocosmos.losses import JointReconSegLoss, HEAD_CHANNELS, N_AFF
from nanocosmos.losses.joint import DAPT, SFT
from nanocosmos.transforms import RandResolutionDegraded


# Spatial dims must exceed the default long-range affinity offsets
# (in-plane reach 27, z reach 4); the real network grid is 160^3.
def _head(b=2, d=32, h=32, w=32, requires_grad=True):
    return torch.randn(b, HEAD_CHANNELS, d, h, w, requires_grad=requires_grad)


def _labels(b=2, d=32, h=32, w=32):
    # A couple of instances + background so aff/sem targets are non-trivial.
    return torch.randint(0, 3, (b, d, h, w), dtype=torch.long)


# ---------------------------------------------------------------------------
# JointReconSegLoss
# ---------------------------------------------------------------------------

def test_head_layout_delegated():
    loss = JointReconSegLoss()
    assert loss.head_channels == HEAD_CHANNELS == N_AFF + 2
    assert loss.aff_slice.stop == N_AFF
    # The inner affinity loss must not also charge an input-recon term.
    assert loss.seg.weight_raw == 0.0


def test_dapt_branch_recon_only():
    loss = JointReconSegLoss()
    head = _head()
    em = torch.rand(2, 1, 32, 32, 32)
    out = loss(head, {"task": DAPT, "recon_image": em})
    assert "loss/recon" in out
    assert "loss/aff" not in out and "loss/sem" not in out
    out["loss"].backward()
    assert head.grad is not None and torch.isfinite(head.grad).all()


def test_dapt_recon_pools_raw_to_larger_voxel_target():
    # FIB-style DAPT: clean EM target on a larger-voxel native grid than the
    # small-voxel prediction -> raw is pooled down to the target grid before L1.
    loss = JointReconSegLoss()
    head = _head(d=32, h=32, w=32)
    em = torch.rand(2, 1, 16, 16, 16)      # larger-voxel native EM (e.g. FIB 8 nm)
    out = loss(head, {"task": DAPT, "recon_image": em})
    assert "loss/recon" in out
    out["loss"].backward()
    assert torch.isfinite(head.grad).all()


def test_sft_branch_seg_only_no_recon():
    # SFT is seg-only when no recon target is supplied; labels on the small-
    # voxel grid -> pool factor 1 (no-op).
    loss = JointReconSegLoss()
    head = _head()
    labels = _labels()
    cached = loss.build_targets(labels)
    out = loss(head, {"task": SFT, "labels": labels, "_cached_targets": cached})
    assert "loss/aff" in out and "loss/sem" in out
    assert "loss/recon" not in out          # no recon target supplied
    out["loss"].backward()
    assert torch.isfinite(head.grad).all()


def test_sft_pools_small_voxel_head_to_label_grid():
    # Large-voxel labels (z and xy both coarser) -> the small-voxel head is
    # pooled to the label shape, factor derived from the labels.
    loss = JointReconSegLoss()
    head = _head(d=40, h=64, w=64)          # small-voxel grid
    labels = _labels(d=8, h=32, w=32)       # larger voxel on ALL three axes
    cached = loss.build_targets(labels)
    out = loss(head, {"task": SFT, "labels": labels, "_cached_targets": cached})
    assert "loss/aff" in out and "loss/sem" in out
    assert "loss/recon" not in out
    out["loss"].backward()
    assert torch.isfinite(head.grad).all()


def test_sft_raw_data_consistency_when_recon_image_present():
    # The 32-ch head always carries `raw`; on sft we also supervise it by
    # pooling the small-voxel raw down to the ORIGINAL large-voxel EM.
    loss = JointReconSegLoss()
    head = _head(d=40, h=64, w=64)            # small-voxel grid
    labels = _labels(d=8, h=32, w=32)         # native (large-voxel) label grid
    native_em = torch.rand(2, 1, 8, 32, 32)   # original large-voxel EM
    cached = loss.build_targets(labels)
    out = loss(head, {
        "task": SFT, "labels": labels, "recon_image": native_em,
        "_cached_targets": cached,
    })
    assert "loss/aff" in out and "loss/sem" in out
    assert "loss/recon" in out                # raw head supervised on sft too
    out["loss"].backward()
    assert torch.isfinite(head.grad).all()


def test_pool_to_shape_all_axes():
    head = _head(d=40, h=64, w=64)
    pooled = JointReconSegLoss._pool_to(head, (8, 32, 32))
    assert pooled.shape == (2, HEAD_CHANNELS, 8, 32, 32)
    # No-op when already at the target shape.
    same = JointReconSegLoss._pool_to(head, (40, 64, 64))
    assert same is head


def test_mixed_task_batch_rejected():
    loss = JointReconSegLoss()
    head = _head()
    with pytest.raises(ValueError):
        loss(head, {"task": [DAPT, SFT], "recon_image": torch.rand(2, 1, 32, 32, 32)})


def test_homogeneous_task_list_accepted():
    loss = JointReconSegLoss()
    head = _head()
    em = torch.rand(2, 1, 32, 32, 32)
    out = loss(head, {"task": [DAPT, DAPT], "recon_image": em})
    assert "loss/recon" in out


def test_branch_weights_scale_total():
    head = _head(requires_grad=False)
    em = torch.rand(2, 1, 32, 32, 32)
    base = JointReconSegLoss()
    scaled = JointReconSegLoss(weight_dapt=3.0)
    b = base(head, {"task": DAPT, "recon_image": em})["loss"]
    s = scaled(head, {"task": DAPT, "recon_image": em})["loss"]
    assert torch.allclose(s, 3.0 * b, atol=1e-5)


def test_canonical_loss_keys_stable():
    loss = JointReconSegLoss()
    keys = loss.canonical_loss_keys()
    assert keys[0] == "loss"
    assert set(keys) == {"loss", "loss/recon", "loss/aff", "loss/sem"}


# ---------------------------------------------------------------------------
# RandResolutionDegraded
# ---------------------------------------------------------------------------

def test_degrade_preserves_shape_and_writes_recon_target():
    t = RandResolutionDegraded(prob=1.0, z_sections_key="z_sections")
    t.set_random_state(0)
    img = torch.rand(1, 32, 16, 16)
    out = t({"image": img})
    assert out["image"].shape == img.shape
    assert out["recon_image"].shape == img.shape
    # The recon target is the pristine input; the degraded image differs.
    assert torch.allclose(out["recon_image"].float(), img)
    assert not torch.allclose(out["image"].float(), img)
    # Coarse section count recorded and strictly fewer than D.
    assert 1 <= out["z_sections"] < 32


def test_degrade_passthrough_still_writes_target():
    t = RandResolutionDegraded(prob=0.0)
    t.set_random_state(0)
    img = torch.rand(1, 32, 16, 16)
    out = t({"image": img})
    assert "recon_image" in out
    assert torch.allclose(out["image"].float(), img)


def test_degrade_zf_controls_section_count():
    t = RandResolutionDegraded(prob=1.0, zf_range=(8.0, 8.0), z_sections_key="z_sections")
    t.set_random_state(1)
    out = t({"image": torch.rand(1, 32, 16, 16)})
    # zf == 8 on D == 32 -> ~4 sections.
    assert out["z_sections"] == 4
