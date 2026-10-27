"""Tests for the joint reconstruction + segmentation recipe.

Covers the two-branch :class:`Joint3DReconSegLoss` routing / shapes / gradient
flow and the :class:`RandResolutionDegraded` self-supervised (small-voxel ->
large-voxel) degradation transform.
"""

import pytest
import torch

from nanocosmos.losses import Joint3DReconSegLoss, HEAD_CHANNELS, N_AFF
from nanocosmos.losses.joint3d import SSL, SFT
from nanocosmos.transforms import RandResolutionDegraded, RandTransposeZYXd


# Spatial dims must exceed the default long-range affinity offsets
# (in-plane reach 27, z reach 4); the real network grid is 160^3.
def _head(b=2, d=32, h=32, w=32, requires_grad=True):
    return torch.randn(b, HEAD_CHANNELS, d, h, w, requires_grad=requires_grad)


def _labels(b=2, d=32, h=32, w=32):
    # A couple of instances + background so aff/sem targets are non-trivial.
    return torch.randint(0, 3, (b, d, h, w), dtype=torch.long)


# ---------------------------------------------------------------------------
# Joint3DReconSegLoss
# ---------------------------------------------------------------------------

def test_head_layout_delegated():
    loss = Joint3DReconSegLoss()
    assert loss.head_channels == HEAD_CHANNELS == N_AFF + 2
    assert loss.aff_slice.stop == N_AFF
    # The inner affinity loss must not also charge an input-recon term.
    assert loss.seg.weight_raw == 0.0


def test_ssl_branch_recon_only():
    loss = Joint3DReconSegLoss()
    head = _head()
    em = torch.rand(2, 1, 32, 32, 32)
    out = loss(head, {"task": SSL, "recon_image": em})
    assert "loss/recon" in out
    assert "loss/aff" not in out and "loss/sem" not in out
    out["loss"].backward()
    assert head.grad is not None and torch.isfinite(head.grad).all()


def test_ssl_recon_pools_raw_to_larger_voxel_target():
    # FIB-style SSL: clean EM target on a larger-voxel native grid than the
    # small-voxel prediction -> raw is pooled down to the target grid before L1.
    loss = Joint3DReconSegLoss()
    head = _head(d=32, h=32, w=32)
    em = torch.rand(2, 1, 16, 16, 16)      # larger-voxel native EM (e.g. FIB 8 nm)
    out = loss(head, {"task": SSL, "recon_image": em})
    assert "loss/recon" in out
    out["loss"].backward()
    assert torch.isfinite(head.grad).all()


def test_sft_branch_seg_only_no_recon():
    # SFT is seg-only when no recon target is supplied; labels on the small-
    # voxel grid -> pool factor 1 (no-op).
    loss = Joint3DReconSegLoss()
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
    loss = Joint3DReconSegLoss()
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
    loss = Joint3DReconSegLoss()
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
    pooled = Joint3DReconSegLoss._pool_to(head, (8, 32, 32))
    assert pooled.shape == (2, HEAD_CHANNELS, 8, 32, 32)
    # No-op when already at the target shape.
    same = Joint3DReconSegLoss._pool_to(head, (40, 64, 64))
    assert same is head


def test_mixed_task_batch_rejected():
    loss = Joint3DReconSegLoss()
    head = _head()
    with pytest.raises(ValueError):
        loss(head, {"task": [SSL, SFT], "recon_image": torch.rand(2, 1, 32, 32, 32)})


def test_homogeneous_task_list_accepted():
    loss = Joint3DReconSegLoss()
    head = _head()
    em = torch.rand(2, 1, 32, 32, 32)
    out = loss(head, {"task": [SSL, SSL], "recon_image": em})
    assert "loss/recon" in out


def test_branch_weights_scale_total():
    head = _head(requires_grad=False)
    em = torch.rand(2, 1, 32, 32, 32)
    base = Joint3DReconSegLoss()
    scaled = Joint3DReconSegLoss(weight_ssl=3.0)
    b = base(head, {"task": SSL, "recon_image": em})["loss"]
    s = scaled(head, {"task": SSL, "recon_image": em})["loss"]
    assert torch.allclose(s, 3.0 * b, atol=1e-5)


def test_canonical_loss_keys_stable():
    loss = Joint3DReconSegLoss()
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


def _axis_markers(shape):
    """``[1, D, H, W]`` with a 1 / 2 / 3 on the +z / +y / +x neighbour of the origin."""
    img = torch.zeros(1, *shape)
    img[0, 1, 0, 0] = 1.0
    img[0, 0, 1, 0] = 2.0
    img[0, 0, 0, 1] = 3.0
    return img


def _perm_from_markers(out: torch.Tensor):
    """Recover the spatial perm from :func:`_axis_markers` (``perm[dst] = src``)."""
    perm = [0, 0, 0]
    for dst in range(3):
        idx = [0, 0, 0]
        idx[dst] = 1
        perm[dst] = int(out[0, idx[0], idx[1], idx[2]].item()) - 1
    return tuple(perm)


def test_rand_transpose_zyx_moves_finest_axis_with_prompt():
    """A zyx perm reorients arrays and the prompt together.

    Native sizes are all distinct so the finest axis (4 nm, stored on x) is
    visible in the prompt.  Across draws it must occupy z, y, and x.
    """
    res = (40, 8, 4)
    t = RandTransposeZYXd(
        keys=["image", "recon_image"], native_resolution=res,
    )
    t.set_random_state(0)
    base = _axis_markers((6, 5, 4))
    recon = _axis_markers((6, 5, 4)) + 10.0
    prompt0 = "ssTEM · z40 y8 x4 nm · ssl"
    finest_on = set()
    seen_perms = set()
    for _ in range(80):
        out = t({
            "image": base.clone(),
            "recon_image": recon.clone(),
            "prompt": prompt0,
        })
        perm = _perm_from_markers(out["image"])
        seen_perms.add(perm)
        assert _perm_from_markers(out["recon_image"] - 10.0) == perm
        assert torch.equal(
            out["image"],
            base.permute(0, *(1 + p for p in perm)),
        )
        vals = [res[i] for i in perm]
        assert out["prompt"] == f"ssTEM · z{vals[0]} y{vals[1]} x{vals[2]} nm · ssl"
        finest_on.add(vals.index(4))
    assert seen_perms == {
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
    }
    # Finest native axis (4 nm) landed on every network axis.
    assert finest_on == {0, 1, 2}


def test_rand_transpose_zyx_identity_keeps_prompt():
    t = RandTransposeZYXd(keys=["image"], native_resolution=(8, 8, 8))
    t.randomize = lambda data=None: None
    t._perm = (0, 1, 2)
    img = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    prompt = "FIB-SEM · z8 y8 x8 nm · ssl"
    out = t({"image": img, "prompt": prompt})
    assert torch.equal(out["image"], img)
    assert out["prompt"] == prompt


def test_degrade_zf_controls_section_count():
    t = RandResolutionDegraded(prob=1.0, zf_range=(8.0, 8.0), z_sections_key="z_sections")
    t.set_random_state(1)
    out = t({"image": torch.rand(1, 32, 16, 16)})
    # zf == 8 on D == 32 -> ~4 sections.
    assert out["z_sections"] == 4
