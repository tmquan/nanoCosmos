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


def test_ssl_l1_pools_4nm_to_native():
    """Each native voxel is the mean of the 4 nm voxels in its bin. The target
    is the native crop recovered from the nearest-upsampled batch tensor."""
    loss = Joint3DReconSegLoss()
    fine = (8, 8, 4)
    sizes = [(4, 8, 2), (8, 4, 2)]
    natives = [torch.rand(1, 1, *s) for s in sizes]
    target = torch.cat([
        torch.nn.functional.interpolate(n, size=fine, mode="nearest") for n in natives
    ], 0)
    head = _head(b=2, d=fine[0], h=fine[1], w=fine[2])
    out = loss(head, {
        "task": SSL,
        "recon_image": target,
        "recon_native_size": torch.tensor(sizes),
    })
    raw = head[:, loss.raw_slice].detach().float()
    terms = []
    for i, size in enumerate(sizes):
        pred = torch.nn.functional.adaptive_avg_pool3d(raw[i:i + 1], size)
        terms.append(torch.nn.functional.l1_loss(pred, natives[i].float()))
    expected = torch.stack(terms).mean()
    assert torch.allclose(out["loss/recon"], expected)
    out["loss"].backward()
    assert torch.isfinite(head.grad).all()


def test_ssl_4nm_native_skips_resample():
    loss = Joint3DReconSegLoss()
    head = _head()
    em = torch.rand(2, 1, 32, 32, 32)
    out = loss(head, {
        "task": SSL,
        "recon_image": em,
        "recon_native_size": torch.tensor([[32, 32, 32], [32, 32, 32]]),
    })
    raw = head[:, loss.raw_slice].detach().float()
    assert torch.allclose(out["loss/recon"], torch.nn.functional.l1_loss(raw, em))


def test_ssl_recon_pools_raw_to_larger_voxel_target():
    # Coarser clean EM than the 4 nm prediction: raw is average-pooled onto
    # that grid before L1.
    loss = Joint3DReconSegLoss()
    head = _head(d=32, h=32, w=32)
    em = torch.rand(2, 1, 16, 16, 16)
    out = loss(head, {"task": SSL, "recon_image": em})
    raw = head[:, loss.raw_slice].detach().float()
    pred = torch.nn.functional.adaptive_avg_pool3d(raw, (16, 16, 16))
    assert torch.allclose(out["loss/recon"], torch.nn.functional.l1_loss(pred, em))
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
    assert set(keys) == {
        "loss", "loss/recon", "loss/aff", "loss/sem", "loss/rflow",
    }
    quiet = Joint3DReconSegLoss(weight_dif=0.0)
    assert "loss/rflow" not in quiet.canonical_loss_keys()


# ---------------------------------------------------------------------------
# RandResolutionDegraded
# ---------------------------------------------------------------------------

def test_degrade_downsamples_z_and_keeps_clean_target():
    t = RandResolutionDegraded(
        prob=1.0, zf_range=(8.0, 8.0), missing_prob=0.0, jitter_prob=0.0,
        noise_prob=0.0, contrast_prob=0.0, volume_contrast_prob=0.0,
        z_sections_key="z_sections",
    )
    t.set_random_state(0)
    img = torch.arange(32, dtype=torch.float32).view(1, 32, 1, 1).expand(1, 32, 16, 16).contiguous()
    out = t({"image": img.clone()})
    assert out["z_sections"] == 4
    assert tuple(out["image"].shape) == (1, 4, 16, 16)
    assert out["recon_image"].shape == img.shape
    assert torch.equal(out["recon_image"].float(), img)
    nearest = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(4, 16, 16), mode="nearest",
    ).squeeze(0)
    assert torch.equal(out["image"].float(), nearest)


def test_degrade_without_downsample_keeps_depth():
    t = RandResolutionDegraded(
        prob=1.0, downsample=False, zf_range=(8.0, 8.0), missing_prob=0.0,
        jitter_prob=0.0, noise_prob=0.0, contrast_prob=0.0,
        volume_contrast_prob=0.0, z_sections_key="z_sections",
    )
    t.set_random_state(0)
    img = torch.rand(1, 32, 16, 16)
    out = t({"image": img.clone()})
    assert out["z_sections"] == 32
    assert out["image"].shape == img.shape
    assert torch.equal(out["recon_image"].float(), img)
    assert torch.equal(out["image"].float(), img)


def test_degrade_loss_target_is_augmented_clean():
    """Augment is the target. Degrade and corrupt are derived from it and
    do not change it."""
    t = RandResolutionDegraded(
        prob=1.0, zf_range=(8.0, 8.0), jitter_prob=0.0, contrast_prob=0.0,
        volume_contrast_prob=1.0, volume_contrast_gamma=(2.0, 2.0),
        missing_prob=0.0, noise_prob=0.0,
    )
    t.set_random_state(0)
    img = torch.linspace(0.1, 0.9, 32).view(1, 32, 1, 1).expand(1, 32, 8, 8).contiguous()
    out = t({"image": img.clone()})
    augmented = img.clamp(0.0, 1.0).pow(2.0)
    assert torch.allclose(out["recon_image"].float(), augmented)
    nearest = torch.nn.functional.interpolate(
        augmented.unsqueeze(0), size=(4, 8, 8), mode="nearest",
    ).squeeze(0)
    assert torch.allclose(out["image"].float(), nearest)


def test_degrade_noise_does_not_enter_target():
    t = RandResolutionDegraded(
        prob=1.0, downsample=False, jitter_prob=0.0, contrast_prob=0.0,
        volume_contrast_prob=0.0, missing_prob=1.0, max_missing=2,
        noise_prob=1.0, noise_std=0.05,
    )
    t.set_random_state(0)
    img = torch.rand(1, 32, 8, 8).clamp(0.2, 0.8)
    out = t({"image": img.clone()})
    assert torch.equal(out["recon_image"].float(), img)
    assert out["image"].shape == img.shape
    assert not torch.allclose(out["image"].float(), img)


def test_degrade_missing_section_is_whole_native_section():
    """Missing runs after corrupt, so no noise or shift reaches the section."""
    t = RandResolutionDegraded(
        prob=1.0, downsample=False, jitter_prob=1.0, contrast_prob=1.0,
        volume_contrast_prob=0.0, missing_prob=1.0, max_missing=2,
        missing_fill="zero", noise_prob=1.0, noise_std=0.05,
    )
    img = torch.rand(1, 26, 16, 16).clamp(0.1, 0.9)
    for seed in range(5):
        t.set_random_state(seed)
        out = t({"image": img.clone()})["image"].float()
        blank = [z for z in range(26) if torch.count_nonzero(out[:, z]) == 0]
        assert 1 <= len(blank) <= 2
        partial = [z for z in range(26) if z not in blank and (out[:, z] == 0).all(dim=-1).any()]
        assert partial == []


def test_degrade_replicated_section_is_exact_copy():
    t = RandResolutionDegraded(
        prob=1.0, downsample=False, jitter_prob=1.0, contrast_prob=1.0,
        volume_contrast_prob=0.0, missing_prob=1.0, max_missing=1,
        missing_fill="replicate", noise_prob=1.0, noise_std=0.05,
    )
    t.set_random_state(3)
    img = torch.rand(1, 12, 16, 16)
    out = t({"image": img.clone()})["image"].float()
    dup = [z for z in range(11) if torch.equal(out[:, z], out[:, z + 1])]
    assert len(dup) == 1


def test_degrade_section_jitter_and_gamma_stay_out_of_target():
    """Per-section terms are Corrupt. The target only sees one gamma."""
    t = RandResolutionDegraded(
        prob=1.0, downsample=False, jitter_prob=1.0, max_jitter=2.0,
        contrast_prob=1.0, contrast_gamma=(0.5, 2.0),
        volume_contrast_prob=1.0, volume_contrast_gamma=(2.0, 2.0),
        missing_prob=0.0, noise_prob=0.0,
    )
    t.set_random_state(0)
    img = torch.rand(1, 16, 12, 12).clamp(0.1, 0.9)
    out = t({"image": img.clone()})
    assert torch.allclose(out["recon_image"].float(), img.pow(2.0))
    assert not torch.allclose(out["image"].float(), img.pow(2.0))


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
    # zf == 8 on D == 32 -> 4 sections, and the image stays at that depth.
    assert out["z_sections"] == 4
    assert out["image"].shape[1] == 4
    assert out["recon_image"].shape[1] == 32
