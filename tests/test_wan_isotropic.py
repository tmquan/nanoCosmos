"""Released Wan2.2 is 16x in xy and 4x in z. Load makes both axes 8x."""

import torch

from nanocosmos.models.cosmos_2_5_common.vae_isotropic import (
    _fold_packed_in,
    _fold_packed_out,
    make_wan_compression_isotropic,
)


def test_fold_averages_xy_slots() -> None:
    weight = torch.zeros(2, 12, 1, 1, 1)
    # index = (c * 2 + p_w) * 2 + p_h. Color 0's four slots are 1, 3, 5, 7.
    for c, base in ((0, 1.0), (1, 10.0), (2, 100.0)):
        for p_w in range(2):
            for p_h in range(2):
                weight[:, (c * 2 + p_w) * 2 + p_h] = base + p_w + p_h
    folded = _fold_packed_in(weight)
    assert folded.shape == (2, 3, 1, 1, 1)
    # mean of base, base+1, base+1, base+2 = base + 1
    assert torch.allclose(folded[:, 0], torch.full((2, 1, 1, 1), 2.0))
    assert torch.allclose(folded[:, 2], torch.full((2, 1, 1, 1), 101.0))

    packed_out = weight.transpose(0, 1).contiguous()  # [12, 2, 1, 1, 1]
    folded_out = _fold_packed_out(packed_out)
    assert folded_out.shape == (3, 2, 1, 1, 1)


def _tiny_wan():
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan(
        base_dim=32,
        decoder_base_dim=32,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        is_residual=True,
        in_channels=12,
        out_channels=12,
        patch_size=2,
        scale_factor_spatial=16,
        scale_factor_temporal=4,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )


def test_isotropic_vae_compresses_by_8() -> None:
    torch.manual_seed(0)
    vae = _tiny_wan().eval()
    # 1 + 8 frames so the causal encoder covers the volume at factor 4 and 8.
    volume = torch.randn(1, 3, 9, 32, 32)
    with torch.no_grad():
        before = vae.encode(volume).latent_dist.mode()
    assert before.shape[-3:] == (3, 2, 2)  # z chunks of 4, xy 32 / 16

    make_wan_compression_isotropic(vae)
    assert vae.config.patch_size == 1
    assert vae.config.scale_factor_spatial == 8
    assert vae.config.scale_factor_temporal == 8
    assert vae.encoder.conv_in.in_channels == 3
    assert vae.decoder.conv_out.out_channels == 3
    assert not any(
        getattr(getattr(block, "downsampler", None), "mode", None) == "downsample2d"
        for block in vae.encoder.down_blocks
    )
    promoted = [
        block for block in vae.encoder.down_blocks
        if getattr(getattr(block, "downsampler", None), "mode", None) == "downsample3d"
    ]
    assert len(promoted) == 3

    with torch.no_grad():
        after = vae.encode(volume).latent_dist.mode()
        decoded = vae.decode(after).sample
    assert after.shape[-3:] == (2, 4, 4)  # z chunks of 8, xy 32 / 8
    assert decoded.shape[1] == 3
    assert decoded.shape[-2:] == (32, 32)

    make_wan_compression_isotropic(vae)  # already 8x: no-op
