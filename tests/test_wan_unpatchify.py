"""Wan diagnostic decode: unpatchify the packed conv_out before display.

Averaging the 12 packed channels paints a 2-pixel weave on xy. The
diagnostic has to unpatchify (patch along H and W only) and clamp, the
same way ``AutoencoderKLWan._decode`` does.
"""

import torch
import torch.nn as nn

from nanocosmos.models.cosmos_2_5_common.decoder import (
    _DecoderAdapter3D,
    _wan_patch_size_for,
    unpatchify_wan,
)


def _patchify_wan(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """diffusers ``patchify``, kept here so the round-trip is explicit."""
    p = int(patch_size)
    batch, channels, frames, height, width = x.shape
    x = x.view(
        batch, channels, frames, height // p, p, width // p, p,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(
        batch, channels * p * p, frames, height // p, width // p,
    )


def test_unpatchify_matches_diffusers() -> None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    packed = torch.randn(2, 12, 4, 5, 6)
    assert torch.equal(unpatchify_wan(packed, 2), unpatchify(packed, 2))
    assert torch.equal(unpatchify_wan(packed, 1), packed)


def test_unpatchify_inverts_patchify() -> None:
    image = torch.randn(2, 3, 4, 8, 6)
    packed = _patchify_wan(image, 2)
    assert packed.shape == (2, 12, 4, 4, 3)
    assert torch.allclose(unpatchify_wan(packed, 2), image)


def test_patch_size_inferred_from_twelve_channels() -> None:
    assert _wan_patch_size_for(12, None) == 2
    assert _wan_patch_size_for(3, None) == 1
    assert _wan_patch_size_for(12, 2) == 2


class _NativeGridDecoder(nn.Module):
    """Body that keeps the latent grid. ``conv_out`` is the packed head."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.conv_out = nn.Conv3d(2, 12, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def test_wan_reconstruct_unpatchifies_native_grid() -> None:
    adapter = _DecoderAdapter3D(
        vae_decoder=_NativeGridDecoder(),
        latent_channels=2,
        feature_size=4,
        spatial_compression=16,
        temporal_compression=4,
        wan_patch_size=2,
    )
    assert adapter._wan_patch_size == 2
    conv = adapter.original_conv_out
    p = 2
    bias = torch.zeros(12)
    for c in range(3):
        for p_w in range(p):
            for p_h in range(p):
                index = (c * p + p_w) * p + p_h
                bias[index] = 0.1 * (index + 1)
    bias[0] = 5.0  # clamp to 1
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.copy_(bias)

    features = torch.randn(1, 4, 2, 3, 3)
    rgb = adapter.wan_reconstruct(features, target_size=(2, 6, 6))
    assert rgb.shape == (1, 3, 2, 6, 6)
    assert rgb.min() >= -1.0 and rgb.max() <= 1.0
    # index 0 is color 0, row offset 0, col offset 0, and was clamped.
    assert torch.equal(rgb[0, 0, :, 0::2, 0::2], torch.ones(2, 3, 3))
    gray = rgb[0, :, 0, 0:2, 0:2].mean(dim=0)
    # The four xy subpixels are different values. A channel-mean of the
    # packed 12 would have been one constant, which is the weave bug.
    assert gray.std() > 0
    assert not torch.allclose(gray, torch.full_like(gray, float(bias.mean())))
