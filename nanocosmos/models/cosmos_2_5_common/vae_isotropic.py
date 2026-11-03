"""Make the released Wan2.2 VAE compress z and xy by 8.

The checkpoint is 16x in xy and 4x in z. Two edits meet in the middle:

* drop ``patch_size=2`` (an xy-only factor of 2). The 12-channel patch
  conv is averaged down to 3 channels.
* turn the remaining xy-only resample into a 3d resample, so that stage
  also halves z. Its spatial conv stays the pretrained one. The new
  temporal conv is a per-channel smoother (encoder) or a nearest repeat
  (decoder); that axis was not in the checkpoint.

The two existing 3d stages are unchanged. Result: 8x on z and xy, so a
4 nm cube is an isotropic latent. diffusers encodes depth in chunks of
``scale_factor_temporal``; that chunk size is repointed from the
hardcoded 4 to 8.
"""

from __future__ import annotations

import logging
import types

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PATCH = 2


def _fold_packed_in(weight: torch.Tensor, patch_size: int = _PATCH) -> torch.Tensor:
    """Average a ``[out, 3 * p * p, ...]`` patch-packed conv down to 3 inputs.

    Channel order matches diffusers ``patchify``: index
    ``(c * p + p_w) * p + p_h``.
    """
    p = int(patch_size)
    out, packed, *rest = weight.shape
    if packed != 3 * p * p:
        raise ValueError(
            f"expected {3 * p * p} packed input channels, got {packed}."
        )
    grouped = weight.contiguous().view(out, 3, p, p, *rest)
    return grouped.mean(dim=(2, 3))


def _fold_packed_out(weight: torch.Tensor, patch_size: int = _PATCH) -> torch.Tensor:
    """Average a ``[3 * p * p, in, ...]`` patch-packed conv down to 3 outputs."""
    p = int(patch_size)
    packed, *rest = weight.shape
    if packed != 3 * p * p:
        raise ValueError(
            f"expected {3 * p * p} packed output channels, got {packed}."
        )
    grouped = weight.contiguous().view(3, p, p, *rest)
    return grouped.mean(dim=(1, 2))


def _causal_padding(conv: nn.Module) -> int | tuple[int, int, int]:
    """Padding argument that rebuilds this ``WanCausalConv3d``."""
    pad = conv._padding
    pad_w = int(pad[0])
    pad_h = int(pad[2])
    pad_t = int(pad[4]) // 2
    if pad_w == pad_h == pad_t:
        return pad_t
    return (pad_t, pad_h, pad_w)


def _replace_causal_conv(
    module: nn.Module,
    attr: str,
    *,
    in_channels: int,
    out_channels: int,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> None:
    old = getattr(module, attr)
    new = type(old)(
        in_channels,
        out_channels,
        kernel_size=tuple(old.kernel_size),
        stride=tuple(old.stride),
        padding=_causal_padding(old),
    )
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(weight)
        if bias is not None and new.bias is not None:
            new.bias.copy_(bias)
    new.weight.requires_grad = old.weight.requires_grad
    if new.bias is not None and old.bias is not None:
        new.bias.requires_grad = old.bias.requires_grad
    setattr(module, attr, new)


def _init_temporal_downsample(conv: nn.Module) -> None:
    """Per-channel causal smoother. No cross-channel mix."""
    weight = conv.weight
    weight.zero_()
    channels = weight.shape[0]
    index = torch.arange(channels, device=weight.device)
    weight[index, index, 0, 0, 0] = 0.25
    weight[index, index, 1, 0, 0] = 0.5
    weight[index, index, 2, 0, 0] = 0.25
    if conv.bias is not None:
        conv.bias.zero_()


def _init_temporal_upsample(conv: nn.Module) -> None:
    """Copy the current frame into both temporal slots (nearest repeat)."""
    weight = conv.weight
    weight.zero_()
    channels = weight.shape[1]
    index = torch.arange(channels, device=weight.device)
    # Kernel tap 2 is the current frame under causal padding of 1.
    weight[index, index, 2, 0, 0] = 1
    weight[channels + index, index, 2, 0, 0] = 1
    if conv.bias is not None:
        conv.bias.zero_()


def _copy_resample_conv(src: nn.Module, dst: nn.Module) -> None:
    dst.weight.copy_(src.weight)
    if src.bias is not None and dst.bias is not None:
        dst.bias.copy_(src.bias)
    dst.weight.requires_grad = src.weight.requires_grad
    if dst.bias is not None and src.bias is not None:
        dst.bias.requires_grad = src.bias.requires_grad


def _promote_xy_downsample(block: nn.Module) -> None:
    """Make one xy-only residual downsample also halve z."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        AvgDown3D,
        WanResample,
    )

    old = block.downsampler
    shortcut = block.avg_shortcut
    if old is None or old.mode != "downsample2d":
        raise RuntimeError("xy-only encoder downsample is missing.")
    if int(shortcut.factor_s) != _PATCH or int(shortcut.factor_t) != 1:
        raise RuntimeError(
            "xy-only encoder shortcut is "
            f"factor_s={shortcut.factor_s}, factor_t={shortcut.factor_t}."
        )
    spatial = old.resample[1]
    new = WanResample(int(old.dim), mode="downsample3d")
    new = new.to(device=spatial.weight.device, dtype=spatial.weight.dtype)
    with torch.no_grad():
        _copy_resample_conv(spatial, new.resample[1])
        _init_temporal_downsample(new.time_conv)
    block.downsampler = new
    block.avg_shortcut = AvgDown3D(
        shortcut.in_channels,
        shortcut.out_channels,
        factor_t=2,
        factor_s=2,
    )


def _promote_xy_upsample(block: nn.Module) -> None:
    """Make one xy-only residual upsample also double z."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        DupUp3D,
        WanResample,
    )

    old = block.upsampler
    shortcut = block.avg_shortcut
    if old is None or old.mode != "upsample2d" or shortcut is None:
        raise RuntimeError("xy-only decoder upsample is missing.")
    if int(shortcut.factor_t) != 1 or int(shortcut.factor_s) != _PATCH:
        raise RuntimeError(
            "xy-only decoder shortcut is "
            f"factor_t={shortcut.factor_t}, factor_s={shortcut.factor_s}."
        )
    spatial = old.resample[1]
    new = WanResample(
        int(old.dim),
        mode="upsample3d",
        upsample_out_dim=int(spatial.out_channels),
    )
    new = new.to(device=spatial.weight.device, dtype=spatial.weight.dtype)
    with torch.no_grad():
        _copy_resample_conv(spatial, new.resample[1])
        _init_temporal_upsample(new.time_conv)
    block.upsampler = new
    block.avg_shortcut = DupUp3D(
        shortcut.in_channels,
        shortcut.out_channels,
        factor_t=2,
        factor_s=2,
    )


def _encode_temporal_factor(self, x: torch.Tensor) -> torch.Tensor:
    """``AutoencoderKLWan._encode`` with chunks of ``scale_factor_temporal``.

    The released method hardcodes 4, which matches 4x depth. After the
    xy-only stage also downsamples z, a later chunk has to be 8 frames
    so the three stride-2 causal convs emit one latent frame.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

    _, _, num_frame, height, width = x.shape
    self.clear_cache()
    patch = self.config.patch_size
    if patch is not None:
        x = patchify(x, patch_size=int(patch))
    if self.use_tiling and (
        width > self.tile_sample_min_width or height > self.tile_sample_min_height
    ):
        return self.tiled_encode(x)

    factor = int(self.config.scale_factor_temporal)
    steps = 1 + (num_frame - 1) // factor
    for i in range(steps):
        self._enc_conv_idx = [0]
        if i == 0:
            chunk = x[:, :, :1, :, :]
            out = self.encoder(
                chunk, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx,
            )
        else:
            start = 1 + factor * (i - 1)
            stop = 1 + factor * i
            chunk = x[:, :, start:stop, :, :]
            nxt = self.encoder(
                chunk, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, nxt], 2)
    encoded = self.quant_conv(out)
    self.clear_cache()
    return encoded


def _one(modules: list, kind: str) -> nn.Module:
    if len(modules) != 1:
        raise RuntimeError(
            f"expected one xy-only Wan {kind}, found {len(modules)}."
        )
    return modules[0]


def make_wan_compression_isotropic(vae: nn.Module) -> None:
    """Turn a released 16x-xy / 4x-z Wan VAE into 8x on z and xy.

    No-op when the VAE is already isotropic at 8x. Raises if the
    architecture is not the released Wan2.2 layout this edit knows.
    """
    cfg = vae.config
    patch = getattr(cfg, "patch_size", None)
    spatial = int(cfg.scale_factor_spatial)
    temporal = int(cfg.scale_factor_temporal)
    if patch in (None, 1) and spatial == 8 and temporal == 8:
        return
    if patch != _PATCH or spatial != 16 or temporal != 4:
        raise RuntimeError(
            "isotropic 8x Wan compression is implemented for the released "
            f"16x-xy / 4x-z VAE (patch_size=2). Got patch_size={patch}, "
            f"spatial={spatial}, temporal={temporal}."
        )

    encoder = vae.encoder
    conv_in = encoder.conv_in
    # Bias is one value per output channel. Only the packed input axis is folded.
    _replace_causal_conv(
        encoder,
        "conv_in",
        in_channels=3,
        out_channels=int(conv_in.out_channels),
        weight=_fold_packed_in(conv_in.weight),
        bias=None if conv_in.bias is None else conv_in.bias.detach().clone(),
    )

    down = _one(
        [
            block for block in encoder.down_blocks
            if getattr(getattr(block, "downsampler", None), "mode", None) == "downsample2d"
        ],
        "downsample",
    )
    _promote_xy_downsample(down)

    decoder = vae.decoder
    up = _one(
        [
            block for block in decoder.up_blocks
            if getattr(getattr(block, "upsampler", None), "mode", None) == "upsample2d"
        ],
        "upsample",
    )
    _promote_xy_upsample(up)

    conv_out = decoder.conv_out
    out_bias = (
        _fold_packed_out(conv_out.bias.view(-1, 1)).view(-1)
        if conv_out.bias is not None else None
    )
    _replace_causal_conv(
        decoder,
        "conv_out",
        in_channels=int(conv_out.in_channels),
        out_channels=3,
        weight=_fold_packed_out(conv_out.weight),
        bias=out_bias,
    )

    vae.register_to_config(
        patch_size=1,
        scale_factor_spatial=8,
        scale_factor_temporal=8,
        in_channels=3,
        out_channels=3,
    )
    vae.spatial_compression_ratio = 8
    vae._encode = types.MethodType(_encode_temporal_factor, vae)
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    vae._cached_conv_counts = {
        "decoder": sum(isinstance(m, WanCausalConv3d) for m in decoder.modules()),
        "encoder": sum(isinstance(m, WanCausalConv3d) for m in encoder.modules()),
    }
    logger.info(
        "Wan VAE compression is 8x on z and xy "
        "(dropped the xy-only patch; the xy-only resample now also "
        "halves z; the two original 3d stages are unchanged)."
    )
