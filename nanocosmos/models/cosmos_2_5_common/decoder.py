"""Decoder-side modules shared by the Cosmos 3 wrappers.

The decoder hosts one unified task head that emits the canonical
``[B, HEAD_CHANNELS, D, H, W]`` affinity + sem + raw tensor consumed by
``nanocosmos.losses.AffinityFGLoss``.  The head applies no activation: it
emits raw logits for the affinity + sem channels and a linear value for
the raw-reconstruction channel.  Each consumer applies its own activation
(logit-stable BCE in the loss; sigmoid for metrics / Mutex Watershed /
TensorBoard).
"""

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nanocosmos.models.cosmos_2_5_common.heads import TaskHead3D
from nanocosmos.models.cosmos_2_5_common.layers import (
    _NORM,
    _PointwiseLinear,
)
from nanocosmos.losses import HEAD_CHANNELS

logger = logging.getLogger(__name__)


def unpatchify_wan(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Invert the Wan xy pixel-shuffle.

    ``x`` is ``[B, C * p * p, D, H, W]`` from ``conv_out``. The result is
    ``[B, C, D, H * p, W * p]``. Depth is not patched. ``p == 1`` returns
    ``x``. The layout matches diffusers ``unpatchify``.
    """
    p = int(patch_size)
    if p == 1:
        return x
    if p < 1:
        raise ValueError(f"Wan patch_size must be >= 1, got {p}.")
    if x.dim() != 5:
        raise ValueError(
            f"Wan conv_out must be [B, C, D, H, W], got {tuple(x.shape)}."
        )
    batch, packed, depth, height, width = x.shape
    area = p * p
    if packed % area != 0:
        raise ValueError(
            f"Wan conv_out channels {packed} are not divisible by "
            f"patch_size**2={area}."
        )
    channels = packed // area
    x = x.view(batch, channels, p, p, depth, height, width)
    # (p_w, p_h) packing: the second patch axis is the row offset.
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.view(batch, channels, depth, height * p, width * p)


def _wan_patch_size_for(out_channels: int, configured: Optional[int]) -> int:
    """Pixel-shuffle factor for a Wan ``conv_out``.

    The released Edge VAE uses ``patch_size=2`` and ``out_channels=12``
    (``3 * 2 * 2``). When the VAE config does not say, infer ``p`` from
    ``out_channels == 3 * p * p``.
    """
    if configured is not None:
        return int(configured)
    if out_channels > 0 and out_channels % 3 == 0:
        area = out_channels // 3
        root = int(round(area ** 0.5))
        if root >= 1 and root * root == area:
            return root
    return 1


class _FeatureProjector3D(nn.Module):
    """Fuse multi-layer DiT features into a 3-D spatial feature map."""

    def __init__(
        self,
        hidden_dim: int,
        num_feature_layers: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        total_in = hidden_dim * num_feature_layers
        self.proj = nn.Sequential(
            _PointwiseLinear(total_in, out_dim * 2),
            _NORM(out_dim * 2),
            nn.GELU(),
            _PointwiseLinear(out_dim * 2, out_dim),
        )

    def forward(
        self,
        features: List[torch.Tensor],
        d: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        spatial = [
            rearrange(f, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
            for f in features
        ]
        fused = torch.cat(spatial, dim=1)
        return self.proj(fused)


class _ProgressiveUpsampler3D(nn.Module):
    """Progressive 3-D upsampling (each stage doubles spatial dims)."""

    def __init__(self, in_dim: int, out_dim: int, num_stages: int) -> None:
        super().__init__()
        dims = self._interpolate_dims(in_dim, out_dim, num_stages + 1)
        layers: List[nn.Module] = []
        for i in range(num_stages):
            layers.append(nn.Sequential(
                nn.ConvTranspose3d(
                    dims[i], dims[i + 1],
                    kernel_size=4, stride=2, padding=1,
                ),
                _NORM(dims[i + 1]),
                nn.GELU(),
            ))
        self.stages = nn.ModuleList(layers)

    @staticmethod
    def _interpolate_dims(start: int, end: int, n: int) -> List[int]:
        if n <= 1:
            return [start]
        step = (end - start) / (n - 1)
        dims = [
            max(8, int(round((start + i * step) / 8)) * 8)
            for i in range(n)
        ]
        dims[0], dims[-1] = start, end
        return dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class _DecoderAdapter3D(nn.Module):
    """Reuses pretrained VAE decoder for the unified-head volumetric segmentation.

    Replaces the decoder's final output convolution with the unified
    affinity + sem + raw task head while preserving all pretrained upsampling
    weights.  The pretrained ``conv_out`` itself is kept on the side as
    :attr:`original_conv_out` (frozen) so the diagnostic ``conv_out``
    activation can still be emitted for visualisation
    (see :meth:`wan_reconstruct`). Joint runs log it as
    ``pred/wan_decoder``; the affinity ``ImageLogger`` logs it as
    ``true/wan_decoder``. The ``setattr(..., Identity())`` swap would
    otherwise drop the pretrained weights.

    Freeze policy when ``freeze_vae_decoder`` is set:
      - Decoder body (early / mid blocks): frozen
      - Last up-block + output norm: trainable
      - ``to_latent`` and the task head: trainable
      - Preserved ``original_conv_out``: frozen (diagnostic only)
    The body stays in the autograd graph, so the loss reaches the DiT
    through the decoder.

    CHECKPOINT INVARIANT: which submodules exist here is conditional. With a
    pretrained VAE (``pretrained=True``) this holds ``to_latent`` +
    ``original_conv_out`` and ``decoder_body`` is the Wan decoder; the
    standalone path (``pretrained=False``) drops ``to_latent`` and uses a
    ``_ProgressiveUpsampler3D`` body instead. ``highres_skip=True`` additionally
    adds ``skip_stem.*`` keys AND widens the head's first conv. With
    ``highres_skip=False``, ``head_in`` equals the Wan pre-``conv_out`` width
    (256 on the released Edge Wan2.2 config, ``decoder_base_dim=256``).
    Changing either flag between train and resume adds or removes keys and
    breaks a strict load.
    """

    def __init__(
        self,
        vae_decoder: Optional[nn.Module],
        latent_channels: int,
        feature_size: int,
        spatial_compression: int,
        temporal_compression: int,
        dropout: float = 0.0,
        freeze_vae_decoder: bool = False,
        head_channels: int = HEAD_CHANNELS,
        highres_skip: bool = False,
        skip_channels: int = 8,
        image_channels: int = 1,
        symmetrize_z: bool = False,
        wan_patch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._has_pretrained = vae_decoder is not None
        self.head_channels = int(head_channels)
        self.highres_skip = bool(highres_skip)
        # Average the causal Wan decoder body with a z-flipped decode so EM
        # depth is symmetric (same ``vae_symmetrize_z`` as the encoder).
        # Early/mid blocks may be frozen; the last up-block and output norm
        # stay trainable when ``freeze_vae_decoder`` is set, and gradients
        # still flow through the body. The random-init
        # ``_ProgressiveUpsampler3D`` fallback is not causal, so this is a
        # no-op there. See :meth:`_decode_body`.
        self._symmetrize_z = bool(symmetrize_z)
        # xy pixel-shuffle of the pretrained ``conv_out``. Set for real in
        # ``_replace_conv_out``; 1 means the diagnostic is already RGB.
        self._wan_patch_size_cfg = (
            None if wan_patch_size is None else int(wan_patch_size)
        )
        self._wan_patch_size = 1

        # Will be populated by ``_replace_conv_out`` when a pretrained
        # decoder is provided; stays ``None`` for the random-init
        # standalone path (where there is no pretrained reconstruction
        # to emit).
        self.original_conv_out: Optional[nn.Module] = None

        if vae_decoder is not None:
            self.to_latent = _PointwiseLinear(feature_size, latent_channels)
            self.decoder_body = vae_decoder
            self._hidden_ch = self._replace_conv_out()
            if freeze_vae_decoder:
                self._freeze_body()
        else:
            self.to_latent = None
            num_up_spatial = int(math.log2(spatial_compression))
            num_up_temporal = int(math.log2(temporal_compression))
            num_stages = max(num_up_spatial, num_up_temporal)
            self.decoder_body = _ProgressiveUpsampler3D(
                in_dim=feature_size, out_dim=feature_size,
                num_stages=num_stages,
            )
            self._hidden_ch = feature_size

        # High-resolution input skip.  The decoder features above are
        # bandwidth-limited by the VAE's spatial_compression (16x on Wan2.2): they
        # carry semantics but not the 1-voxel membranes the affinity head
        # needs.  A thin full-resolution conv stem on the raw input routes
        # those sharp edges *around* the latent bottleneck and is fused with
        # the decoded features just before the head, so the head can place
        # boundaries the latent cannot represent.  Kept thin (few channels)
        # because it runs at full input resolution.
        if self.highres_skip:
            c = int(skip_channels)
            self.skip_stem = nn.Sequential(
                nn.Conv3d(int(image_channels), c, kernel_size=3, padding=1),
                _NORM(c),
                nn.GELU(),
                nn.Conv3d(c, c, kernel_size=3, padding=1),
                _NORM(c),
                nn.GELU(),
            )
            head_in = self._hidden_ch + c
        else:
            self.skip_stem = None
            head_in = self._hidden_ch

        # Residual task head. Two UnetrBasicBlocks refine at
        # ``feature_size`` and a 1×1 conv emits the HEAD_CHANNELS field.
        # Refinement width stays at ``feature_size`` so the cost does not
        # track the VAE decoder's output width (``_hidden_ch``).
        self.head = TaskHead3D(
            in_channels=head_in,
            out_channels=self.head_channels,
            refine_channels=feature_size,
            dropout=dropout,
        )

    def _replace_conv_out(self) -> int:
        for attr in ("conv_out", "output_conv", "proj_out", "final_conv"):
            if hasattr(self.decoder_body, attr):
                final = getattr(self.decoder_body, attr)
                if hasattr(final, "in_channels"):
                    ch = final.in_channels
                elif hasattr(final, "weight") and final.weight.dim() >= 2:
                    ch = final.weight.shape[1]
                else:
                    continue
                # Preserve the pretrained final conv on the side BEFORE
                # overwriting it with Identity inside the body.  Without
                # this snapshot the weights become unreachable as soon
                # as ``setattr`` swaps in ``nn.Identity()``, and the
                # only way to recover them is a fresh HF download.
                # Keeping them frozen mirrors their pretrained-only
                # contract -- they are diagnostic, never optimised.
                self.original_conv_out = final
                for p in self.original_conv_out.parameters():
                    p.requires_grad = False
                if hasattr(final, "out_channels"):
                    out_ch = int(final.out_channels)
                else:
                    out_ch = int(final.weight.shape[0])
                self._wan_patch_size = _wan_patch_size_for(
                    out_ch, self._wan_patch_size_cfg,
                )
                setattr(self.decoder_body, attr, nn.Identity())
                logger.info(
                    "Replaced decoder.%s (hidden_ch=%d) with Identity; "
                    "preserved original conv_out as adapter.original_conv_out "
                    "(frozen, out_channels=%d, patch_size=%d) for the "
                    "diagnostic wan_decoder panel "
                    "(joint: pred/wan_decoder; affinity ImageLogger: "
                    "true/wan_decoder).",
                    attr, ch, out_ch, self._wan_patch_size,
                )
                return ch
        logger.warning(
            "Could not find decoder final conv; using latent_channels as "
            "hidden_ch.  The diagnostic wan_decoder panel will be unavailable."
        )
        return self.to_latent.linear.out_features

    def _freeze_body(self) -> None:
        for p in self.decoder_body.parameters():
            p.requires_grad = False
        for attr in ("up_blocks", "up"):
            if hasattr(self.decoder_body, attr):
                blocks = getattr(self.decoder_body, attr)
                if hasattr(blocks, "__len__") and len(blocks) > 0:
                    for p in blocks[-1].parameters():
                        p.requires_grad = True
                break
        for attr in ("conv_norm_out", "norm_out"):
            if hasattr(self.decoder_body, attr):
                for p in getattr(self.decoder_body, attr).parameters():
                    p.requires_grad = True
                break

    def _unfreeze_body(self) -> None:
        for p in self.decoder_body.parameters():
            p.requires_grad = True

    def forward(
        self,
        features: torch.Tensor,
        target_size: tuple,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        decoded = self._decode_body(features, target_size)
        if self.skip_stem is not None:
            if image is None:
                raise ValueError(
                    "_DecoderAdapter3D was built with highres_skip=True but "
                    "forward() received image=None; the wrapper must pass the "
                    "raw input volume so the skip stem can run."
                )
            if hasattr(image, "as_tensor"):
                image = image.as_tensor()
            if tuple(image.shape[-3:]) != tuple(target_size):
                image = F.interpolate(
                    image.float(), size=target_size,
                    mode="trilinear", align_corners=False,
                )
            skip = self.skip_stem(image.to(decoded.dtype))
            decoded = torch.cat([decoded, skip.to(decoded.dtype)], dim=1)
        # Raw logits / linear values -- no activation here (see module
        # docstring).  Consumers sigmoid the aff / sem channels themselves.
        return self.head(decoded)

    def _run_body(self, latent: torch.Tensor) -> torch.Tensor:
        """One pretrained ``decoder_body`` pass, unwrapping tuple/sample outputs."""
        decoded = self.decoder_body(latent)
        if isinstance(decoded, (tuple, list)):
            decoded = decoded[0]
        if hasattr(decoded, "sample"):
            decoded = decoded.sample
        return decoded

    def _decode_body(
        self, features: torch.Tensor, target_size: tuple, *, resize: bool = True,
    ) -> torch.Tensor:
        """Run features through ``decoder_body`` (with the swapped-out
        ``Identity`` final conv). When ``resize`` is set, trilinear-resize
        to ``target_size`` for the task head.

        Returns the post-norm-out, pre-conv-out feature volume that the
        unified head consumes.  Shared between :meth:`forward` and
        :meth:`wan_reconstruct` so both paths see identical body
        activations on the same call.

        When ``symmetrize_z`` is set, the Wan body is run twice -- forward
        and z-flipped -- and averaged, so the causal temporal decode is
        symmetric along the EM depth axis. ``to_latent`` and the head run
        once each. With ``freeze_vae_decoder``, early/mid blocks stay frozen
        and the last up-block plus output norm still train, so this decode
        stays on the loss path. Costs a second decode.
        """
        if self._has_pretrained:
            latent = self.to_latent(features)
            body_dtype = next(self.decoder_body.parameters()).dtype
            lat = latent.to(body_dtype)
            if self._symmetrize_z:
                fwd = self._run_body(lat)
                rev = torch.flip(self._run_body(torch.flip(lat, dims=(-3,))), dims=(-3,))
                decoded = 0.5 * (fwd + rev)
            else:
                decoded = self._run_body(lat)
            decoded = decoded.to(features.dtype)
        else:
            decoded = self.decoder_body(features)
        # The task head wants features on the 4 nm grid. The pretrained
        # conv_out does not: it emits a packed xy patch on the decoder's
        # native grid (half xy, full z). ``wan_reconstruct`` passes
        # ``resize=False`` and unpatchifies afterwards.
        if resize and decoded.shape[-3:] != target_size:
            decoded = F.interpolate(
                decoded, size=target_size, mode="trilinear", align_corners=False,
            )
        return decoded

    @torch.no_grad()
    def wan_reconstruct(
        self, features: torch.Tensor, target_size: tuple,
    ) -> Optional[torch.Tensor]:
        """Diagnostic Wan RGB decode of the task-head features.

        Mirrors :meth:`forward`'s body pass, then applies the original
        Wan ``conv_out`` (preserved at construction in
        :attr:`original_conv_out` before the adapter swapped it for
        ``Identity``) instead of the unified task head.

        Diagnostic only. ``conv_out`` runs on the decoder's native grid.
        On the released Edge Wan2.2 VAE that grid is full depth and half
        xy, and ``conv_out`` emits 12 channels (``3 * patch_size**2`` with
        ``patch_size=2``). Those channels are an xy pixel-shuffle, not an
        image. This path unpatchifies them to 3-channel RGB and clamps to
        ``[-1, 1]``, matching ``AutoencoderKLWan._decode``. Resizing the
        body to the full 4 nm cube *before* ``conv_out`` and then averaging
        the 12 channels paints a 2-pixel weave on xy and stripes on xz/yz.

        ``to_latent`` is randomly initialised, so the panel is the
        pretrained decoder on a random projection of the DiT features, not
        the training reconstruction (``pred/recon``).

        Returns:
            ``[B, 3, D, H, W]`` RGB in ``[-1, 1]`` at ``target_size``, or
            ``None`` if the wrapper was built without a pretrained VAE.
        """
        if not self._has_pretrained or self.original_conv_out is None:
            return None
        decoded = self._decode_body(features, target_size, resize=False)
        body_dtype = next(self.original_conv_out.parameters()).dtype
        packed = self.original_conv_out(decoded.to(body_dtype))
        rgb = unpatchify_wan(packed, self._wan_patch_size).clamp(-1.0, 1.0)
        spatial = tuple(int(s) for s in target_size)
        if tuple(rgb.shape[-3:]) != spatial:
            rgb = F.interpolate(
                rgb.float(), size=spatial, mode="trilinear", align_corners=False,
            )
        return rgb.to(features.dtype)


__all__ = [
    "_DecoderAdapter3D",
    "_FeatureProjector3D",
    "_ProgressiveUpsampler3D",
]
