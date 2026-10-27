"""Dense prediction head used by the Cosmos 3 decoder.

Two residual 3-D blocks refine the decoder feature map at a shared width,
then a 1×1 convolution projects to the affinity + sem + raw head. The
block layout matches the historical checkpoint contract: ``head.block.{N}``
indices depend on whether a width-matcher conv and dropout are present.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from monai.networks.blocks import UnetrBasicBlock
except ImportError as exc:  # pragma: no cover - monai is a hard dep
    raise ImportError(
        "The Cosmos 3 task head requires MONAI "
        "(monai.networks.blocks.UnetrBasicBlock). "
        "Install with `pip install monai>=1.3`."
    ) from exc


class TaskHead3D(nn.Module):
    """Residual dense-prediction head on top of the Cosmos 3 decoder.

    Architecture::

        (optional) Conv1x1:  in_channels -> refine_channels   # width matcher
        UnetrBasicBlock(res=True, kernel=3, norm="instance")  # refine
        UnetrBasicBlock(res=True, kernel=3, norm="instance")  # refine
        Dropout3d(dropout)
        Conv1x1:             refine_channels -> out_channels   # task head

    The width matcher is inserted only when ``in_channels != refine_channels``
    so the head can sit on a decoder whose width is wider than
    ``feature_size``.

    Args:
        in_channels: Channel count of the decoder feature map.
        out_channels: Task output width. Cosmos 3 wrappers use
            ``HEAD_CHANNELS`` (affinity + sem + raw).
        refine_channels: Width of the residual blocks. Defaults to
            ``in_channels``. The decoder passes ``feature_size`` so the
            cost stays independent of the VAE decoder width.
        dropout: Channel-wise dropout just before the final 1×1.
        norm_name: Normalisation inside the residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        refine_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm_name: str = "instance",
    ) -> None:
        super().__init__()
        refine = int(refine_channels) if refine_channels is not None else int(in_channels)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.refine_channels = refine

        # CHECKPOINT INVARIANT -- ``block`` is an ``nn.Sequential`` whose child
        # INDICES (and therefore the ``...head.block.{N}.*`` state_dict keys)
        # depend on two construction-time booleans:
        #   * width matcher present  <=>  in_channels != refine_channels
        #   * Dropout3d present      <=>  dropout > 0
        # Resulting index maps:
        #   matcher + dropout : 0=matcher 1,2=UnetrBasicBlock 3=Dropout 4=Conv(out)
        #   matcher, no drop  : 0=matcher 1,2=UnetrBasicBlock         3=Conv(out)
        #   no matcher, drop  : 0,1=UnetrBasicBlock 2=Dropout         3=Conv(out)
        #   neither           : 0,1=UnetrBasicBlock                   2=Conv(out)
        # A checkpoint built with feature_size=64 (head_in != refine) and
        # dropout>0 stores the final conv at ``head.block.4``. Changing
        # feature_size / head_in / dropout between train and resume renumbers
        # these keys and breaks a strict load. Do not reindex.
        layers: list[nn.Module] = []
        if self.in_channels != refine:
            layers.append(nn.Conv3d(self.in_channels, refine, kernel_size=1, bias=True))

        layers.extend([
            UnetrBasicBlock(
                spatial_dims=3,
                in_channels=refine,
                out_channels=refine,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            ),
            UnetrBasicBlock(
                spatial_dims=3,
                in_channels=refine,
                out_channels=refine,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            ),
        ])

        if dropout and dropout > 0.0:
            layers.append(nn.Dropout3d(float(dropout)))

        layers.append(nn.Conv3d(refine, self.out_channels, kernel_size=1, bias=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


__all__ = ["TaskHead3D"]
