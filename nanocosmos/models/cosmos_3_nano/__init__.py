"""Cosmos3-Nano (16B) **3D** model wrapper for volumetric connectomics.

Thin tier package over :mod:`nanocosmos.models.cosmos_3_common`: the shared
omni wrapper + all scaffolding live there; this package only pins the Nano
variant registry (real HF config numbers) and the public class name.

References:
    - HuggingFace: nvidia/Cosmos3-Nano
    - https://github.com/nvidia/cosmos
"""

from nanocosmos.models.cosmos_3_nano.variants import _VARIANT_CONFIGS
from nanocosmos.models.cosmos_3_nano.wrapper import (
    Cosmos3Nano3DWrapper,
    Cosmos3NanoWrapper,
)

__all__ = ["Cosmos3NanoWrapper", "Cosmos3Nano3DWrapper", "_VARIANT_CONFIGS"]
