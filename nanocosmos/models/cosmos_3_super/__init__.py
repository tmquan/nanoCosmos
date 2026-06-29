"""Cosmos3-Super (64B) **3D** model wrapper for volumetric connectomics.

Thin tier package over :mod:`nanocosmos.models.cosmos_3_common`: the shared
omni wrapper + all scaffolding live there; this package only pins the Super
variant registry (real HF config numbers) and the public class name.

References:
    - HuggingFace: nvidia/Cosmos3-Super
    - https://github.com/nvidia/cosmos
"""

from nanocosmos.models.cosmos_3_super.variants import _VARIANT_CONFIGS
from nanocosmos.models.cosmos_3_super.wrapper import Cosmos3SuperWrapper

__all__ = ["Cosmos3SuperWrapper", "_VARIANT_CONFIGS"]
