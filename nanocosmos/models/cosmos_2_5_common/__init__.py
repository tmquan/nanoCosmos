"""Shared scaffolding for the Cosmos 3 backbone wrappers.

This package holds the pieces Nano, Edge, and Super share: primitive
layers, the random-init standalone DiT fallback, the rank-aware
HuggingFace snapshot downloader, the task head, the decoder adapter,
and :class:`_BaseCosmos25Wrapper`. Each tier package
(``cosmos_3_nano`` / ``cosmos_3_edge`` / ``cosmos_3_super``) only owns
its variant registry and any tier-specific load hook.

The module name is historical. Cosmos 3 is the only family that uses it.

Module layout::

    heads.py          -- TaskHead3D (affinity + sem + raw head)
    layers.py         -- shared primitives (_NORM, _PointwiseLinear, _adapt_to_rgb)
    variants.py       -- _VariantConfigBase dataclass
    hf_loader.py      -- rank-aware HuggingFace snapshot download
    standalone_dit.py -- _DiTBlock / _StandaloneDiT3D (random-init fallback)
    decoder.py        -- _FeatureProjector3D, _ProgressiveUpsampler3D,
                         _DecoderAdapter3D (VAE decoder + unified head)
    wrapper_base.py   -- _BaseCosmos25Wrapper (abstract public API)
"""

from nanocosmos.models.cosmos_2_5_common.decoder import (
    _DecoderAdapter3D,
    _FeatureProjector3D,
)
from nanocosmos.models.cosmos_2_5_common.hf_loader import _download_from_hf
from nanocosmos.models.cosmos_2_5_common.layers import (
    _NORM,
    _PointwiseLinear,
    _SPATIAL_DIMS,
    _adapt_to_rgb,
)
from nanocosmos.models.cosmos_2_5_common.standalone_dit import _StandaloneDiT3D
from nanocosmos.models.cosmos_2_5_common.variants import _VariantConfigBase
from nanocosmos.models.cosmos_2_5_common.wrapper_base import _BaseCosmos25Wrapper

__all__ = [
    "_BaseCosmos25Wrapper",
    "_DecoderAdapter3D",
    "_FeatureProjector3D",
    "_NORM",
    "_PointwiseLinear",
    "_SPATIAL_DIMS",
    "_StandaloneDiT3D",
    "_VariantConfigBase",
    "_adapt_to_rgb",
    "_download_from_hf",
]
