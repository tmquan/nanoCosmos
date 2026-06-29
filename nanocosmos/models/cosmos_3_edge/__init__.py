"""Cosmos3-Edge (4B) **3D** model wrapper for volumetric connectomics.

Thin tier package over :mod:`nanocosmos.models.cosmos_3_common`.  Cosmos3-Edge
is announced but not yet released, so the wrapper warm-starts from the released
Cosmos3-Nano via structured reduction (depth + width truncation) -- see
:class:`Cosmos3EdgeWrapper` and
:func:`nanocosmos.models.cosmos_3_common.reduce.reduce_omni_transformer`.

References:
    - Cosmos 3 technical report: arXiv 2606.02800
    - https://github.com/nvidia/cosmos
"""

from nanocosmos.models.cosmos_3_edge.variants import _VARIANT_CONFIGS
from nanocosmos.models.cosmos_3_edge.wrapper import Cosmos3EdgeWrapper

__all__ = ["Cosmos3EdgeWrapper", "_VARIANT_CONFIGS"]
