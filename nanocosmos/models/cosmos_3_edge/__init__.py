"""Cosmos3-Edge (4B) **3D** model wrapper for volumetric connectomics.

Thin tier package over :mod:`nanocosmos.models.cosmos_3_common`.  Cosmos3-Edge
has released weights (``nvidia/Cosmos3-Edge``) and the wrapper loads them
directly.  A legacy ``reduce_from_parent`` path can still truncate a parent
transformer; the shipped variant leaves it off.  See
:class:`Cosmos3EdgeWrapper`.

References:
    - Cosmos 3 technical report: arXiv 2606.02800
    - https://github.com/nvidia/cosmos
"""

from nanocosmos.models.cosmos_3_edge.variants import _VARIANT_CONFIGS
from nanocosmos.models.cosmos_3_edge.wrapper import Cosmos3EdgeWrapper

__all__ = ["Cosmos3EdgeWrapper", "_VARIANT_CONFIGS"]
