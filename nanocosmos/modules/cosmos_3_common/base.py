"""Cosmos 3 family Lightning-module base (soft re-export).

The training / evaluation / logging loop for every Cosmos backbone is
implemented once in
:class:`nanocosmos.modules.cosmos_2_5_common.base.BaseCosmosModule`.  The
Cosmos 3 tiers (Edge / Nano / Super) reuse it verbatim, but import it through
this module so the ``cosmos_3_*`` module packages never reference the
``cosmos_2_5_common`` namespace directly -- a thin "soft link".
"""

from nanocosmos.modules.cosmos_2_5_common.base import BaseCosmosModule

__all__ = ["BaseCosmosModule"]
