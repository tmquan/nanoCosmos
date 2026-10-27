"""Shared Lightning-module scaffolding for the Cosmos 3 family.

Houses :class:`BaseCosmosModule`. Nano, Edge, and Super import it
(directly or through ``modules/cosmos_3_common``) for the freeze
schedule, NaN/Inf gradient handling, and the backbone / head
optimiser-group split.
"""

from nanocosmos.modules.cosmos_2_5_common.base import BaseCosmosModule

__all__ = ["BaseCosmosModule"]
