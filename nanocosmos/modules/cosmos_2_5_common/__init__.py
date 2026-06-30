"""Shared Lightning-module scaffolding for the Cosmos 2.5 family.

Houses :class:`BaseCosmosModule` -- the abstract base every
Cosmos 2.5 backbone module inherits from.

The subclass lives next door under
:mod:`nanocosmos.modules.cosmos_predict_2_5`; it imports this base
class to get the freeze schedule, NaN/Inf gradient handling, and the
backbone / heads optimiser-group split for free.
"""

from nanocosmos.modules.cosmos_2_5_common.base import BaseCosmosModule

__all__ = ["BaseCosmosModule"]
