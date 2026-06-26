"""Backwards-compat re-export of :class:`BaseCosmosModule`.

The class itself moved to
:mod:`nanocosmos.modules.cosmos_2_5_common.base` so it can be shared
with :mod:`nanocosmos.modules.cosmos_predict_2_5`.  External callers
(saved checkpoints, downstream notebooks) that still do
``from nanocosmos.modules.cosmos_transfer_2_5 import BaseCosmosModule``
keep working via this re-export.
"""

from nanocosmos.modules.cosmos_2_5_common.base import BaseCosmosModule

__all__ = ["BaseCosmosModule"]
