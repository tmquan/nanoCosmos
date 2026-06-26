"""
Cosmos-Transfer 2.5 Lightning-module package.

Wires the generic :class:`BaseCosmosModule` (lives in
:mod:`nanocosmos.modules.cosmos_2_5_common`) to the concrete 3-D
Lightning module that loads the
:class:`CosmosTransfer3DWrapper` model.

Module layout::

    base.py    -- BaseCosmosModule re-export (back-compat shim)
    module.py  -- CosmosTransfer3DModule (concrete 3-D Lightning module)

The :class:`BaseCosmosModule` class is re-exported here for backward
compatibility with callers / saved checkpoints that import it from
``nanocosmos.modules.cosmos_transfer_2_5``::

    from nanocosmos.modules.cosmos_transfer_2_5 import (
        BaseCosmosModule, CosmosTransfer3DModule,
    )
"""

from nanocosmos.modules.cosmos_2_5_common.base import BaseCosmosModule
from nanocosmos.modules.cosmos_transfer_2_5.module import CosmosTransfer3DModule

__all__ = ["BaseCosmosModule", "CosmosTransfer3DModule"]
