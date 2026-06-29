"""Cosmos3-Nano (16B) Lightning-module package.

Re-exports :class:`BaseCosmosModule` (shared base, lives in
:mod:`nanocosmos.modules.cosmos_3_common`) and the concrete 3-D Lightning
module that wires the :class:`Cosmos3NanoWrapper` model into the base class.
``Cosmos3Nano3DModule`` is kept as a back-compat alias.
"""

from nanocosmos.modules.cosmos_3_common.base import BaseCosmosModule
from nanocosmos.modules.cosmos_3_nano.module import (
    Cosmos3Nano3DModule,
    Cosmos3NanoModule,
)

__all__ = ["BaseCosmosModule", "Cosmos3NanoModule", "Cosmos3Nano3DModule"]
