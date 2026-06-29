"""Shared Cosmos 3 family Lightning-module package.

Re-exports :class:`BaseCosmosModule` (the shared base, which lives in
:mod:`nanocosmos.modules.cosmos_2_5_common`).  The concrete per-tier Lightning
modules live in the sibling packages
:mod:`nanocosmos.modules.cosmos_3_edge` / ``...cosmos_3_nano`` /
``...cosmos_3_super`` and are ~10-line declarations of which ``model_cls`` to
wire to the shared loop.
"""

from nanocosmos.modules.cosmos_3_common.base import BaseCosmosModule

__all__ = ["BaseCosmosModule"]
