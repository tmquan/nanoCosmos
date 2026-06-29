"""Cosmos3-Super (64B) **3D** Lightning module for volumetric segmentation.

Only the **automatic** training mode is supported.  See
:class:`BaseCosmosModule` for the full training / evaluation logic.
"""

from nanocosmos.losses import AffinityFGLoss
from nanocosmos.models.cosmos_3_super import Cosmos3SuperWrapper
from nanocosmos.modules.cosmos_3_common.base import BaseCosmosModule


class Cosmos3SuperModule(BaseCosmosModule):
    """Cosmos3-Super (64B) 3-D volumetric segmentation module."""

    _model_cls = Cosmos3SuperWrapper
    _loss_cls = AffinityFGLoss


__all__ = ["Cosmos3SuperModule"]
