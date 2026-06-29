"""Cosmos3-Edge (4B) **3D** Lightning module for volumetric segmentation.

Only the **automatic** training mode is supported.  See
:class:`BaseCosmosModule` for the full training / evaluation logic.  The Edge
backbone is warm-started from Nano (or random-init when ``pretrained=false``);
that logic lives entirely in :class:`Cosmos3EdgeWrapper`, so this module is a
plain ``_model_cls`` declaration.
"""

from nanocosmos.losses import AffinityFGLoss
from nanocosmos.models.cosmos_3_edge import Cosmos3EdgeWrapper
from nanocosmos.modules.cosmos_3_common.base import BaseCosmosModule


class Cosmos3EdgeModule(BaseCosmosModule):
    """Cosmos3-Edge (4B) 3-D volumetric segmentation module."""

    _model_cls = Cosmos3EdgeWrapper
    _loss_cls = AffinityFGLoss


__all__ = ["Cosmos3EdgeModule"]
