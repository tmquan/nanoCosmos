"""Cosmos3-Nano (16B) **3D** Lightning module for volumetric segmentation.

Only the **automatic** training mode is supported.  See
:class:`BaseCosmosModule` for the full training / evaluation logic.
"""

from nanocosmos.losses import AffinityFGLoss
from nanocosmos.models.cosmos_3_nano import Cosmos3NanoWrapper
from nanocosmos.modules.cosmos_3_common.base import BaseCosmosModule


class Cosmos3NanoModule(BaseCosmosModule):
    """Cosmos3-Nano (16B) 3-D volumetric segmentation module.

    Like Cosmos-Predict (and unlike Cosmos-Transfer), Cosmos 3 has no
    ControlNet branch, so :meth:`_extra_model_kwargs` returns the empty
    default; everything else (in/head channels, ``freeze_dit_backbone``
    schedule, optimiser param-group split on ``model.dit.*``) is inherited
    from :class:`BaseCosmosModule`.
    """

    _model_cls = Cosmos3NanoWrapper
    _loss_cls = AffinityFGLoss


# Back-compat alias for the historical class name.
Cosmos3Nano3DModule = Cosmos3NanoModule

__all__ = ["Cosmos3NanoModule", "Cosmos3Nano3DModule"]
