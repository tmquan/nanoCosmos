"""
Model architectures for connectomics segmentation.

Four end-to-end backbone wrappers live here:

* :class:`CosmosPredict3DWrapper` -- Cosmos-Predict 2.5 (base DiT +
  VAE, no ControlNet) with the affinity + sem + raw head
  (``HEAD_CHANNELS``).  Backbone of the 2B joint recipe
  (``nanocosmos-2B.yaml``).  Shares all scaffolding with the Cosmos 2.5
  family via :mod:`nanocosmos.models.cosmos_2_5_common`.
* :class:`CosmosTransfer3DWrapper` -- Cosmos-Transfer 2.5 (base DiT +
  ControlNet residual branch + VAE) with the same head.
* :class:`Cosmos3Nano3DWrapper` -- Cosmos 3 (Nano) 16B omni
  Mixture-of-Transformers (``Cosmos3OmniTransformer`` + Wan VAE) with
  the same head.
* :class:`Vista3DWrapper` -- SegResNetDS2 backbone with the same
  head, for fast local iteration.

All wrappers project their backbone features through
:class:`nanocosmos.models.vista.VistaTaskHead3D` so the post-backbone
refinement stack is shared.

The abstract :class:`BaseModel` lays out the common contract every
wrapper honours: ``forward`` returns a single tensor of shape
``[B, HEAD_CHANNELS, ...]`` (the affinity + sem + raw head,
``HEAD_CHANNELS = N_AFF + 2``) and ``get_output_channels()`` returns
that integer width for downstream code that needs to know the head
shape without running a forward pass.  The fixed-slice layout is owned
by :data:`nanocosmos.losses.HEAD_LAYOUT`.
"""

from nanocosmos.models.base import BaseModel
from nanocosmos.models.cosmos_3_common import Cosmos3OmniWrapper
from nanocosmos.models.cosmos_3_edge import Cosmos3EdgeWrapper
from nanocosmos.models.cosmos_3_nano import Cosmos3Nano3DWrapper, Cosmos3NanoWrapper
from nanocosmos.models.cosmos_3_super import Cosmos3SuperWrapper
from nanocosmos.models.cosmos_predict_2_5 import CosmosPredict3DWrapper
from nanocosmos.models.cosmos_transfer_2_5 import CosmosTransfer3DWrapper
from nanocosmos.models.vista import Vista3DWrapper

__all__ = [
    "BaseModel",
    "Cosmos3OmniWrapper",
    "Cosmos3EdgeWrapper",
    "Cosmos3NanoWrapper",
    "Cosmos3Nano3DWrapper",
    "Cosmos3SuperWrapper",
    "CosmosPredict3DWrapper",
    "CosmosTransfer3DWrapper",
    "Vista3DWrapper",
]
