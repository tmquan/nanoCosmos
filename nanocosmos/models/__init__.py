"""
Cosmos 3 backbones for connectomics segmentation.

Three end-to-end wrappers live here. Each emits the affinity + sem + raw
head (``HEAD_CHANNELS``):

* :class:`Cosmos3Nano3DWrapper` -- Cosmos 3 Nano (16B) omni
  Mixture-of-Transformers (``Cosmos3OmniTransformer`` + Wan2.2 VAE).
* :class:`Cosmos3EdgeWrapper` -- Cosmos 3 Edge (4B). Loads the released
  ``nvidia/Cosmos3-Edge`` checkpoint directly
  (``nanocosmos-4B.yaml``, ``nanocosmos-4B-pretrain.yaml``).
* :class:`Cosmos3SuperWrapper` -- Cosmos 3 Super (64B).

All three share
:class:`nanocosmos.models.cosmos_3_common.Cosmos3OmniWrapper`. The VAE,
DiT feature hooks, and decoder live in
:mod:`nanocosmos.models.cosmos_2_5_common` and are only used as that
shared scaffold. The dense head is
:class:`nanocosmos.models.cosmos_2_5_common.heads.TaskHead3D`.

The abstract :class:`BaseModel` lays out the common contract every
wrapper honours: ``forward`` returns a single tensor of shape
``[B, HEAD_CHANNELS, ...]`` (``HEAD_CHANNELS = N_AFF + 2``) and
``get_output_channels()`` returns that width. The channel layout is
owned by :data:`nanocosmos.losses.HEAD_LAYOUT`.
"""

from nanocosmos.models.base import BaseModel
from nanocosmos.models.cosmos_3_common import Cosmos3OmniWrapper
from nanocosmos.models.cosmos_3_edge import Cosmos3EdgeWrapper
from nanocosmos.models.cosmos_3_nano import Cosmos3Nano3DWrapper, Cosmos3NanoWrapper
from nanocosmos.models.cosmos_3_super import Cosmos3SuperWrapper

__all__ = [
    "BaseModel",
    "Cosmos3OmniWrapper",
    "Cosmos3EdgeWrapper",
    "Cosmos3NanoWrapper",
    "Cosmos3Nano3DWrapper",
    "Cosmos3SuperWrapper",
]
