"""
PyTorch Lightning modules for connectomics segmentation training.

Why this package exists
-----------------------
A Lightning module is the **glue** between a backbone wrapper
(:mod:`nanocosmos.models`), the loss (:class:`nanocosmos.losses.AffinityFGLoss`),
and the Trainer.  Nanocosmos keeps the loop in one shared base
(:class:`BaseCircuitModule`) so that every architecture gets the same
training step, evaluation accumulation, scalar tag hierarchy and
Mutex Watershed agglomeration wiring for free.

Architecture-specific concerns -- freeze schedules, optimiser param-
group splits, gradient sanitisation -- live in a per-family base class.
The Cosmos 2.5 family shares one :class:`BaseCosmosModule` from
:mod:`nanocosmos.modules.cosmos_2_5_common`; concrete subclasses are
~20-line declarations of which ``model_cls`` and ``loss_cls`` to use.

Public surface
--------------
* :class:`BaseCircuitModule` -- shared training / eval loop.
* :class:`BaseCosmosModule` -- adds the freeze schedule and
  ``dit_backbone_lr`` parameter-group split.
* :class:`BaseVistaModule` -- adds Vista-specific wiring.
* :class:`Cosmos3Nano3DModule` -- concrete Lightning class for the
  Cosmos 3 (Nano) omni backbone (16B MoT, no ControlNet).
* :class:`CosmosPredict3DModule` -- concrete Lightning class for the
  Cosmos-Predict 2.5 backbone (base DiT).
* :class:`Vista3DModule` -- concrete Lightning class for the Vista
  backbone (unified aff/sem/raw head).
* :class:`Joint3DModule` -- joint reconstruction + segmentation recipe
  (:class:`~nanocosmos.losses.Joint3DReconSegLoss`) on the Cosmos 3 Nano
  backbone (``model.type: joint3d``).
* :class:`JointPredict3DModule` -- the joint recipe on the Cosmos-Predict
  2.5 2B backbone (``model.type: joint3d_2b`` / ``joint_predict3d``).

Extending this module
---------------------
See ``doc/CONTRIBUTING.md`` "How to add a new model architecture".
"""

from nanocosmos.modules.base import BaseCircuitModule
from nanocosmos.modules.cosmos_2_5_common import BaseCosmosModule
from nanocosmos.modules.cosmos_3_edge import Cosmos3EdgeModule
from nanocosmos.modules.cosmos_3_nano import Cosmos3Nano3DModule, Cosmos3NanoModule
from nanocosmos.modules.cosmos_3_super import Cosmos3SuperModule
from nanocosmos.modules.cosmos_predict_2_5 import CosmosPredict3DModule
from nanocosmos.modules.joint3d import (
    Joint3DModule,
    JointEdge3DModule,
    JointPredict3DModule,
    JointSuper3DModule,
)
from nanocosmos.modules.vista import BaseVistaModule, Vista3DModule

__all__ = [
    "BaseCircuitModule",
    "BaseCosmosModule",
    "BaseVistaModule",
    "Cosmos3EdgeModule",
    "Cosmos3NanoModule",
    "Cosmos3Nano3DModule",
    "Cosmos3SuperModule",
    "CosmosPredict3DModule",
    "Joint3DModule",
    "JointEdge3DModule",
    "JointPredict3DModule",
    "JointSuper3DModule",
    "Vista3DModule",
]
