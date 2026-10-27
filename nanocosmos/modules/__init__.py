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
Cosmos 3 shares one :class:`BaseCosmosModule` from
:mod:`nanocosmos.modules.cosmos_2_5_common`; concrete subclasses are
~20-line declarations of which ``model_cls`` and ``loss_cls`` to use.

Public surface
--------------
* :class:`BaseCircuitModule` -- shared training / eval loop.
* :class:`BaseCosmosModule` -- adds the freeze schedule and
  ``dit_backbone_lr`` parameter-group split.
* :class:`Cosmos3NanoModule` (alias :class:`Cosmos3Nano3DModule`) --
  concrete Lightning class for the Cosmos 3 Nano omni backbone (16B).
* :class:`Cosmos3EdgeModule` -- Cosmos 3 Edge (4B).
* :class:`Cosmos3SuperModule` -- Cosmos 3 Super (64B).
* :class:`Joint3DModule` -- joint reconstruction + segmentation recipe
  (:class:`~nanocosmos.losses.Joint3DReconSegLoss`) on Cosmos 3 Nano
  (``model.type: joint3d``).
* :class:`JointEdge3DModule` / :class:`JointSuper3DModule` -- the same
  joint recipe on Cosmos 3 Edge (``joint3d_edge``) and Super
  (``joint3d_super``).

"""

from nanocosmos.modules.base import BaseCircuitModule
from nanocosmos.modules.cosmos_2_5_common import BaseCosmosModule
from nanocosmos.modules.cosmos_3_edge import Cosmos3EdgeModule
from nanocosmos.modules.cosmos_3_nano import Cosmos3Nano3DModule, Cosmos3NanoModule
from nanocosmos.modules.cosmos_3_super import Cosmos3SuperModule
from nanocosmos.modules.joint3d import (
    Joint3DModule,
    JointEdge3DModule,
    JointSuper3DModule,
)

__all__ = [
    "BaseCircuitModule",
    "BaseCosmosModule",
    "Cosmos3EdgeModule",
    "Cosmos3NanoModule",
    "Cosmos3Nano3DModule",
    "Cosmos3SuperModule",
    "Joint3DModule",
    "JointEdge3DModule",
    "JointSuper3DModule",
]
