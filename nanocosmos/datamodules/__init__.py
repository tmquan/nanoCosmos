"""
PyTorch Lightning DataModules for connectomics datasets.

Why this package exists
-----------------------
A datamodule sits between the raw dataset (one of
:mod:`nanocosmos.datasets`) and the Lightning ``Trainer``: it owns the
**MONAI augmentation pipeline**, the **patch / slice sampling
strategy**, and the **DataLoader** configuration (batch size, workers,
``forkserver`` multiprocessing context).  All architectures share the
same datamodule contract -- swapping the backbone does not require a
new datamodule.

Public surface
--------------
* :class:`CircuitDataModule` -- shared base.  Implements
  ``get_train_transforms`` / ``get_val_transforms`` / ``setup`` and the
  three dataloader hooks; subclasses only declare ``dataset_class`` and
  any per-leaf overrides.
* :class:`SNEMI3DDataModule`, :class:`MICRONSDataModule`,
  :class:`NeuronsDataModule`, :class:`CREMI3DDataModule`,
  :class:`FLYEM3DDataModule` -- one leaf per dataset (the last two are
  thin metadata subclasses of :class:`MICRONSDataModule`).

Extending this module
---------------------
A new datamodule is typically a 50-line subclass declaring its
``dataset_class``.  See ``doc/CONTRIBUTING.md`` "How to add a new
dataset" for the full recipe.
"""

from nanocosmos.datamodules.base import CircuitDataModule
from nanocosmos.datamodules.snemi3d import SNEMI3DDataModule
from nanocosmos.datamodules.microns import MICRONSDataModule
from nanocosmos.datamodules.flyem3d import FLYEM3DDataModule
from nanocosmos.datamodules.cremi3d import CREMI3DDataModule
from nanocosmos.datamodules.neurons import NeuronsDataModule
from nanocosmos.datamodules.joint3d import Joint3DDataModule

__all__ = [
    "CircuitDataModule",
    "SNEMI3DDataModule",
    "MICRONSDataModule",
    "FLYEM3DDataModule",
    "CREMI3DDataModule",
    "NeuronsDataModule",
    "Joint3DDataModule",
]
