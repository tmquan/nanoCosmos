"""
FlyEM (3D) DataModule for PyTorch Lightning.

Thin subclass of :class:`MICRONSDataModule`: FlyEM crops (FIB-25,
Hemibrain, MaleCNS) share the MICRONS HDF5 layout (dataset key ``main``,
``[Z, Y, X]``), so only the leaf :attr:`dataset_class` differs.  Lazy
3-D patch mode (``slice_mode: false`` with ``patch_size``) reads patches
on demand via :class:`nanocosmos.datasets.LazyVolDataset`.
"""

import logging

from nanocosmos.datamodules.microns import MICRONSDataModule
from nanocosmos.datasets import FLYEM3DDataset

logger = logging.getLogger(__name__)


class FLYEM3DDataModule(MICRONSDataModule):
    """PyTorch Lightning DataModule for FlyEM 8 nm FIB-SEM
    (FIB-25 / Hemibrain / MaleCNS).

    Inherits the full MICrONS datamodule behaviour (eager slice mode /
    lazy 3-D patch mode, the shared MONAI augmentation pipeline, and the
    dataloader hooks); only the underlying :attr:`dataset_class` is
    swapped to :class:`FLYEM3DDataset`.
    """

    dataset_class = FLYEM3DDataset


__all__ = ["FLYEM3DDataModule"]
