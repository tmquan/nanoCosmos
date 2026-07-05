"""
FlyWire (3D) DataModule for PyTorch Lightning.

Thin subclass of :class:`MICRONSDataModule`: FlyWire crops share the
MICRONS HDF5 layout (dataset key ``main``, ``[Z, Y, X]``), so only the
leaf :attr:`dataset_class` differs.
"""

from nanocosmos.datamodules.microns import MICRONSDataModule
from nanocosmos.datasets import FLYWIREDataset


class FLYWIREDataModule(MICRONSDataModule):
    """PyTorch Lightning DataModule for the FlyWire FAFB dataset."""

    dataset_class = FLYWIREDataset


__all__ = ["FLYWIREDataModule"]
