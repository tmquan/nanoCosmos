"""
Dataset classes for connectomics research.

Two dataset families live here, with **different** semantics:

* :class:`CircuitDataset` (eager, MONAI ``CacheDataset``-backed) and its
  leaves :class:`SNEMI3DDataset`, :class:`MICRONSDataset`,
  :class:`NeuronsDataset`, :class:`CREMI3DDataset`, :class:`FLYEM3DDataset`
  (the last two are thin metadata subclasses of :class:`MICRONSDataset`)
  -- preload entire volumes into RAM at ``__init__``, then serve crops via
  the MONAI transform pipeline.
  Best for small-to-medium datasets (single-machine, many epochs).
* :class:`LazyVolDataset` -- a thin ``torch.utils.data.Dataset`` that
  reads patches on demand from HDF5 with a thread-local file cache and
  a normalisation-stats probe.  Best for very large volumes where eager
  loading would OOM.

Required overrides for new ``CircuitDataset`` leaves
----------------------------------------------------
* ``paper``, ``resolution``, ``labels`` -- metadata properties.
* ``data_files`` -- dict with ``vol`` (raw image) and optional ``seg``
  (instance label) paths/arrays per volume.
* ``_prepare_data`` -- builds the list of MONAI ``data_dicts`` consumed
  by ``CacheDataset``.

Datamodules in :mod:`nanocosmos.datamodules` choose between eager and
lazy at runtime based on the ``use_lazy`` config flag.

Extending this module
---------------------
See ``doc/CONTRIBUTING.md`` "How to add a new dataset" for a copy-paste
recipe (preprocessor -> leaf dataset -> leaf datamodule -> YAML).
"""

from nanocosmos.datasets.base import CircuitDataset
from nanocosmos.datasets.lazy import LazyVolDataset
from nanocosmos.datasets.snemi3d import SNEMI3DDataset
from nanocosmos.datasets.microns import MICRONSDataset
from nanocosmos.datasets.flyem3d import FLYEM3DDataset
from nanocosmos.datasets.cremi3d import CREMI3DDataset
from nanocosmos.datasets.neurons import NeuronsDataset

__all__ = [
    "CircuitDataset",
    "LazyVolDataset",
    "SNEMI3DDataset",
    "MICRONSDataset",
    "FLYEM3DDataset",
    "CREMI3DDataset",
    "NeuronsDataset",
]
