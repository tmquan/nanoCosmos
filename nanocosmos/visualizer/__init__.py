"""
Web-based volume visualizer with a neuroglancer-style 4-panel layout.

Why this package exists
-----------------------
TensorBoard images are great for **per-step debugging** but a full
volume in 3-D is much easier to skim in an interactive viewer.  The
visualizer is a small FastAPI server.  The browser downloads a
downsampled HDF5 / NRRD / TIFF / NIfTI / NumPy volume, draws three
orthogonal slice panels on 2-D canvases, and renders selected segments
in the fourth panel as a WebGL Gaussian-splat point cloud plus three
slice planes.  Use it to QA datasets and predictions; do not depend on
it from training.

Public surface
--------------
* :class:`VolumeData` -- tiny container for a loaded volume + spacing.
* :func:`load_volume` -- format-agnostic loader (HDF5 / NRRD / TIFF /
  NIfTI) backed by :mod:`nanocosmos.preprocessors`.

The FastAPI app and CLI live in :mod:`nanocosmos.visualizer.app` and
:mod:`nanocosmos.visualizer.__main__` respectively; launch with::

    python -m nanocosmos.visualizer --raw vol.h5 --seg seg.h5

Extending this module
---------------------
To support a new format, add the corresponding
:class:`nanocosmos.preprocessors.BasePreprocessor` subclass; the visualizer
will pick it up via :func:`load_volume` automatically.
"""

from nanocosmos.visualizer.volume_loader import VolumeData, load_volume

__all__ = ["VolumeData", "load_volume"]
