"""
Neurons Dataset for neuron segmentation.

Kasthuri et al., Cell 2015 — mouse somatosensory cortex (S1).  The
densely-annotated cylinder from the public ``kasthuri2011`` volume (EM +
ground-truth neuron ids) at **6 × 6 × 30 nm**.  Fetched by
``scripts/download_snemi3d.py --source neurons`` (default crop
5000 × 2900 × 300 at start ``(x=3000, y=7200, z=950)``).

Loading / patching / normalization are identical to
:class:`~nanocosmos.datasets.microns.MICRONSDataset`, so this is a thin
metadata subclass (like :class:`CREMI3DDataset` / :class:`FLYEM3DDataset`)
that only overrides the paper / resolution and restricts I/O to HDF5/TIFF.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from nanocosmos.datasets.microns import MICRONSDataset
from nanocosmos.utils.io import find_folder


class NeuronsDataset(MICRONSDataset):
    """Neurons Dataset for neuron segmentation.

    Volume format: ``[{"vol": "volume_basename", "seg": "seg_basename"}]``

    Optional per-volume keys:
        - ``root``: override ``root_dir`` for this volume.
        - ``find_boundaries``: when > 0, boundary pixels between adjacent
          labels are zeroed out at load time.

    All construction / patching behaviour is inherited from
    :class:`MICRONSDataset`; only the citation, voxel resolution, and the
    (HDF5/TIFF-only) volume loader differ.
    """

    _paper = (
        "Kasthuri, N., et al. (2015). Saturated Reconstruction of a Volume of "
        "Neocortex. Cell, 162(3), 648-661. doi:10.1016/j.cell.2015.06.054"
    )
    _resolution = {"x": 6.0, "y": 6.0, "z": 30.0}

    def _load_volume(
        self,
        base_name: str,
        required: bool = True,
        root_dir: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """Load a volume by basename (HDF5/TIFF only; Neurons ships no NRRD)."""
        search_dir = root_dir if root_dir is not None else self.root_dir
        path = find_folder(search_dir, base_name)
        if path is None:
            if required:
                raise FileNotFoundError(
                    f"Could not find data file '{base_name}' in {search_dir}.\n"
                    f"Expected one of: {base_name}.h5, {base_name}.hdf5, "
                    f"{base_name}.tiff, {base_name}.tif"
                )
            return None
        suffix = path.suffix.lower()
        if suffix in [".h5", ".hdf5"]:
            return self._hdf5_preprocessor.load(str(path))
        return self._tiff_preprocessor.load(str(path))
