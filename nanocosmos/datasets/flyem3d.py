"""
FLYEM3D dataset: Janelia FlyEM 8 nm FIB-SEM connectomics (small voxel size).

``FLYEM3D`` is the umbrella source for the FlyEM FIB-SEM volumes that share an
**8 x 8 x 8 nm** voxel and the brainbow HDF5 layout:

* **FIB-25** (Takemura et al. 2015) -- 7-column *Drosophila* medulla;
* **Hemibrain** -- *Drosophila* central brain (~25 k neurons);
* **MaleCNS** -- full male *Drosophila* CNS (v1.0, 2026).

Crops are fetched with ``scripts/download_flyem3d.py`` and written as HDF5
(dataset key ``main``, axis order ``[Z, Y, X]``) -- byte-for-byte the layout
:class:`nanocosmos.datasets.MICRONSDataset` consumes.

This leaf is therefore a thin **metadata override** of
:class:`MICRONSDataset`: the loading / patching / normalisation logic is
shared verbatim, and only the resolution (8x8x8 nm, the *smallest-voxel*
labeled connectomics rung, vs MICrONS' 8x8x40 nm) and label names differ.
"""

from typing import Dict, List

from nanocosmos.datasets.microns import MICRONSDataset


class FLYEM3DDataset(MICRONSDataset):
    """FlyEM 8x8x8 nm FIB-SEM dataset (FIB-25 / Hemibrain / MaleCNS).

    Identical loading + patching to :class:`MICRONSDataset` (HDF5 crops in
    ``[Z, Y, X]`` order, per-volume ``{vol, seg, root, find_boundaries}``
    specs); only the dataset metadata differs.  Download crops with
    ``scripts/download_flyem3d.py``.
    """

    _paper = (
        "FlyEM (Janelia): FIB-25 (Takemura et al. 2015, PNAS, "
        "doi:10.1073/pnas.1509820112); Hemibrain (Scheffer et al. 2020, eLife); "
        "MaleCNS (FlyEM, 2026)."
    )
    # All FlyEM FIB-SEM here are 8 nm in z, y, x (unlike MICrONS' 8x8x40 nm).
    _resolution: Dict[str, float] = {"x": 8.0, "y": 8.0, "z": 8.0}
    _labels_base: List[str] = ["background", "neuron"]


__all__ = ["FLYEM3DDataset"]
