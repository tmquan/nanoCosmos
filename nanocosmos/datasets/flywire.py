"""
FLYWIRE dataset: Princeton/Seung FlyWire FAFB female adult fly brain.

FlyWire crops are fetched with ``scripts/download_flywire.py`` and written
as HDF5 (dataset key ``main``, axis order ``[Z, Y, X]``) -- the same layout
:class:`nanocosmos.datasets.MICRONSDataset` consumes.

This leaf is a thin **metadata override** of :class:`MICRONSDataset`: the
loading / patching / normalisation logic is shared verbatim; only the
resolution (8x8x40 nm at mip 1 EM / upsampled m783 seg) and citation differ.
"""

from typing import Dict, List

from nanocosmos.datasets.microns import MICRONSDataset


class FLYWIREDataset(MICRONSDataset):
    """FlyWire FAFB 8x8x40 nm dataset (materialized release v783).

    Identical loading + patching to :class:`MICRONSDataset` (HDF5 crops in
    ``[Z, Y, X]`` order, per-volume ``{vol, seg, root, find_boundaries}``
    specs); only the dataset metadata differs.  Download crops with
    ``scripts/download_flywire.py``.
    """

    _paper = (
        "FlyWire Consortium (2024). Whole-brain annotation and multi-connectome "
        "cell typing of Drosophila. Nature. "
        "doi:10.1038/s41586-024-07686-5"
    )
    _resolution: Dict[str, float] = {"x": 8.0, "y": 8.0, "z": 40.0}
    _labels_base: List[str] = ["background", "neuron"]


__all__ = ["FLYWIREDataset"]
