"""Small shared sampling helpers for the connectomics transform pipeline."""

import numpy as np


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Sample log-uniformly from ``[lo, hi]`` (scale-symmetric).

    Shared by :mod:`~nanocosmos.transforms.degrade` (voxel-size factor) and
    :mod:`~nanocosmos.transforms.resolution_zoom` (target-resolution jitter).
    """
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
