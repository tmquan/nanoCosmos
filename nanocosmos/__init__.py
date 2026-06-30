"""
Nanocosmos: a PyTorch Lightning infrastructure for connectomics research
with spatially-coloured (nanocosmos-style) instance segmentation targets.

Provides:
- MONAI-compatible dataset classes with a standardised interface
- Preprocessors for common data formats (TIFF, HDF5, NRRD, NIfTI)
- Cosmos model wrappers (DiT + VAE backbone) for volumetric
  segmentation -- Cosmos-Predict 2.5 (2B) and Cosmos3-Nano (16B) are the
  default backbones, with a Vista3D reference
- An affinity + foreground head supervised by ``AffinityFGLoss`` and
  agglomerated into instances at eval/inference by the Mutex Watershed
  (``nanocosmos.inference.mutex_watershed``).
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="The cuda.cudart module is deprecated",
    category=FutureWarning,
)

__version__ = "0.1.0"

from nanocosmos.datasets import (
    CircuitDataset,
    LazyVolDataset,
    SNEMI3DDataset,
    MICRONSDataset,
    FLYEM3DDataset,
    CREMI3DDataset,
    NeuronsDataset,
)
from nanocosmos.preprocessors import (
    BasePreprocessor,
    TIFFPreprocessor,
    HDF5Preprocessor,
    NRRDPreprocessor,
    NFTYPreprocessor,
)
from nanocosmos.datamodules import (
    CircuitDataModule,
    SNEMI3DDataModule,
    MICRONSDataModule,
    FLYEM3DDataModule,
    CREMI3DDataModule,
    NeuronsDataModule,
)
from nanocosmos.losses import AffinityFGLoss, HEAD_CHANNELS, HEAD_LAYOUT, slice_head
from nanocosmos.models import (
    BaseModel,
    Cosmos3Nano3DWrapper,
    CosmosPredict3DWrapper,
    Vista3DWrapper,
)
from nanocosmos.modules import (
    BaseCircuitModule,
    BaseCosmosModule,
    BaseVistaModule,
    Cosmos3Nano3DModule,
    CosmosPredict3DModule,
    Vista3DModule,
)

__all__ = [
    # Data
    "CircuitDataset",
    "LazyVolDataset",
    "SNEMI3DDataset",
    "MICRONSDataset",
    "FLYEM3DDataset",
    "CREMI3DDataset",
    "NeuronsDataset",
    "BasePreprocessor",
    "TIFFPreprocessor",
    "HDF5Preprocessor",
    "NRRDPreprocessor",
    "NFTYPreprocessor",
    "CircuitDataModule",
    "SNEMI3DDataModule",
    "MICRONSDataModule",
    "FLYEM3DDataModule",
    "CREMI3DDataModule",
    "NeuronsDataModule",
    # Losses
    "AffinityFGLoss",
    "HEAD_CHANNELS",
    "HEAD_LAYOUT",
    "slice_head",
    # Models (backbone wrappers)
    "BaseModel",
    "Cosmos3Nano3DWrapper",
    "CosmosPredict3DWrapper",
    "Vista3DWrapper",
    # Modules (Lightning)
    "BaseCircuitModule",
    "BaseCosmosModule",
    "Cosmos3Nano3DModule",
    "CosmosPredict3DModule",
    "BaseVistaModule",
    "Vista3DModule",
]
